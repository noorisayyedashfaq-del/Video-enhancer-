package com.example.engine

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.media.MediaCodecInfo
import android.media.MediaCodecList
import android.media.MediaFormat
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import android.os.StatFs
import com.example.model.CpuTopologyInfo
import com.example.model.DeviceCapabilities
import com.example.model.DeviceProfileTier
import com.example.model.DeviceProfilesRoot
import com.example.model.HardwareProbeSnapshot
import com.example.model.ThermalState
import kotlinx.serialization.json.Json
import java.io.File
import kotlin.math.roundToInt

class DeviceCapabilityDetector(private val context: Context) {

    private val json = Json { ignoreUnknownKeys = true }
    private val cachedTiers: List<DeviceProfileTier> by lazy { loadTiersFromAssets() }

    fun detect(): DeviceCapabilities {
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)

        val totalRamMb = (memInfo.totalMem / (1024L * 1024L)).coerceAtLeast(1024L)
        val availableRamMb = (memInfo.availMem / (1024L * 1024L)).coerceAtLeast(256L)
        val cpuTopology = detectCpuTopology()
        val cpuCores = cpuTopology.totalCores

        val abis = Build.SUPPORTED_ABIS?.toList().orEmpty()
        val primaryAbi = abis.firstOrNull() ?: "unknown"

        val probeSnapshot = probeHardwareSnapshot(
            activityManager = activityManager,
            cpuTopology = cpuTopology
        )

        val hardwareH264 = hasHardwareEncoder(MediaFormat.MIMETYPE_VIDEO_AVC)
        val hardwareH265 = hasHardwareEncoder(MediaFormat.MIMETYPE_VIDEO_HEVC)

        val statFs = StatFs(context.cacheDir.absolutePath)
        val availableStorageMb = (statFs.availableBlocksLong * statFs.blockSizeLong) / (1024L * 1024L)

        val (thermalState, tempCelsius) = readThermalInfo()

        val baseTier = cachedTiers.firstOrNull { totalRamMb >= it.minRamMb && totalRamMb < it.maxRamMb }
            ?: cachedTiers.last()

        // Holistic capacity score factoring RAM, big cores, GPU compute, and thermal headroom
        val ramScore = ((totalRamMb.coerceIn(2048L, 12288L) - 2048L) * 38f / 10240f)
        val cpuScore = ((cpuTopology.bigCoreCount.coerceIn(1, 6) - 1) * 24f / 5f)
        val gpuScore = when {
            probeSnapshot.qnnGpuAvailable || probeSnapshot.vulkanComputeSupported -> 23f
            probeSnapshot.vulkanSupported || probeSnapshot.openClSupported -> 18f
            probeSnapshot.openGlEsMajor >= 3 -> 11f
            else -> 0f
        }
        val headroomScore = if (availableRamMb > 2500L) 15f else if (availableRamMb > 1200L) 8f else 2f
        val capacityScore = (ramScore + cpuScore + gpuScore + headroomScore).roundToInt().coerceIn(10, 100)

        var tileSize = when {
            capacityScore >= 82 && availableRamMb >= 3000L -> baseTier.tileSize
            capacityScore >= 55 && availableRamMb >= 1800L -> baseTier.tileSize.coerceAtMost(192)
            capacityScore >= 35 && availableRamMb >= 1000L -> baseTier.tileSize.coerceAtMost(128)
            else -> 96
        }
        if (thermalState == ThermalState.THROTTLING || thermalState == ThermalState.CRITICAL) {
            tileSize = tileSize.coerceAtMost(128)
        }

        // Default inference threads prioritize big-core cluster count cleanly while honoring thermal caps
        val preferredBig = cpuTopology.bigCoreCount.coerceIn(2, baseTier.maxInferenceThreads)
        val optimalThreads = when (thermalState) {
            ThermalState.CRITICAL -> 2
            ThermalState.THROTTLING -> (preferredBig / 2).coerceIn(2, 4)
            ThermalState.WARM -> (preferredBig - 1).coerceIn(2, baseTier.maxInferenceThreads)
            ThermalState.NORMAL -> preferredBig
        }

        val batchSize = when {
            availableRamMb < 900L || thermalState >= ThermalState.THROTTLING -> 1
            else -> baseTier.maxBatchFrames
        }

        val maxEncodeEdge = maxOf(
            if (hardwareH265) queryMaxEncoderLongEdge(MediaFormat.MIMETYPE_VIDEO_HEVC) else 0,
            if (hardwareH264) queryMaxEncoderLongEdge(MediaFormat.MIMETYPE_VIDEO_AVC) else 0,
            1920
        ).coerceIn(1920, 4096)

        return DeviceCapabilities(
            tierName = baseTier.name,
            totalRamMb = totalRamMb,
            availableRamMb = availableRamMb,
            cpuCores = cpuCores,
            primaryAbi = primaryAbi,
            supportedAbis = abis,
            vulkanSupported = probeSnapshot.vulkanSupported,
            vulkanVersionLabel = probeSnapshot.vulkanVersionLabel,
            nnapiSupported = probeSnapshot.nnapiSupported,
            openGlEsMajor = probeSnapshot.openGlEsMajor,
            hardwareH264Supported = hardwareH264,
            hardwareH265Supported = hardwareH265,
            maxEncodeLongEdge = maxEncodeEdge,
            availableStorageMb = availableStorageMb,
            thermalState = thermalState,
            temperatureCelsius = tempCelsius,
            adaptiveTileSize = tileSize,
            tileOverlap = baseTier.tileOverlap,
            batchSize = batchSize,
            inferenceThreads = optimalThreads,
            minRequiredFreeStorageMb = baseTier.minRequiredFreeStorageMb,
            throttleDelayMsOnWarm = baseTier.throttleDelayMsOnWarm,
            throttleDelayMsOnHot = baseTier.throttleDelayMsOnHot,
            processingCapacityScore = capacityScore,
            vulkanComputeSupported = probeSnapshot.vulkanComputeSupported,
            openClSupported = probeSnapshot.openClSupported,
            qnnSupported = probeSnapshot.qnnHtpAvailable || probeSnapshot.qnnGpuAvailable,
            cpuTopology = cpuTopology,
            hardwareProbeSnapshot = probeSnapshot
        )
    }

    fun probeHardwareSnapshot(
        activityManager: ActivityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager,
        cpuTopology: CpuTopologyInfo = detectCpuTopology()
    ): HardwareProbeSnapshot {
        val pm = context.packageManager
        val vulkanSupported = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            pm.hasSystemFeature(PackageManager.FEATURE_VULKAN_HARDWARE_LEVEL)
        } else {
            false
        }
        val vulkanComputeSupported = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            pm.hasSystemFeature(PackageManager.FEATURE_VULKAN_HARDWARE_COMPUTE)
        } else {
            false
        }
        val vulkanVersionLabel = if (vulkanSupported && Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            val verInfo = pm.systemAvailableFeatures.firstOrNull {
                it.name == PackageManager.FEATURE_VULKAN_HARDWARE_VERSION
            }
            if (verInfo != null) {
                val major = (verInfo.version ushr 22) and 0x3FF
                val minor = (verInfo.version ushr 12) and 0x3FF
                "Vulkan $major.$minor"
            } else {
                "Vulkan 1.x"
            }
        } else {
            "Unavailable"
        }

        val configInfo = activityManager.deviceConfigurationInfo
        val openGlMajor = (configInfo.reqGlEsVersion ushr 16).coerceAtLeast(2)

        val (openClPresent, openClSource) = detectOpenClDriver()
        val (isQualcomm, qnnHtp, qnnGpu, qnnLabel) = detectQualcommQnnDrivers()

        val nnapiSupported = Build.VERSION.SDK_INT >= Build.VERSION_CODES.P
        val (nnapiGpu, nnapiNpu, nnapiSummary) = classifyNnapiExpectation(
            nnapiSupported = nnapiSupported,
            vulkanSupported = vulkanSupported,
            openClPresent = openClPresent,
            isQualcomm = isQualcomm
        )

        return HardwareProbeSnapshot(
            vulkanSupported = vulkanSupported,
            vulkanVersionLabel = vulkanVersionLabel,
            vulkanComputeSupported = vulkanComputeSupported,
            openClSupported = openClPresent,
            openClSourceLabel = openClSource,
            openGlEsMajor = openGlMajor,
            nnapiSupported = nnapiSupported,
            nnapiGpuDetected = nnapiGpu,
            nnapiNpuDetected = nnapiNpu,
            nnapiDriversSummary = nnapiSummary,
            isQualcommSoc = isQualcomm,
            qnnHtpAvailable = qnnHtp,
            qnnGpuAvailable = qnnGpu,
            qnnStatusLabel = qnnLabel,
            cpuTopology = cpuTopology
        )
    }

    fun detectCpuTopology(): CpuTopologyInfo {
        val totalCores = Runtime.getRuntime().availableProcessors().coerceAtLeast(2)
        val maxFreqsKhz = ArrayList<Int>(totalCores)
        for (idx in 0 until totalCores) {
            val f = File("/sys/devices/system/cpu/cpu$idx/cpufreq/cpuinfo_max_freq")
            val khz = runCatching {
                if (f.exists() && f.canRead()) f.readText().trim().toIntOrNull() else null
            }.getOrNull()
            if (khz != null && khz > 0) {
                maxFreqsKhz.add(khz)
            }
        }

        if (maxFreqsKhz.size < 2) {
            val fallbackBig = (totalCores / 2).coerceIn(2, totalCores)
            return CpuTopologyInfo(
                totalCores = totalCores,
                bigCoreCount = fallbackBig,
                littleCoreCount = (totalCores - fallbackBig).coerceAtLeast(0),
                maxClockMhz = 0,
                clusterSummary = "${totalCores}T (Uniform/Unknown)"
            )
        }

        val maxKhz = maxFreqsKhz.maxOrNull() ?: 1
        val bigThreshold = (maxKhz * 0.72).toInt()
        val bigCount = maxFreqsKhz.count { it >= bigThreshold }.coerceIn(1, totalCores)
        val littleCount = (totalCores - bigCount).coerceAtLeast(0)
        val maxMhz = maxKhz / 1000
        return CpuTopologyInfo(
            totalCores = totalCores,
            bigCoreCount = bigCount,
            littleCoreCount = littleCount,
            maxClockMhz = maxMhz,
            clusterSummary = "${bigCount} Big + ${littleCount} Little (${maxMhz}MHz)"
        )
    }

    private fun detectOpenClDriver(): Pair<Boolean, String> {
        val candidates = listOf(
            "/vendor/lib64/libOpenCL.so" to "Vendor OpenCL 64",
            "/system/vendor/lib64/libOpenCL.so" to "System Vendor OpenCL 64",
            "/system/lib64/libOpenCL.so" to "System OpenCL 64",
            "/vendor/lib64/egl/libGLES_mali.so" to "ARM Mali OpenCL/EGL",
            "/vendor/lib64/libOpenCL_adreno.so" to "Adreno OpenCL",
            "/vendor/lib/libOpenCL.so" to "Vendor OpenCL 32"
        )
        for ((path, label) in candidates) {
            try {
                if (File(path).exists()) return true to label
            } catch (_: Throwable) {
            }
        }
        return false to "Not found"
    }

    private data class QnnProbeResult(
        val isQualcomm: Boolean,
        val htpAvailable: Boolean,
        val gpuAvailable: Boolean,
        val label: String
    )

    private fun detectQualcommQnnDrivers(): QnnProbeResult {
        val socMfg = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            runCatching { Build.SOC_MANUFACTURER.orEmpty() }.getOrDefault("")
        } else {
            ""
        }
        val socModel = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            runCatching { Build.SOC_MODEL.orEmpty() }.getOrDefault("")
        } else {
            ""
        }
        val hwStr = "${Build.HARDWARE} ${Build.BOARD} $socMfg $socModel".lowercase()
        val isQcom = hwStr.contains("qcom") || hwStr.contains("qualcomm") || hwStr.contains("snapdragon") || hwStr.contains("sm8") || hwStr.contains("sm7")

        val htpPaths = listOf(
            "${context.applicationInfo.nativeLibraryDir}/libQnnHtp.so",
            "/vendor/lib64/libQnnHtp.so",
            "/system/vendor/lib64/libQnnHtp.so"
        )
        val gpuPaths = listOf(
            "${context.applicationInfo.nativeLibraryDir}/libQnnGpu.so",
            "/vendor/lib64/libQnnGpu.so",
            "/system/vendor/lib64/libQnnGpu.so"
        )
        val hasHtp = htpPaths.any { runCatching { File(it).exists() }.getOrDefault(false) }
        val hasGpu = gpuPaths.any { runCatching { File(it).exists() }.getOrDefault(false) }
        val ortHasQnn = runCatching {
            ai.onnxruntime.OrtEnvironment.getAvailableProviders().any { it.name.contains("QNN", ignoreCase = true) }
        }.getOrDefault(false)

        val label = when {
            ortHasQnn && hasHtp && hasGpu -> "QNN HTP NPU + GPU Ready"
            ortHasQnn && hasHtp -> "QNN HTP NPU Ready"
            ortHasQnn && hasGpu -> "QNN GPU Ready"
            isQcom && (hasHtp || hasGpu) -> "Snapdragon Vendor QNN Libs"
            isQcom -> "Snapdragon (NNAPI Fallback)"
            else -> "Non-Qualcomm SoC"
        }
        return QnnProbeResult(
            isQualcomm = isQcom,
            htpAvailable = ortHasQnn && hasHtp,
            gpuAvailable = ortHasQnn && hasGpu,
            label = label
        )
    }

    private fun classifyNnapiExpectation(
        nnapiSupported: Boolean,
        vulkanSupported: Boolean,
        openClPresent: Boolean,
        isQualcomm: Boolean
    ): Triple<Boolean, Boolean, String> {
        if (!nnapiSupported) return Triple(false, false, "API < 28 (Unsupported)")
        val hw = "${Build.HARDWARE} ${Build.BOARD} ${Build.MANUFACTURER}".lowercase()
        val hasNpuHint = Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q && (
            isQualcomm || hw.contains("mt") || hw.contains("dimensity") ||
                hw.contains("exynos") || hw.contains("s5e") || hw.contains("tensor") ||
                hw.contains("kirin") || !hw.contains("ranchu") && !hw.contains("goldfish")
            )
        val hasGpuHint = vulkanSupported || openClPresent
        val summary = buildList {
            if (hasGpuHint) add("GPU Delegate")
            if (hasNpuHint) add("NPU/DSP Accelerator")
        }.joinToString(" + ").ifBlank { "CPU Reference Only" }
        return Triple(hasGpuHint, hasNpuHint, summary)
    }

    fun readThermalInfo(): Pair<ThermalState, Float> {
        val batteryIntent = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val tempTenths = batteryIntent?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 300) ?: 300
        val tempC = tempTenths / 10f

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val powerManager = context.getSystemService(Context.POWER_SERVICE) as PowerManager
            val state = when (powerManager.currentThermalStatus) {
                PowerManager.THERMAL_STATUS_NONE,
                PowerManager.THERMAL_STATUS_LIGHT -> if (tempC >= 41f) ThermalState.WARM else ThermalState.NORMAL
                PowerManager.THERMAL_STATUS_MODERATE -> ThermalState.WARM
                PowerManager.THERMAL_STATUS_SEVERE -> ThermalState.THROTTLING
                PowerManager.THERMAL_STATUS_CRITICAL,
                PowerManager.THERMAL_STATUS_EMERGENCY,
                PowerManager.THERMAL_STATUS_SHUTDOWN -> ThermalState.CRITICAL
                else -> ThermalState.NORMAL
            }
            return state to tempC
        }

        val fallbackState = when {
            tempC >= 46f -> ThermalState.CRITICAL
            tempC >= 43f -> ThermalState.THROTTLING
            tempC >= 39.5f -> ThermalState.WARM
            else -> ThermalState.NORMAL
        }
        return fallbackState to tempC
    }

    fun currentAvailableRamMb(): Long {
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)
        return memInfo.availMem / (1024L * 1024L)
    }

    fun currentAvailableStorageMb(): Long {
        val statFs = StatFs(context.cacheDir.absolutePath)
        return (statFs.availableBlocksLong * statFs.blockSizeLong) / (1024L * 1024L)
    }

    private fun hasHardwareEncoder(mimeType: String): Boolean {
        return try {
            val list = MediaCodecList(MediaCodecList.REGULAR_CODECS)
            list.codecInfos.any { info ->
                info.isEncoder &&
                    info.supportedTypes.any { it.equals(mimeType, ignoreCase = true) } &&
                    isLikelyHardwareCodec(info)
            }
        } catch (_: Exception) {
            false
        }
    }

    private fun isLikelyHardwareCodec(info: MediaCodecInfo): Boolean {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            return info.isHardwareAccelerated
        }
        val name = info.name.lowercase()
        return !name.startsWith("omx.google.") && !name.startsWith("c2.android.")
    }

    private fun queryMaxEncoderLongEdge(mimeType: String): Int {
        return try {
            val list = MediaCodecList(MediaCodecList.REGULAR_CODECS)
            var best = 1920
            for (info in list.codecInfos) {
                if (!info.isEncoder || !info.supportedTypes.any { it.equals(mimeType, ignoreCase = true) }) continue
                val caps = info.getCapabilitiesForType(mimeType)?.videoCapabilities ?: continue
                val maxW = caps.supportedWidths.upper
                val maxH = caps.supportedHeights.upper
                if (maxW >= 3840 && caps.isSizeSupported(3840, 2160)) {
                    best = maxOf(best, 3840)
                } else if (maxW >= 2560 && caps.isSizeSupported(2560, 1440)) {
                    best = maxOf(best, 2560)
                } else if (maxH >= 3840 && caps.isSizeSupported(2160, 3840)) {
                    best = maxOf(best, 3840)
                }
            }
            best
        } catch (_: Throwable) {
            1920
        }
    }

    private fun loadTiersFromAssets(): List<DeviceProfileTier> {
        return try {
            val raw = context.assets.open("configuration/device_profiles.json").bufferedReader().use { it.readText() }
            json.decodeFromString<DeviceProfilesRoot>(raw).tiers
        } catch (_: Exception) {
            listOf(
                DeviceProfileTier("LOW_4GB", 0, 4608, 96, 8, 1, 4, true, 350, 45, 140),
                DeviceProfileTier("MID_6GB", 4608, 6656, 128, 12, 2, 6, true, 450, 30, 110),
                DeviceProfileTier("HIGH_8GB", 6656, 10240, 192, 16, 4, 8, true, 600, 20, 90),
                DeviceProfileTier("ULTRA_12GB_PLUS", 10240, 999999, 256, 16, 6, 8, true, 750, 10, 75)
            )
        }
    }

    companion object {
        /**
         * Returns candidate thread counts for XNNPACK benchmarking (4, 6, 8 where supported by CPU cores).
         */
        fun benchmarkThreadCandidates(cpuCores: Int, maxTierThreads: Int): List<Int> {
            val safeCores = cpuCores.coerceAtLeast(2)
            val ceiling = minOf(safeCores, maxTierThreads.coerceAtLeast(4))
            val base = listOf(4, 6, 8).filter { it <= ceiling }
            return base.ifEmpty { listOf(safeCores.coerceIn(2, 4)) }
        }
    }
}
