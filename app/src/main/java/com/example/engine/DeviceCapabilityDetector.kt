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
import com.example.model.DeviceCapabilities
import com.example.model.DeviceProfileTier
import com.example.model.DeviceProfilesRoot
import com.example.model.ThermalState
import kotlinx.serialization.json.Json
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
        val cpuCores = Runtime.getRuntime().availableProcessors().coerceAtLeast(2)

        val abis = Build.SUPPORTED_ABIS?.toList().orEmpty()
        val primaryAbi = abis.firstOrNull() ?: "unknown"

        val pm = context.packageManager
        val vulkanSupported = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            pm.hasSystemFeature(PackageManager.FEATURE_VULKAN_HARDWARE_LEVEL)
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
            "CPU/OpenGL"
        }

        val configInfo = activityManager.deviceConfigurationInfo
        val openGlMajor = (configInfo.reqGlEsVersion ushr 16).coerceAtLeast(2)
        val nnapiSupported = Build.VERSION.SDK_INT >= Build.VERSION_CODES.P

        val hardwareH264 = hasHardwareEncoder(MediaFormat.MIMETYPE_VIDEO_AVC)
        val hardwareH265 = hasHardwareEncoder(MediaFormat.MIMETYPE_VIDEO_HEVC)

        val statFs = StatFs(context.cacheDir.absolutePath)
        val availableStorageMb = (statFs.availableBlocksLong * statFs.blockSizeLong) / (1024L * 1024L)

        val (thermalState, tempCelsius) = readThermalInfo()

        val baseTier = cachedTiers.firstOrNull { totalRamMb >= it.minRamMb && totalRamMb < it.maxRamMb }
            ?: cachedTiers.last()

        // Never rely on RAM alone: calculate holistic processing capacity score (0..100)
        val ramScore = ((totalRamMb.coerceIn(2048L, 12288L) - 2048L) * 40f / 10240f)
        val cpuScore = ((cpuCores.coerceIn(2, 8) - 2) * 25f / 6f)
        val gpuScore = if (vulkanSupported) 20f else if (openGlMajor >= 3) 10f else 0f
        val headroomScore = if (availableRamMb > 2500L) 15f else if (availableRamMb > 1200L) 8f else 2f
        val capacityScore = (ramScore + cpuScore + gpuScore + headroomScore).roundToInt().coerceIn(10, 100)

        // Adapt tile size by BOTH holistic capacity, available RAM, and thermal state
        var tileSize = when {
            capacityScore >= 82 && availableRamMb >= 3000L -> baseTier.tileSize
            capacityScore >= 55 && availableRamMb >= 1800L -> baseTier.tileSize.coerceAtMost(192)
            capacityScore >= 35 && availableRamMb >= 1000L -> baseTier.tileSize.coerceAtMost(128)
            else -> 96
        }
        if (thermalState == ThermalState.THROTTLING || thermalState == ThermalState.CRITICAL) {
            tileSize = tileSize.coerceAtMost(128)
        }

        val optimalThreads = when (thermalState) {
            ThermalState.CRITICAL -> 1
            ThermalState.THROTTLING -> (cpuCores / 2).coerceIn(1, 2)
            ThermalState.WARM -> (cpuCores / 2).coerceIn(2, baseTier.maxInferenceThreads)
            ThermalState.NORMAL -> (cpuCores - 1).coerceIn(2, baseTier.maxInferenceThreads)
        }

        val batchSize = when {
            availableRamMb < 900L || thermalState >= ThermalState.THROTTLING -> 1
            else -> baseTier.maxBatchFrames
        }

        return DeviceCapabilities(
            tierName = baseTier.name,
            totalRamMb = totalRamMb,
            availableRamMb = availableRamMb,
            cpuCores = cpuCores,
            primaryAbi = primaryAbi,
            supportedAbis = abis,
            vulkanSupported = vulkanSupported,
            vulkanVersionLabel = vulkanVersionLabel,
            nnapiSupported = nnapiSupported,
            openGlEsMajor = openGlMajor,
            hardwareH264Supported = hardwareH264,
            hardwareH265Supported = hardwareH265,
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
            processingCapacityScore = capacityScore
        )
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

    private fun loadTiersFromAssets(): List<DeviceProfileTier> {
        return try {
            val raw = context.assets.open("configuration/device_profiles.json").bufferedReader().use { it.readText() }
            json.decodeFromString<DeviceProfilesRoot>(raw).tiers
        } catch (_: Exception) {
            listOf(
                DeviceProfileTier("LOW_4GB", 0, 4608, 96, 8, 1, 2, false, 350, 45, 140),
                DeviceProfileTier("MID_6GB", 4608, 6656, 128, 12, 2, 4, true, 450, 30, 110),
                DeviceProfileTier("HIGH_8GB", 6656, 10240, 192, 16, 4, 4, true, 600, 20, 90),
                DeviceProfileTier("ULTRA_12GB_PLUS", 10240, 999999, 256, 16, 6, 6, true, 750, 10, 75)
            )
        }
    }
}
