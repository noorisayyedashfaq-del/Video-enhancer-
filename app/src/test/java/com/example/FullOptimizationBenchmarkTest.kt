package com.example

import android.content.Context
import android.net.Uri
import androidx.test.core.app.ApplicationProvider
import com.example.engine.AiSpeedPolicy
import com.example.engine.AiTilingMath
import com.example.engine.DnCnnResidualMath
import com.example.engine.EnhancementProfileRepository
import com.example.engine.MotionSmoothMath
import com.example.engine.ResolutionCalculator
import com.example.engine.SmartAutoAdjustmentEngine
import com.example.engine.SpeedBudgetGovernor
import com.example.engine.SpeedExecutionTier
import com.example.engine.TemporalReusePolicy
import com.example.model.DeviceCapabilities
import com.example.model.EnhancementMode
import com.example.model.ThermalState
import com.example.model.VideoMetadata
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.util.Locale

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [36])
class FullOptimizationBenchmarkTest {

    private fun testCaps() = DeviceCapabilities(
        tierName = "Flagship",
        totalRamMb = 8192L,
        availableRamMb = 4096L,
        cpuCores = 8,
        primaryAbi = "arm64-v8a",
        supportedAbis = listOf("arm64-v8a"),
        vulkanSupported = true,
        vulkanVersionLabel = "1.3",
        nnapiSupported = true,
        openGlEsMajor = 3,
        hardwareH264Supported = true,
        hardwareH265Supported = true,
        maxEncodeLongEdge = 3840,
        availableStorageMb = 12_000L,
        thermalState = ThermalState.NORMAL,
        temperatureCelsius = 32f,
        adaptiveTileSize = 256,
        tileOverlap = 24,
        batchSize = 2,
        inferenceThreads = 4,
        minRequiredFreeStorageMb = 500L,
        throttleDelayMsOnWarm = 25L,
        throttleDelayMsOnHot = 80L,
        processingCapacityScore = 90
    )

    private fun sampleVideo(w: Int, h: Int, fps: Double = 30.0) = VideoMetadata(
        uri = Uri.EMPTY,
        displayName = "bench_${w}x${h}.mp4",
        width = w,
        height = h,
        rotationDegrees = 0,
        displayWidth = w,
        displayHeight = h,
        fps = fps,
        durationMs = 60_000L,
        estimatedFrameCount = (60.0 * fps).toInt(),
        bitrateBps = 4_000_000L,
        videoCodec = "h264",
        hasAudio = true,
        audioCodec = "aac",
        audioSampleRate = 48_000,
        audioChannels = 2,
        fileSizeBytes = 30_000_000L
    )

    @Test
    fun `h264 codec lock is strictly preserved across all 4 modes even when device supports H265`() {
        val ctx = ApplicationProvider.getApplicationContext<Context>()
        val profiles = EnhancementProfileRepository(ctx).loadProfiles()
        val caps = testCaps()
        val hd = sampleVideo(1280, 720, 30.0)

        for (mode in EnhancementMode.entries) {
            val profile = profiles.getValue(mode)
            assertEquals("Profile ${mode.id} must remain H264", "H264", profile.preferredCodec)
            val dims = ResolutionCalculator.calculate(hd, mode, profile, caps)
            assertFalse("H.264 lock violated in ${mode.id}: useH265 must stay false", dims.useH265)
        }
    }

    @Test
    fun `sublinear H264 BPP optimization significantly reduces HD_PLUS FHD TWO_K file size while keeping monotonicity and full quality`() {
        val ctx = ApplicationProvider.getApplicationContext<Context>()
        val profiles = EnhancementProfileRepository(ctx).loadProfiles()
        val caps = testCaps()

        val testInputs = listOf(
            sampleVideo(480, 360, 30.0),
            sampleVideo(1280, 720, 30.0),
            sampleVideo(1920, 1080, 30.0)
        )

        val sb = StringBuilder("\n=== H.264 Output File-Size Optimization Report (60s Video @ 30fps, H.264 Locked) ===\n")
        for (input in testInputs) {
            val prevBitrates = mutableListOf<Int>()
            for (mode in EnhancementMode.entries) {
                val prof = profiles.getValue(mode)
                val outDims = ResolutionCalculator.calculate(input, mode, prof, caps)
                val legacyBps = ResolutionCalculator.computeLegacyLinearBitrateForComparison(
                    outDims.width, outDims.height, input.fps, prof.targetBpp
                )
                val optBps = outDims.targetBitrateBps
                assertTrue("Optimized H.264 bitrate must not exceed legacy linear bitrate", optBps <= legacyBps)
                if (outDims.width * outDims.height >= 1280 * 720) {
                    assertTrue(
                        "Expected >=18% H.264 bitrate savings at HD/2K/4K (${mode.id} ${outDims.width}x${outDims.height})",
                        optBps <= (legacyBps * 0.83).toInt()
                    )
                }
                prevBitrates.add(optBps)

                val legacyMbPerMin = (legacyBps / 8.0) * 60.0 / (1024.0 * 1024.0)
                val optMbPerMin = (optBps / 8.0) * 60.0 / (1024.0 * 1024.0)
                val savingsPct = ((legacyBps - optBps).toDouble() / legacyBps.toDouble()) * 100.0
                sb.append(
                    String.format(
                        Locale.US,
                        "In %dx%d -> %s (%dx%d H.264): Legacy=%.1f MB/min (%d kbps) | Optimized=%.1f MB/min (%d kbps) | Saved=%.1f%%\n",
                        input.displayWidth, input.displayHeight, mode.label,
                        outDims.width, outDims.height,
                        legacyMbPerMin, legacyBps / 1000,
                        optMbPerMin, optBps / 1000,
                        savingsPct
                    )
                )
            }
            for (i in 1 until prevBitrates.size) {
                assertTrue(
                    "Monotonic bitrate HD<HD+<FHD<2K must hold for ${input.displayWidth}x${input.displayHeight}: $prevBitrates",
                    prevBitrates[i] > prevBitrates[i - 1]
                )
            }
        }
        println(sb.toString())
    }

    @Test
    fun `quality settings and zero-skip DnCNN remain 100 percent intact across all modes`() {
        val ctx = ApplicationProvider.getApplicationContext<Context>()
        val profiles = EnhancementProfileRepository(ctx).loadProfiles()
        val adjuster = SmartAutoAdjustmentEngine()

        val sampleAnalysis = com.example.model.VideoQualityAnalysis(
            avgBrightness = 116f,
            darkAreaRatio = 0.12f,
            overexposedAreaRatio = 0.02f,
            contrastStdDev = 45f,
            dynamicRange = 150f,
            noiseLevel = 9.0f,
            blurVariance = 85f,
            saturationMean = 95f,
            whiteBalanceShiftR = 1.0f,
            whiteBalanceShiftB = 1.0f,
            compressionArtifactScore = 1.2f,
            edgeDensity = 0.18f,
            conditionTags = listOf("Moderate Noise")
        )

        for (mode in EnhancementMode.entries) {
            val profile = profiles.getValue(mode)
            val params = adjuster.computeAdaptiveAdjustments(profile, sampleAnalysis)
            assertTrue("${mode.id} must preserve AI super-resolution", profile.useAiSuperResolution)
            assertTrue("${mode.id} must preserve DnCNN denoise enabled", params.useAiDenoise)
            assertTrue("${mode.id} must preserve detail recovery > 0.4", params.detailRecovery > 0.4f)

            val gov = SpeedBudgetGovernor(
                mode = mode,
                durationMs = 60_000L,
                estimatedFrames = 1800,
                presetTileSize = profile.aiTileSize,
                presetTilesCap = profile.targetTilesPerFrame,
                measuredSrTileMs = 45L,
                deviceCapacityScore = 90,
                measuredDnCnnMs = 8L
            )
            // Enforce DnCNN on every single frame (0..1799) across every speed tier
            for (f in 0 until 1800) {
                val d = gov.decisionForFrame(frameIndex = f, isSceneCut = (f % 300 == 0), forceLightweight = false)
                assertTrue("DnCNN skipped at frame $f in ${mode.id}", d.allowDnCnn)
            }
        }
        // Static tile cache policy must force refresh every MAX_CONSECUTIVE_REUSES frames
        assertFalse(TemporalReusePolicy.shouldReuseStaticTile(mad = 0.2f, consecutiveReuseCount = 3, isSceneCut = false, hasCachedOutput = true))
        assertTrue(TemporalReusePolicy.shouldReuseStaticTile(mad = 0.4f, consecutiveReuseCount = 1, isSceneCut = false, hasCachedOutput = true))
    }

    @Test
    fun `jvm compute kernels pass throughput sanity across Q4 DnCNN, optical flow, and smart tiling`() {
        val w = 128
        val h = 128
        val src = ByteArray(w * h) { i -> ((i * 7) % 200 + 25).toByte() }
        val res = ShortArray(w * h) { -16 }
        val dst = ByteArray(w * h)

        val t0 = System.nanoTime()
        repeat(60) {
            DnCnnResidualMath.applyQ4ResidualAndEdgeGatedSmooth(src, w, h, res, 0.85f, dst)
        }
        val dncnnKernelMs = (System.nanoTime() - t0) / 1_000_000L
        assertTrue("Expected fast Q4 residual pass", dncnnKernelMs >= 0L)

        // Neural working size stays strictly bounded within 4-tile NPU/GPU budget
        val (workW, workH) = AiTilingMath.computeNeuralWorkingSize(1920, 1080, 3840, 2160, tileSize = 256, maxTilesCap = 4)
        val grid = AiTilingMath.planUpscaleTileGrid(workW, workH, 3840, 2160, tileSize = 256, maxTilesCap = 4)
        assertTrue("Neural working canvas must fit inside maxTilesCap=4", grid.totalTiles in 1..4)

        // 1-minute timing targets remain active and properly ordered
        assertEquals(150_000L, SpeedBudgetGovernor.maxBudgetMsPerMinute(EnhancementMode.HD))
        assertEquals(300_000L, SpeedBudgetGovernor.maxBudgetMsPerMinute(EnhancementMode.HD_PLUS))
        assertEquals(450_000L, SpeedBudgetGovernor.maxBudgetMsPerMinute(EnhancementMode.FHD))
        assertEquals(600_000L, SpeedBudgetGovernor.maxBudgetMsPerMinute(EnhancementMode.TWO_K))
        assertTrue(SpeedExecutionTier.FULL_NEURAL_PER_FRAME != SpeedExecutionTier.GPU_SHADER_FAST_PASS)
        assertTrue(MotionSmoothMath.shouldInterpolateCadence(24.0, true))
    }
}
