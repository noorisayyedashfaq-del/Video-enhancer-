package com.example

import android.content.Context
import android.net.Uri
import androidx.test.core.app.ApplicationProvider
import com.example.engine.AiTilingMath
import com.example.engine.EngineDiagnosticsLogger
import com.example.engine.EnhancementProfileRepository
import com.example.engine.OpenCvFrameProcessor
import com.example.engine.ResolutionCalculator
import com.example.engine.SmartAutoAdjustmentEngine
import com.example.model.DeviceCapabilities
import com.example.model.EnhancementMode
import com.example.model.ProcessingStage
import com.example.model.ThermalState
import com.example.model.VideoMetadata
import com.example.model.VideoQualityAnalysis
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.core.Rect
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [36])
class EnhancementLogicUnitTest {

    private fun testCapabilities(maxLongEdge: Int = 3840) = DeviceCapabilities(
        tierName = "High-End",
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
        maxEncodeLongEdge = maxLongEdge,
        availableStorageMb = 10_000L,
        thermalState = ThermalState.NORMAL,
        temperatureCelsius = 33f,
        adaptiveTileSize = 256,
        tileOverlap = 24,
        batchSize = 2,
        inferenceThreads = 4,
        minRequiredFreeStorageMb = 500L,
        throttleDelayMsOnWarm = 25L,
        throttleDelayMsOnHot = 80L,
        processingCapacityScore = 85
    )

    private val sampleMetadataLowRes = VideoMetadata(
        uri = Uri.EMPTY,
        displayName = "lowres_clip.mp4",
        width = 480,
        height = 360,
        rotationDegrees = 0,
        displayWidth = 480,
        displayHeight = 360,
        fps = 30.0,
        durationMs = 5000L,
        estimatedFrameCount = 150,
        bitrateBps = 1_200_000L,
        videoCodec = "h264",
        hasAudio = true,
        audioCodec = "aac",
        audioSampleRate = 48000,
        audioChannels = 2,
        fileSizeBytes = 1_500_000L
    )

    private val sampleMetadataHd = sampleMetadataLowRes.copy(
        displayName = "hd_clip.mp4",
        width = 1280,
        height = 720,
        displayWidth = 1280,
        displayHeight = 720,
        bitrateBps = 5_000_000L
    )

    private val sampleMetadataPortrait = sampleMetadataLowRes.copy(
        displayName = "portrait.mp4",
        width = 360,
        height = 640,
        displayWidth = 360,
        displayHeight = 640
    )

    @Test
    fun `all 4 Clarity modes apply progressive true super-resolution scaling and bitrate increases`() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val profiles = EnhancementProfileRepository(context).loadProfiles()
        val caps = testCapabilities(maxLongEdge = 3840)

        // Low-res 480x360 progressively upscales: 1x (480x360) -> 2x (960x720) -> 3x (1440x1080) -> 4x (1920x1440)
        val c1 = ResolutionCalculator.calculate(sampleMetadataLowRes, EnhancementMode.CLARITY_1X, profiles.getValue(EnhancementMode.CLARITY_1X), caps)
        val c2 = ResolutionCalculator.calculate(sampleMetadataLowRes, EnhancementMode.CLARITY_2X, profiles.getValue(EnhancementMode.CLARITY_2X), caps)
        val c3 = ResolutionCalculator.calculate(sampleMetadataLowRes, EnhancementMode.CLARITY_3X, profiles.getValue(EnhancementMode.CLARITY_3X), caps)
        val c4 = ResolutionCalculator.calculate(sampleMetadataLowRes, EnhancementMode.CLARITY_4X, profiles.getValue(EnhancementMode.CLARITY_4X), caps)

        assertEquals(480 to 360, c1.width to c1.height)
        assertEquals(960 to 720, c2.width to c2.height)
        assertEquals(1440 to 1080, c3.width to c3.height)
        assertEquals(1920 to 1440, c4.width to c4.height)

        assertTrue(c1.needsAiSuperResolution && c2.needsAiSuperResolution && c3.needsAiSuperResolution && c4.needsAiSuperResolution)
        assertTrue(c2.targetBitrateBps > c1.targetBitrateBps)
        assertTrue(c3.targetBitrateBps > c2.targetBitrateBps)
        assertTrue(c4.targetBitrateBps > c3.targetBitrateBps)
        // Small 480x360 should NOT explode into 25 wasteful tiny tiles
        assertTrue("Expected lean tile count <= 6 for 480x360, got ${c4.tilesPerFrame}", c4.tilesPerFrame <= 6)

        // 720p source cleanly scales up to 2560x1440 (2x), 3200x1800 (3x cap), and 3840x2160 4K (4x cap)
        val hd2 = ResolutionCalculator.calculate(sampleMetadataHd, EnhancementMode.CLARITY_2X, profiles.getValue(EnhancementMode.CLARITY_2X), caps)
        val hd3 = ResolutionCalculator.calculate(sampleMetadataHd, EnhancementMode.CLARITY_3X, profiles.getValue(EnhancementMode.CLARITY_3X), caps)
        val hd4 = ResolutionCalculator.calculate(sampleMetadataHd, EnhancementMode.CLARITY_4X, profiles.getValue(EnhancementMode.CLARITY_4X), caps)

        assertEquals(2560 to 1440, hd2.width to hd2.height)
        assertEquals(3200 to 1800, hd3.width to hd3.height)
        assertEquals(3840 to 2160, hd4.width to hd4.height)

        // Respects lower hardware maxEncodeLongEdge cap when present
        val cappedCaps = testCapabilities(maxLongEdge = 1920)
        val hd4Capped = ResolutionCalculator.calculate(sampleMetadataHd, EnhancementMode.CLARITY_4X, profiles.getValue(EnhancementMode.CLARITY_4X), cappedCaps)
        assertEquals(1920 to 1080, hd4Capped.width to hd4Capped.height)
    }

    @Test
    fun `portrait video scales aspect-accurately across Clarity 1x to 4x`() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val profiles = EnhancementProfileRepository(context).loadProfiles()
        val caps = testCapabilities(maxLongEdge = 3840)

        val p1 = ResolutionCalculator.calculate(sampleMetadataPortrait, EnhancementMode.CLARITY_1X, profiles.getValue(EnhancementMode.CLARITY_1X), caps)
        val p2 = ResolutionCalculator.calculate(sampleMetadataPortrait, EnhancementMode.CLARITY_2X, profiles.getValue(EnhancementMode.CLARITY_2X), caps)
        val p4 = ResolutionCalculator.calculate(sampleMetadataPortrait, EnhancementMode.CLARITY_4X, profiles.getValue(EnhancementMode.CLARITY_4X), caps)

        assertEquals(360 to 640, p1.width to p1.height)
        assertEquals(720 to 1280, p2.width to p2.height)
        assertEquals(1440 to 2560, p4.width to p4.height)
    }

    @Test
    fun `upscale tile planner covers 100 percent of input and output dimensions with guaranteed overlap and zero gaps`() {
        val cases = listOf(
            listOf(480, 360, 960, 720, 256, 9),
            listOf(640, 360, 2560, 1440, 384, 16),
            listOf(1280, 720, 3840, 2160, 384, 16),
            listOf(1920, 1080, 3840, 2160, 384, 16),
            listOf(360, 640, 1440, 2560, 384, 16),
            listOf(200, 180, 800, 720, 256, 6)
        )

        for (c in cases) {
            val inW = c[0]
            val inH = c[1]
            val outW = c[2]
            val outH = c[3]
            val tileSize = c[4]
            val cap = c[5]
            val grid = AiTilingMath.planUpscaleTileGrid(inW, inH, outW, outH, tileSize, cap)
            assertTrue("Must have >= 1 tile", grid.tiles.isNotEmpty())
            assertTrue("Must respect maxTilesCap ($cap)", grid.totalTiles <= cap)

            // Input full coverage
            assertEquals(0, grid.tiles.minOf { it.x })
            assertEquals(0, grid.tiles.minOf { it.y })
            assertEquals(inW, grid.tiles.maxOf { it.x + it.width })
            assertEquals(inH, grid.tiles.maxOf { it.y + it.height })

            // Output full coverage
            assertEquals(0, grid.tiles.minOf { it.outX })
            assertEquals(0, grid.tiles.minOf { it.outY })
            assertEquals(outW, grid.tiles.maxOf { it.outX + it.outWidth })
            assertEquals(outH, grid.tiles.maxOf { it.outY + it.outHeight })

            for (r in 0 until grid.rows) {
                val rowTiles = grid.tiles.filter { it.row == r }.sortedBy { it.col }
                for (i in 0 until rowTiles.size - 1) {
                    assertTrue(rowTiles[i].x + rowTiles[i].width > rowTiles[i + 1].x)
                    assertTrue(rowTiles[i].outX + rowTiles[i].outWidth > rowTiles[i + 1].outX)
                }
            }
            for (col in 0 until grid.cols) {
                val colTiles = grid.tiles.filter { it.col == col }.sortedBy { it.row }
                for (i in 0 until colTiles.size - 1) {
                    assertTrue(colTiles[i].y + colTiles[i].height > colTiles[i + 1].y)
                    assertTrue(colTiles[i].outY + colTiles[i].outHeight > colTiles[i + 1].outY)
                }
            }
        }
    }

    @Test
    fun `smart adjuster progressively ramps Real-ESRGAN, DnCNN, and Face AI while keeping color filters natural`() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val profiles = EnhancementProfileRepository(context).loadProfiles()
        val engine = SmartAutoAdjustmentEngine()

        val blurryNoisyCompressed = VideoQualityAnalysis(
            avgBrightness = 112f,
            darkAreaRatio = 0.14f,
            overexposedAreaRatio = 0.01f,
            contrastStdDev = 42f,
            dynamicRange = 140f,
            noiseLevel = 11.5f,
            blurVariance = 78f,
            saturationMean = 92f,
            whiteBalanceShiftR = 1.0f,
            whiteBalanceShiftB = 1.0f,
            compressionArtifactScore = 1.32f,
            edgeDensity = 0.15f,
            conditionTags = listOf("High Noise", "Soft / Blurry", "Compression Artifacts")
        )

        val p1 = engine.computeAdaptiveAdjustments(profiles.getValue(EnhancementMode.CLARITY_1X), blurryNoisyCompressed)
        val p2 = engine.computeAdaptiveAdjustments(profiles.getValue(EnhancementMode.CLARITY_2X), blurryNoisyCompressed)
        val p3 = engine.computeAdaptiveAdjustments(profiles.getValue(EnhancementMode.CLARITY_3X), blurryNoisyCompressed)
        val p4 = engine.computeAdaptiveAdjustments(profiles.getValue(EnhancementMode.CLARITY_4X), blurryNoisyCompressed)

        // Strict progressive neural enhancement across 1x -> 2x -> 3x -> 4x
        assertTrue(p1.aiStrength < p2.aiStrength && p2.aiStrength < p3.aiStrength && p3.aiStrength < p4.aiStrength)
        assertTrue(p1.dncnnStrength < p2.dncnnStrength && p2.dncnnStrength < p3.dncnnStrength && p3.dncnnStrength < p4.dncnnStrength)
        assertTrue(p1.faceEnhanceStrength < p2.faceEnhanceStrength && p2.faceEnhanceStrength < p3.faceEnhanceStrength && p3.faceEnhanceStrength < p4.faceEnhanceStrength)
        assertTrue(p1.useAiDenoise && p2.useAiDenoise && p3.useAiDenoise && p4.useAiDenoise)

        // Genuine visible yet natural color mastering + active detail recovery & smoothness parameters
        assertTrue("Brightness shift should stay calibrated and non-blown (${p4.brightnessShift})", p4.brightnessShift in 2.5f..5.5f)
        assertTrue("Saturation boost must visibly enrich without clipping (${p4.saturationGain})", p4.saturationGain in 1.10f..1.22f)
        assertTrue("Vibrance must ramp progressively across 1x->4x", p1.vibranceGain < p4.vibranceGain && p4.vibranceGain in 0.25f..0.65f)
        assertTrue("Detail recovery must ramp progressively", p1.detailRecovery < p2.detailRecovery && p2.detailRecovery < p3.detailRecovery && p3.detailRecovery < p4.detailRecovery)
        assertTrue("Deblur strength must ramp progressively", p1.deblurStrength < p2.deblurStrength && p2.deblurStrength < p3.deblurStrength && p3.deblurStrength < p4.deblurStrength)
        assertTrue("AI sharpening must stay progressive and halo-safe", p1.aiSharpening < p2.aiSharpening && p2.aiSharpening < p3.aiSharpening && p3.aiSharpening < p4.aiSharpening && p4.aiSharpening <= 0.68f)
        assertTrue("Motion smoothness must be active", p1.enableSmoothness && p4.enableSmoothness && p4.allowFrameInterpolation)
        assertEquals(1.0f, p2.wbGainR, 0.001f)
        assertEquals(1.0f, p2.wbGainB, 0.001f)
        assertTrue(p2.skinToneProtection)

        // Verify monotonic tone LUT without inversion or highlight clipping
        for (params in listOf(p1, p2, p3, p4)) {
            val lut = OpenCvFrameProcessor.buildToneLutArray(params)
            assertEquals(256, lut.size)
            for (i in 1..255) {
                val prev = lut[i - 1].toInt() and 0xFF
                val curr = lut[i].toInt() and 0xFF
                assertTrue("LUT must be monotonic at $i", curr >= prev)
            }
            val satLut = OpenCvFrameProcessor.buildSaturationLutArray(params.saturationGain)
            assertEquals(0, satLut[0].toInt() and 0xFF)
            assertEquals(255, satLut[255].toInt() and 0xFF)
        }
    }

    @Test
    fun `temporal face box smoothing prevents bounding box shimmer within frame bounds`() {
        val prev = listOf(Rect(100, 80, 60, 70))
        val curr = listOf(Rect(110, 90, 66, 74))
        val smoothed = OpenCvFrameProcessor.smoothFaceBoxes(prev, curr, frameW = 480, frameH = 360)
        assertEquals(1, smoothed.size)
        val box = smoothed[0]
        assertTrue(box.x in 101..109)
        assertTrue(box.y in 81..89)
        assertTrue(box.x + box.width <= 480 && box.y + box.height <= 360)
    }

    @Test
    fun `diagnostics logger records SR, DnCNN, Face AI, and upscale scale factor telemetry`() {
        val logger = EngineDiagnosticsLogger()
        logger.modelInitMs = 210L
        logger.analysisMs = 140L
        logger.aiInferenceMs.set(3200L)
        logger.dncnnInferenceMs.set(900L)
        logger.faceAiMs.set(620L)
        logger.encodingMs.set(950L)

        val report = logger.buildReport(
            totalMs = 5900L,
            executionProvider = "CPU XNNPACK (4T)",
            benchmarkSummary = "XNN-4T=45ms",
            codecUsed = "H.265 (Hardware)",
            initialTileSize = 384,
            finalTileSize = 384,
            framesProcessed = 120,
            recoveredFrames = 0,
            analysisReusedFromCache = true,
            upscaleScaleLabel = "4.0x"
        )

        assertEquals(3200L, report.aiInferenceMs)
        assertEquals(900L, report.dncnnInferenceMs)
        assertEquals(620L, report.faceAiMs)
        assertEquals("4.0x", report.upscaleScaleLabel)
        assertEquals(384, report.finalTileSize)
        assertTrue(report.analysisReusedFromCache)
        assertEquals("AI Enhancement", ProcessingStage.ENHANCING.title)
    }
}
