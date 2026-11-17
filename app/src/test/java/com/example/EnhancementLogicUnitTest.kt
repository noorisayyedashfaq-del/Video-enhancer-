package com.example

import android.net.Uri
import com.example.engine.ResolutionCalculator
import com.example.engine.SmartAutoAdjustmentEngine
import com.example.model.EnhancementMode
import com.example.model.EnhancementProfile
import com.example.model.VideoMetadata
import com.example.model.VideoQualityAnalysis
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [36])
class EnhancementLogicUnitTest {

    private val sampleMetadataLandscape = VideoMetadata(
        uri = Uri.EMPTY,
        displayName = "clip.mp4",
        width = 640,
        height = 360,
        rotationDegrees = 0,
        displayWidth = 640,
        displayHeight = 360,
        fps = 30.0,
        durationMs = 5000L,
        estimatedFrameCount = 150,
        bitrateBps = 2_000_000L,
        videoCodec = "h264",
        hasAudio = true,
        audioCodec = "aac",
        audioSampleRate = 48000,
        audioChannels = 2,
        fileSizeBytes = 1_500_000L
    )

    private val sampleMetadataPortrait = sampleMetadataLandscape.copy(
        width = 480,
        height = 854,
        displayWidth = 480,
        displayHeight = 854
    )

    private fun profileFor(mode: EnhancementMode) = when (mode) {
        EnhancementMode.HD -> EnhancementProfile("HD", "HD", "", 0, 0, true, false, false, false, true, 3f, 8f, 15f, 10f, 5f, 1.03f, "Automatic", "Automatic", "Low", 5f, 10f, "Automatic", "H264", 0.16f)
        EnhancementMode.HD_PLUS -> EnhancementProfile("HD_PLUS", "HD+", "", 1280, 720, false, true, false, true, true, 3f, 10f, 20f, 15f, 6f, 1.04f, "Automatic", "Automatic", "Medium", 8f, 15f, "Automatic", "H264", 0.17f)
        EnhancementMode.QHD_2K -> EnhancementProfile("QHD_2K", "2K", "", 2560, 1440, false, true, false, true, true, 3f, 12f, 25f, 20f, 7f, 1.05f, "Automatic", "Automatic", "Medium", 10f, 20f, "Automatic", "H265", 0.14f)
        EnhancementMode.UHD_4K -> EnhancementProfile("UHD_4K", "4K", "", 3840, 2160, false, true, true, true, true, 3f, 12f, 25f, 20f, 7f, 1.05f, "Automatic", "Automatic", "Medium", 10f, 20f, "Automatic", "H265", 0.12f)
    }

    @Test
    fun `HD mode preserves original resolution`() {
        val out = ResolutionCalculator.calculate(sampleMetadataLandscape, EnhancementMode.HD, profileFor(EnhancementMode.HD), null)
        assertEquals(640, out.width)
        assertEquals(360, out.height)
        assertFalse(out.needsAiSuperResolution)
    }

    @Test
    fun `portrait video preserves portrait aspect ratio across HD+ 2K and 4K`() {
        val inRatio = 480f / 854f
        val p720 = ResolutionCalculator.calculate(sampleMetadataPortrait, EnhancementMode.HD_PLUS, profileFor(EnhancementMode.HD_PLUS), null)
        assertTrue(p720.height > p720.width)
        assertTrue(p720.height in 1276..1280)
        assertTrue(kotlin.math.abs((p720.width.toFloat() / p720.height) - inRatio) < 0.01f)
        assertTrue(p720.needsAiSuperResolution)

        val p4k = ResolutionCalculator.calculate(sampleMetadataPortrait, EnhancementMode.UHD_4K, profileFor(EnhancementMode.UHD_4K), null)
        assertTrue(p4k.height > p4k.width)
        assertTrue(p4k.height in 3836..3840)
        assertTrue(kotlin.math.abs((p4k.width.toFloat() / p4k.height) - inRatio) < 0.01f)
        assertTrue(p4k.needsMultiStageUpscaling)
    }

    @Test
    fun `higher resolution input is not downscaled in HD+ mode`() {
        val input1080p = sampleMetadataLandscape.copy(width = 1920, height = 1080, displayWidth = 1920, displayHeight = 1080)
        val out = ResolutionCalculator.calculate(input1080p, EnhancementMode.HD_PLUS, profileFor(EnhancementMode.HD_PLUS), null)
        assertEquals(1920, out.width)
        assertEquals(1080, out.height)
        assertFalse(out.needsAiSuperResolution)
    }

    @Test
    fun `smart auto adjuster adapts to dark noisy vs already high quality source`() {
        val engine = SmartAutoAdjustmentEngine()
        val profile = profileFor(EnhancementMode.QHD_2K)

        val darkNoisy = VideoQualityAnalysis(
            avgBrightness = 48f, darkAreaRatio = 0.55f, overexposedAreaRatio = 0.0f,
            contrastStdDev = 28f, dynamicRange = 90f, noiseLevel = 14f,
            blurVariance = 70f, saturationMean = 40f, whiteBalanceShiftR = 1.1f,
            whiteBalanceShiftB = 0.9f, compressionArtifactScore = 1.4f, edgeDensity = 0.1f,
            conditionTags = listOf("Dark", "Noisy")
        )
        val cleanSharp = VideoQualityAnalysis(
            avgBrightness = 128f, darkAreaRatio = 0.05f, overexposedAreaRatio = 0.02f,
            contrastStdDev = 55f, dynamicRange = 170f, noiseLevel = 2.5f,
            blurVariance = 480f, saturationMean = 110f, whiteBalanceShiftR = 1.0f,
            whiteBalanceShiftB = 1.0f, compressionArtifactScore = 1.04f, edgeDensity = 0.25f,
            conditionTags = listOf("Balanced")
        )

        val darkParams = engine.computeAdaptiveAdjustments(profile, darkNoisy)
        val cleanParams = engine.computeAdaptiveAdjustments(profile, cleanSharp)

        assertTrue(darkParams.brightnessShift > cleanParams.brightnessShift)
        assertTrue(darkParams.gamma < 1.0f)
        assertTrue(darkParams.denoiseStrength > cleanParams.denoiseStrength)
        assertTrue(darkParams.deblockingStrength > cleanParams.deblockingStrength)
        assertTrue(cleanParams.sharpnessAmount < profile.sharpnessPercent / 100f)
        assertFalse("Clean 1080p should use fast temporal+bilateral instead of heavy DnCNN", cleanParams.useAiDenoise)
        assertTrue("Noisy source must trigger DnCNN AI denoising", darkParams.useAiDenoise)
    }

    @Test
    fun `1080p to 2K and 4K resolutions and AI tiling plan avoid 8K intermediate bloat`() {
        val input1080p = sampleMetadataLandscape.copy(
            width = 1920,
            height = 1080,
            displayWidth = 1920,
            displayHeight = 1080
        )

        val out2k = ResolutionCalculator.calculate(input1080p, EnhancementMode.QHD_2K, profileFor(EnhancementMode.QHD_2K), null)
        assertEquals(2560, out2k.width)
        assertEquals(1440, out2k.height)
        assertTrue(out2k.needsAiSuperResolution)

        val plan2k = com.example.engine.AiTilingMath.planAiInputDimensions(1920, 1080, out2k.width, out2k.height)
        assertEquals(640, plan2k.aiWidth)
        assertEquals(360, plan2k.aiHeight)
        assertTrue(plan2k.needsDetailResidualFusion)

        val out4k = ResolutionCalculator.calculate(input1080p, EnhancementMode.UHD_4K, profileFor(EnhancementMode.UHD_4K), null)
        assertEquals(3840, out4k.width)
        assertEquals(2160, out4k.height)
        assertTrue(out4k.needsAiSuperResolution)

        val plan4k = com.example.engine.AiTilingMath.planAiInputDimensions(1920, 1080, out4k.width, out4k.height)
        assertEquals(960, plan4k.aiWidth)
        assertEquals(540, plan4k.aiHeight)
        assertTrue(plan4k.needsDetailResidualFusion)

        // Low-res 480p -> 4K must keep full native input resolution for 4x AI SR
        val plan480pTo4k = com.example.engine.AiTilingMath.planAiInputDimensions(854, 480, 3840, 2160)
        assertEquals(854, plan480pTo4k.aiWidth)
        assertEquals(480, plan480pTo4k.aiHeight)
        assertFalse(plan480pTo4k.needsDetailResidualFusion)
    }

    @Test
    fun `fixed tile planner covers entire frame bounds without gaps`() {
        val startsX = com.example.engine.AiTilingMath.computeTileStarts(totalLen = 960, tileLen = 192, step = 168)
        assertEquals(0, startsX.first())
        assertEquals(960 - 192, startsX.last())
        for (i in 0 until startsX.size - 1) {
            assertTrue("Adjacent tiles must overlap or meet", startsX[i] + 192 >= startsX[i + 1])
        }
    }

    @Test
    fun `diagnostics logger and processing stages capture telemetry without hiding warnings`() {
        val logger = com.example.engine.EngineDiagnosticsLogger()
        logger.modelInitMs = 210L
        logger.analysisMs = 140L
        logger.aiInferenceMs.set(3200L)
        logger.encodingMs.set(950L)
        logger.warn("NNAPI slow on probe tile; selected CPU XNNPACK")

        val report = logger.buildReport(
            totalMs = 4800L,
            executionProvider = "CPU XNNPACK (4T)",
            benchmarkSummary = "NNAPI=45ms vs CPU=22ms",
            codecUsed = "H.265 (Hardware)",
            initialTileSize = 192,
            finalTileSize = 128,
            framesProcessed = 120,
            recoveredFrames = 0,
            analysisReusedFromCache = true
        )

        assertEquals(210L, report.modelInitMs)
        assertEquals(140L, report.analysisMs)
        assertEquals(3200L, report.aiInferenceMs)
        assertEquals(128, report.finalTileSize)
        assertTrue(report.analysisReusedFromCache)
        assertEquals(1, report.warnings.size)

        assertEquals("Analyzing Video", com.example.model.ProcessingStage.ANALYZING.title)
        assertEquals("Loading AI Model", com.example.model.ProcessingStage.LOADING_MODEL.title)
        assertEquals("Extracting Frames", com.example.model.ProcessingStage.EXTRACTING.title)
        assertEquals("AI Enhancement", com.example.model.ProcessingStage.ENHANCING.title)
        assertEquals("Upscaling", com.example.model.ProcessingStage.UPSCALING.title)
        assertEquals("Video Encoding", com.example.model.ProcessingStage.ENCODING.title)
        assertEquals("Finalizing", com.example.model.ProcessingStage.FINALIZING.title)
    }
}
