package com.example

import android.net.Uri
import com.example.engine.AiTilingMath
import com.example.engine.EngineDiagnosticsLogger
import com.example.engine.OpenCvFrameProcessor
import com.example.engine.ResolutionCalculator
import com.example.engine.SmartAutoAdjustmentEngine
import com.example.model.EnhancementMode
import com.example.model.EnhancementProfile
import com.example.model.ProcessingStage
import com.example.model.VideoMetadata
import com.example.model.VideoQualityAnalysis
import org.junit.Assert.assertEquals
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
        width = 1920,
        height = 1080,
        rotationDegrees = 0,
        displayWidth = 1920,
        displayHeight = 1080,
        fps = 30.0,
        durationMs = 5000L,
        estimatedFrameCount = 150,
        bitrateBps = 8_000_000L,
        videoCodec = "h264",
        hasAudio = true,
        audioCodec = "aac",
        audioSampleRate = 48000,
        audioChannels = 2,
        fileSizeBytes = 5_000_000L
    )

    private val sampleMetadataPortrait = sampleMetadataLandscape.copy(
        width = 720,
        height = 1280,
        displayWidth = 720,
        displayHeight = 1280
    )

    private fun profileFor(mode: EnhancementMode) = when (mode) {
        EnhancementMode.CLARITY_1X -> EnhancementProfile(
            "CLARITY_1X", "Clarity 1x", "", 0, 0, true, true, false, true, 512, 4, true,
            1f, 3f, 12f, 25f, 4f, 3f, 5f, -3f, 2f, 2f, -2f, 0f, 0f,
            25f, 25f, 25f, 20f, 15f, 20f,
            1.0f, "Neutral", "Natural", "Low", 2f, 5f, "Automatic", "H264", 0.14f
        )
        EnhancementMode.CLARITY_2X -> EnhancementProfile(
            "CLARITY_2X", "Clarity 2x", "", 0, 0, true, true, false, true, 512, 6, true,
            2f, 5f, 22f, 40f, 7f, 6f, 12f, -6f, 3f, 3f, -3f, 0f, 0f,
            45f, 45f, 40f, 40f, 30f, 40f,
            1.0f, "Neutral", "Balanced", "Medium", 4f, 10f, "Automatic", "H264", 0.16f
        )
        EnhancementMode.CLARITY_3X -> EnhancementProfile(
            "CLARITY_3X", "Clarity 3x", "", 0, 0, true, true, false, true, 512, 12, true,
            3f, 8f, 35f, 60f, 10f, 9f, 20f, -9f, 5f, 5f, -5f, 0f, 0f,
            70f, 70f, 60f, 60f, 45f, 60f,
            1.0f, "Neutral", "Balanced", "Medium", 6f, 14f, "Automatic", "H265", 0.17f
        )
        EnhancementMode.CLARITY_4X -> EnhancementProfile(
            "CLARITY_4X", "Clarity 4x", "", 0, 0, true, true, false, true, 512, 25, true,
            4f, 10f, 45f, 75f, 12f, 12f, 28f, -12f, 6f, 7f, -7f, 0f, 0f,
            90f, 90f, 75f, 80f, 60f, 75f,
            1.0f, "Neutral", "Balanced", "High", 8f, 18f, "Automatic", "H265", 0.18f
        )
    }

    @Test
    fun `all 4 Clarity presets keep original video resolution unchanged and configure exact tile presets`() {
        val c1 = ResolutionCalculator.calculate(sampleMetadataLandscape, EnhancementMode.CLARITY_1X, profileFor(EnhancementMode.CLARITY_1X), null)
        assertEquals(1920, c1.width)
        assertEquals(1080, c1.height)
        assertTrue(c1.needsAiSuperResolution)
        assertEquals(512, c1.aiTileSize)
        assertEquals(4, c1.tilesPerFrame)

        val c2 = ResolutionCalculator.calculate(sampleMetadataLandscape, EnhancementMode.CLARITY_2X, profileFor(EnhancementMode.CLARITY_2X), null)
        assertEquals(1920, c2.width)
        assertEquals(1080, c2.height)
        assertTrue(c2.needsAiSuperResolution)
        assertEquals(512, c2.aiTileSize)
        assertEquals(6, c2.tilesPerFrame)

        val c3 = ResolutionCalculator.calculate(sampleMetadataLandscape, EnhancementMode.CLARITY_3X, profileFor(EnhancementMode.CLARITY_3X), null)
        assertEquals(1920, c3.width)
        assertEquals(1080, c3.height)
        assertTrue(c3.needsAiSuperResolution)
        assertEquals(512, c3.aiTileSize)
        assertEquals(12, c3.tilesPerFrame)

        val c4 = ResolutionCalculator.calculate(sampleMetadataLandscape, EnhancementMode.CLARITY_4X, profileFor(EnhancementMode.CLARITY_4X), null)
        assertEquals(1920, c4.width)
        assertEquals(1080, c4.height)
        assertTrue(c4.needsAiSuperResolution)
        assertEquals(512, c4.aiTileSize)
        assertEquals(25, c4.tilesPerFrame)
    }

    @Test
    fun `portrait video also preserves original resolution across all Clarity presets`() {
        for (mode in EnhancementMode.entries) {
            val out = ResolutionCalculator.calculate(sampleMetadataPortrait, mode, profileFor(mode), null)
            assertEquals(720, out.width)
            assertEquals(1280, out.height)
            assertTrue(out.preserveOriginalResolution)
        }
    }

    @Test
    fun `dynamic tile grids divide frames into 4, 6, 12, and 25 overlapping tiles with zero cropping`() {
        val resolutions = listOf(640 to 360, 1280 to 720, 1920 to 1080, 2560 to 1080, 720 to 1280, 1080 to 1080)
        val targets = listOf(4, 6, 12, 25)

        for ((w, h) in resolutions) {
            for (target in targets) {
                val grid = AiTilingMath.planDynamicTileGrid(w, h, target)
                assertEquals("Expected exact $target tiles for ${w}x${h}", target, grid.totalTiles)
                assertEquals(target, grid.tiles.size)
                if (target == 4) {
                    assertEquals(2, grid.cols)
                    assertEquals(2, grid.rows)
                }
                assertTrue("Horizontal overlap must be > 0 for ${w}x${h}", grid.overlapX > 0)
                assertTrue("Vertical overlap must be > 0 for ${w}x${h}", grid.overlapY > 0)

                assertEquals(0, grid.tiles.minOf { it.x })
                assertEquals(0, grid.tiles.minOf { it.y })
                assertEquals(w, grid.tiles.maxOf { it.x + it.width })
                assertEquals(h, grid.tiles.maxOf { it.y + it.height })

                for (r in 0 until grid.rows) {
                    val rowTiles = grid.tiles.filter { it.row == r }.sortedBy { it.col }
                    for (c in 0 until rowTiles.size - 1) {
                        val curr = rowTiles[c]
                        val next = rowTiles[c + 1]
                        assertTrue(
                            "Tiles in row $r must overlap cleanly for ${w}x${h} ($target tiles)",
                            curr.x + curr.width > next.x
                        )
                    }
                }
                for (c in 0 until grid.cols) {
                    val colTiles = grid.tiles.filter { it.col == c }.sortedBy { it.row }
                    for (r in 0 until colTiles.size - 1) {
                        val curr = colTiles[r]
                        val next = colTiles[r + 1]
                        assertTrue(
                            "Tiles in col $c must overlap cleanly for ${w}x${h} ($target tiles)",
                            curr.y + curr.height > next.y
                        )
                    }
                }
            }
        }

        assertEquals(1.0f, AiTilingMath.featherWeight(0, 100, isFirst = true, isLast = false, feather = 20), 0.001f)
        assertEquals(1.0f, AiTilingMath.featherWeight(99, 100, isFirst = false, isLast = true, feather = 20), 0.001f)
        val interiorStart = AiTilingMath.featherWeight(0, 100, isFirst = false, isLast = false, feather = 20)
        val interiorMid = AiTilingMath.featherWeight(50, 100, isFirst = false, isLast = false, feather = 20)
        assertTrue(interiorStart < interiorMid)
        assertEquals(1.0f, interiorMid, 0.001f)
    }

    @Test
    fun `smart auto adjuster calibrates Clarity 1x-4x colour and monotonic non-clipping LUT`() {
        val engine = SmartAutoAdjustmentEngine()
        val neutralSource = VideoQualityAnalysis(
            avgBrightness = 128f, darkAreaRatio = 0.05f, overexposedAreaRatio = 0.02f,
            contrastStdDev = 52f, dynamicRange = 170f, noiseLevel = 4.5f,
            blurVariance = 240f, saturationMean = 100f, whiteBalanceShiftR = 1.0f,
            whiteBalanceShiftB = 1.0f, compressionArtifactScore = 1.04f, edgeDensity = 0.22f,
            conditionTags = listOf("Balanced")
        )

        val p1 = engine.computeAdaptiveAdjustments(profileFor(EnhancementMode.CLARITY_1X), neutralSource)
        val p2 = engine.computeAdaptiveAdjustments(profileFor(EnhancementMode.CLARITY_2X), neutralSource)
        val p3 = engine.computeAdaptiveAdjustments(profileFor(EnhancementMode.CLARITY_3X), neutralSource)
        val p4 = engine.computeAdaptiveAdjustments(profileFor(EnhancementMode.CLARITY_4X), neutralSource)

        // Verify progressive AI Strength scaling 25% -> 45% -> 70% -> 90%
        assertEquals(0.25f, p1.aiStrength, 0.001f)
        assertEquals(0.45f, p2.aiStrength, 0.001f)
        assertEquals(0.70f, p3.aiStrength, 0.001f)
        assertEquals(0.90f, p4.aiStrength, 0.001f)

        // Verify neutral Temperature = 0 & Hue = 0 keep WB unbiased
        assertEquals(1.0f, p2.wbGainR, 0.001f)
        assertEquals(1.0f, p2.wbGainB, 0.001f)
        assertTrue(p2.skinToneProtection)
        assertTrue(p4.claheClipLimit > p1.claheClipLimit && p4.claheClipLimit <= 1.55)

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
            assertEquals(256, satLut.size)
            assertEquals(0, satLut[0].toInt() and 0xFF)
            assertEquals(255, satLut[255].toInt() and 0xFF)
        }

        val darkNoisy = VideoQualityAnalysis(
            avgBrightness = 48f, darkAreaRatio = 0.55f, overexposedAreaRatio = 0.0f,
            contrastStdDev = 28f, dynamicRange = 90f, noiseLevel = 14f,
            blurVariance = 70f, saturationMean = 40f, whiteBalanceShiftR = 1.1f,
            whiteBalanceShiftB = 0.9f, compressionArtifactScore = 1.4f, edgeDensity = 0.1f,
            conditionTags = listOf("Dark", "Noisy")
        )
        val darkParams = engine.computeAdaptiveAdjustments(profileFor(EnhancementMode.CLARITY_3X), darkNoisy)
        assertTrue(darkParams.brightnessShift > p3.brightnessShift)
        assertTrue(darkParams.gamma < 1.0f)
        assertTrue(darkParams.denoiseStrength > p3.denoiseStrength)
        assertTrue(darkParams.deblockingStrength > p3.deblockingStrength)
    }

    @Test
    fun `diagnostics logger and processing stages capture telemetry without hiding warnings`() {
        val logger = EngineDiagnosticsLogger()
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
            initialTileSize = 512,
            finalTileSize = 512,
            framesProcessed = 120,
            recoveredFrames = 0,
            analysisReusedFromCache = true
        )

        assertEquals(210L, report.modelInitMs)
        assertEquals(140L, report.analysisMs)
        assertEquals(3200L, report.aiInferenceMs)
        assertEquals(512, report.finalTileSize)
        assertTrue(report.analysisReusedFromCache)
        assertEquals(1, report.warnings.size)
        assertEquals("AI Enhancement", ProcessingStage.ENHANCING.title)
    }
}
