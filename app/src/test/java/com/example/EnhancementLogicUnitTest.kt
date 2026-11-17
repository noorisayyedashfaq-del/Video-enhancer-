package com.example

import android.net.Uri
import com.example.engine.AiTilingMath
import com.example.engine.EngineDiagnosticsLogger
import com.example.engine.ResolutionCalculator
import com.example.engine.SmartAutoAdjustmentEngine
import com.example.model.EnhancementMode
import com.example.model.EnhancementProfile
import com.example.model.ProcessingStage
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
        EnhancementMode.HD -> EnhancementProfile("HD", "HD", "", 0, 0, true, false, false, false, 0, 0, true, 3f, 8f, 15f, 10f, 5f, 1.03f, "Automatic", "Automatic", "Low", 5f, 10f, "Automatic", "H264", 0.16f)
        EnhancementMode.HD_PLUS -> EnhancementProfile("HD_PLUS", "HD+", "", 0, 0, true, true, false, true, 512, 6, true, 3f, 10f, 20f, 15f, 6f, 1.04f, "Automatic", "Automatic", "Medium", 8f, 15f, "Automatic", "H264", 0.17f)
        EnhancementMode.QHD_2K -> EnhancementProfile("QHD_2K", "2K", "", 0, 0, true, true, false, true, 384, 12, true, 3f, 12f, 25f, 20f, 7f, 1.05f, "Automatic", "Automatic", "Medium", 10f, 20f, "Automatic", "H265", 0.16f)
        EnhancementMode.UHD_4K -> EnhancementProfile("UHD_4K", "4K", "", 0, 0, true, true, false, true, 256, 25, true, 3f, 12f, 25f, 20f, 7f, 1.05f, "Automatic", "Automatic", "Medium", 10f, 20f, "Automatic", "H265", 0.16f)
    }

    @Test
    fun `all enhancement presets keep original video resolution unchanged and configure exact tile presets`() {
        val hd = ResolutionCalculator.calculate(sampleMetadataLandscape, EnhancementMode.HD, profileFor(EnhancementMode.HD), null)
        assertEquals(1920, hd.width)
        assertEquals(1080, hd.height)
        assertFalse(hd.needsAiSuperResolution)
        assertEquals(0, hd.aiTileSize)
        assertEquals(0, hd.tilesPerFrame)

        val hdPlus = ResolutionCalculator.calculate(sampleMetadataLandscape, EnhancementMode.HD_PLUS, profileFor(EnhancementMode.HD_PLUS), null)
        assertEquals(1920, hdPlus.width)
        assertEquals(1080, hdPlus.height)
        assertTrue(hdPlus.needsAiSuperResolution)
        assertEquals(512, hdPlus.aiTileSize)
        assertEquals(6, hdPlus.tilesPerFrame)

        val qhd2k = ResolutionCalculator.calculate(sampleMetadataLandscape, EnhancementMode.QHD_2K, profileFor(EnhancementMode.QHD_2K), null)
        assertEquals(1920, qhd2k.width)
        assertEquals(1080, qhd2k.height)
        assertTrue(qhd2k.needsAiSuperResolution)
        assertEquals(384, qhd2k.aiTileSize)
        assertEquals(12, qhd2k.tilesPerFrame)

        val uhd4k = ResolutionCalculator.calculate(sampleMetadataLandscape, EnhancementMode.UHD_4K, profileFor(EnhancementMode.UHD_4K), null)
        assertEquals(1920, uhd4k.width)
        assertEquals(1080, uhd4k.height)
        assertTrue(uhd4k.needsAiSuperResolution)
        assertEquals(256, uhd4k.aiTileSize)
        assertEquals(25, uhd4k.tilesPerFrame)
    }

    @Test
    fun `portrait video also preserves original resolution across all presets`() {
        for (mode in EnhancementMode.entries) {
            val out = ResolutionCalculator.calculate(sampleMetadataPortrait, mode, profileFor(mode), null)
            assertEquals(720, out.width)
            assertEquals(1280, out.height)
            assertTrue(out.preserveOriginalResolution)
        }
    }

    @Test
    fun `dynamic tile grids divide frames into 6, 12, and 25 overlapping tiles with zero cropping`() {
        val resolutions = listOf(640 to 360, 1280 to 720, 1920 to 1080, 720 to 1280, 1080 to 1080)
        val targets = listOf(6, 12, 25)

        for ((w, h) in resolutions) {
            for (target in targets) {
                val grid = AiTilingMath.planDynamicTileGrid(w, h, target)
                assertEquals("Expected exact $target tiles for ${w}x${h}", target, grid.totalTiles)
                assertEquals(target, grid.tiles.size)
                assertTrue("Horizontal overlap must be > 0 for ${w}x${h}", grid.overlapX > 0)
                assertTrue("Vertical overlap must be > 0 for ${w}x${h}", grid.overlapY > 0)

                // Verify full coverage from (0,0) to (w,h) with zero cropped pixels and overlapping neighbors
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
            initialTileSize = 384,
            finalTileSize = 384,
            framesProcessed = 120,
            recoveredFrames = 0,
            analysisReusedFromCache = true
        )

        assertEquals(210L, report.modelInitMs)
        assertEquals(140L, report.analysisMs)
        assertEquals(3200L, report.aiInferenceMs)
        assertEquals(384, report.finalTileSize)
        assertTrue(report.analysisReusedFromCache)
        assertEquals(1, report.warnings.size)
        assertEquals("AI Enhancement", ProcessingStage.ENHANCING.title)
    }
}
