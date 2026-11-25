package com.example

import android.content.Context
import android.net.Uri
import androidx.test.core.app.ApplicationProvider
import com.example.engine.AiTilingMath
import com.example.engine.ColorMath
import com.example.engine.EngineDiagnosticsLogger
import com.example.engine.EnhancementProfileRepository
import com.example.engine.MotionSmoothMath
import com.example.engine.ResolutionCalculator
import com.example.engine.SmartAutoAdjustmentEngine
import com.example.engine.SpeedBudgetGovernor
import com.example.engine.SpeedExecutionTier
import com.example.model.DeviceCapabilities
import com.example.model.EnhancementMode
import com.example.model.ThermalState
import com.example.model.VideoMetadata
import com.example.model.VideoQualityAnalysis
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.util.Locale
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [36])
class FourPillarEnhancementTest {

    private fun sampleCapabilities() = DeviceCapabilities(
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
        availableStorageMb = 10_000L,
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

    private fun argb(r: Int, g: Int, b: Int): Int {
        return (0xFF shl 24) or (r.coerceIn(0, 255) shl 16) or (g.coerceIn(0, 255) shl 8) or b.coerceIn(0, 255)
    }

    private fun rgbDiffMean(pixel: Int): Float {
        val r = (pixel ushr 16) and 0xFF
        val g = (pixel ushr 8) and 0xFF
        val b = pixel and 0xFF
        return (max(r, max(g, b)) - min(r, min(g, b))).toFloat()
    }

    private fun lumaOf(pixel: Int): Float {
        val r = (pixel ushr 16) and 0xFF
        val g = (pixel ushr 8) and 0xFF
        val b = pixel and 0xFF
        return 0.299f * r + 0.587f * g + 0.114f * b
    }

    private fun synthesizeMovingScene(width: Int, height: Int, shiftX: Int, shiftY: Int): IntArray {
        val out = IntArray(width * height)
        for (y in 0 until height) {
            val sy = y - shiftY
            for (x in 0 until width) {
                val sx = x - shiftX
                // Rich multi-frequency texture: smooth sky + foliage gradient + crisp grid + circle
                val checker = if (((sx / 8) + (sy / 8)) % 2 == 0) 24 else -24
                val dx = sx - 44
                val dy = sy - 30
                val inDisc = (dx * dx + dy * dy) <= 13 * 13
                val r = if (inDisc) 205 else (95 + ((sx * 2) % 60) + checker).coerceIn(25, 225)
                val g = if (inDisc) 152 else (112 + ((sy * 2) % 60) + checker).coerceIn(30, 225)
                val b = if (inDisc) 124 else (90 + (((sx + sy)) % 55) + checker).coerceIn(25, 225)
                out[y * width + x] = argb(r, g, b)
            }
        }
        return out
    }

    private fun frameMae(a: IntArray, b: IntArray): Double {
        var sum = 0L
        for (i in a.indices) {
            val pa = a[i]
            val pb = b[i]
            sum += abs(((pa ushr 16) and 0xFF) - ((pb ushr 16) and 0xFF))
            sum += abs(((pa ushr 8) and 0xFF) - ((pb ushr 8) and 0xFF))
            sum += abs((pa and 0xFF) - (pb and 0xFF))
        }
        return sum.toDouble() / (a.size * 3.0)
    }

    @Test
    fun `pillar 1 color enhancement corrects warm cast, boosts faded foliage vibrance, shields skin, and rolls off highlights`() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val profiles = EnhancementProfileRepository(context).loadProfiles()
        val adjuster = SmartAutoAdjustmentEngine()

        val dullWarmClip = VideoQualityAnalysis(
            avgBrightness = 114f,
            darkAreaRatio = 0.12f,
            overexposedAreaRatio = 0.03f,
            contrastStdDev = 34f,
            dynamicRange = 115f,
            noiseLevel = 4.8f,
            blurVariance = 160f,
            saturationMean = 42f, // Faded colors
            whiteBalanceShiftR = 0.88f, // Needs warmer reduction (-R)
            whiteBalanceShiftB = 1.14f, // Needs cool blue restoration (+B)
            compressionArtifactScore = 1.08f,
            edgeDensity = 0.16f,
            conditionTags = listOf("Faded Colors", "Low Contrast")
        )

        val params3x = adjuster.computeAdaptiveAdjustments(
            profiles.getValue(EnhancementMode.CLARITY_3X),
            dullWarmClip
        )

        // Moderate cast (0.88 / 1.14) must be detected and corrected (no longer ignored below 0.18!)
        assertTrue("Expected WB red gain < 0.96, got ${params3x.wbGainR}", params3x.wbGainR < 0.96f)
        assertTrue("Expected WB blue gain > 1.05, got ${params3x.wbGainB}", params3x.wbGainB > 1.05f)
        assertTrue("Expected strong vibrance >= 0.35 on faded clip, got ${params3x.vibranceGain}", params3x.vibranceGain >= 0.35f)

        // Build test swatches:
        // [0] Warm cast near-gray neutral (136, 128, 112)
        // [1] Natural human skin (208, 158, 128)
        // [2] Faded green foliage (96, 120, 94)
        // [3] Bright cloud highlight (238, 238, 238)
        val swatches = intArrayOf(
            argb(136, 128, 112),
            argb(208, 158, 128),
            argb(96, 120, 94),
            argb(238, 238, 238)
        )
        val enhanced = ColorMath.enhanceArgbFrameCopy(swatches, params3x)

        // 1) Gray neutral swatch loses warm R-B cast
        val castBefore = abs(((swatches[0] ushr 16) and 0xFF) - (swatches[0] and 0xFF))
        val castAfter = abs(((enhanced[0] ushr 16) and 0xFF) - (enhanced[0] and 0xFF))
        assertTrue("Warm color cast on neutral should drop ($castBefore -> $castAfter)", castAfter < castBefore)

        // 2) Faded foliage chroma increases visibly (> +25% chroma expansion), while skin chroma expansion stays restrained
        val foliageChromaBefore = rgbDiffMean(swatches[2])
        val foliageChromaAfter = rgbDiffMean(enhanced[2])
        val skinChromaBefore = rgbDiffMean(swatches[1])
        val skinChromaAfter = rgbDiffMean(enhanced[1])
        val foliageRelBoost = (foliageChromaAfter - foliageChromaBefore) / foliageChromaBefore
        val skinRelBoost = abs(skinChromaAfter - skinChromaBefore) / skinChromaBefore
        assertTrue(
            "Foliage vibrance boost (${foliageRelBoost}) should visibly lift dull colors (> 0.25)",
            foliageRelBoost > 0.25f
        )
        assertTrue(
            "Skin-tone shield must restrain skin chroma shift ($skinRelBoost) well below foliage ($foliageRelBoost)",
            skinRelBoost < foliageRelBoost * 0.55f
        )

        // 3) Highlight shoulder preserves headroom without hard clipping at 255
        val highlightLumAfter = lumaOf(enhanced[3])
        assertTrue("Highlight should avoid hard clip at 255 (was $highlightLumAfter)", highlightLumAfter < 252f)

        // 4) Vibrance LUT maps centered neutral 128 -> 128 cleanly with symmetric expansion
        val vibLut = ColorMath.buildChromaVibranceLutArray(params3x.vibranceGain, params3x.saturationGain)
        assertEquals(128, vibLut[128].toInt() and 0xFF)
        assertTrue((vibLut[148].toInt() and 0xFF) > 148)
        assertTrue((vibLut[108].toInt() and 0xFF) < 108)

        println(
            String.format(
                Locale.US,
                "[PILLAR1_METRICS] cast=%d->%d, foliageChroma=%.1f->%.1f (+%.0f%%), skinShift=%.0f%%, highlightLum=%.1f",
                castBefore, castAfter, foliageChromaBefore, foliageChromaAfter, foliageRelBoost * 100f, skinRelBoost * 100f, highlightLumAfter
            )
        )
    }

    @Test
    fun `pillar 2 quality enhancement wires deblur, detail recovery, halo-safe sharpening, and progressive DnCNN scheduling`() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val profiles = EnhancementProfileRepository(context).loadProfiles()
        val adjuster = SmartAutoAdjustmentEngine()

        // Verify v6 profile tuning and ONNX assets exist cleanly
        val p1Prof = profiles.getValue(EnhancementMode.CLARITY_1X)
        val p4Prof = profiles.getValue(EnhancementMode.CLARITY_4X)
        assertTrue(p1Prof.detailRecoveryPercent < p4Prof.detailRecoveryPercent)
        assertTrue(p1Prof.deblurPercent < p4Prof.deblurPercent)
        assertTrue(p1Prof.aiSharpeningPercent < p4Prof.aiSharpeningPercent)

        val blurrySource = VideoQualityAnalysis(
            avgBrightness = 110f,
            darkAreaRatio = 0.14f,
            overexposedAreaRatio = 0.02f,
            contrastStdDev = 41f,
            dynamicRange = 135f,
            noiseLevel = 9.5f,
            blurVariance = 72f,
            saturationMean = 88f,
            whiteBalanceShiftR = 1.0f,
            whiteBalanceShiftB = 1.0f,
            compressionArtifactScore = 1.35f,
            edgeDensity = 0.14f,
            conditionTags = listOf("High Noise", "Soft / Blurry", "Compression Artifacts")
        )

        val adj1 = adjuster.computeAdaptiveAdjustments(profiles.getValue(EnhancementMode.CLARITY_1X), blurrySource)
        val adj2 = adjuster.computeAdaptiveAdjustments(profiles.getValue(EnhancementMode.CLARITY_2X), blurrySource)
        val adj3 = adjuster.computeAdaptiveAdjustments(profiles.getValue(EnhancementMode.CLARITY_3X), blurrySource)
        val adj4 = adjuster.computeAdaptiveAdjustments(profiles.getValue(EnhancementMode.CLARITY_4X), blurrySource)

        // All real-detail parameters strictly progress across 1x -> 2x -> 3x -> 4x and remain non-trivial on 1x
        assertTrue(adj1.detailRecovery >= 0.45f)
        assertTrue(adj1.deblurStrength >= 0.40f)
        assertTrue(adj1.detailRecovery < adj2.detailRecovery && adj2.detailRecovery < adj3.detailRecovery && adj3.detailRecovery < adj4.detailRecovery)
        assertTrue(adj1.deblurStrength < adj2.deblurStrength && adj2.deblurStrength < adj3.deblurStrength && adj3.deblurStrength < adj4.deblurStrength)
        assertTrue(adj1.aiSharpening < adj2.aiSharpening && adj2.aiSharpening < adj3.aiSharpening && adj3.aiSharpening < adj4.aiSharpening)

        // SpeedBudgetGovernor must schedule DnCNN denoising on 100% of frames without frame skipping on FULL_NEURAL & SINGLE_TILE tiers
        val gov4x = SpeedBudgetGovernor(
            mode = EnhancementMode.CLARITY_4X,
            durationMs = 60_000L,
            estimatedFrames = 1800,
            presetTileSize = 256,
            presetTilesCap = 4,
            measuredSrTileMs = 24L,
            deviceCapacityScore = 90
        )
        assertEquals(SpeedExecutionTier.FULL_NEURAL_PER_FRAME, gov4x.activeTier)
        assertTrue(gov4x.decisionForFrame(0, isSceneCut = false, forceLightweight = false).allowDnCnn)
        assertTrue(gov4x.decisionForFrame(1, isSceneCut = false, forceLightweight = false).allowDnCnn)
        assertTrue(gov4x.decisionForFrame(3, isSceneCut = false, forceLightweight = false).allowDnCnn)

        val govMid2x = SpeedBudgetGovernor(
            mode = EnhancementMode.CLARITY_2X,
            durationMs = 60_000L,
            estimatedFrames = 1800,
            presetTileSize = 192,
            presetTilesCap = 2,
            measuredSrTileMs = 115L,
            deviceCapacityScore = 65
        )
        assertEquals(SpeedExecutionTier.SINGLE_TILE_NEURAL, govMid2x.activeTier)
        assertTrue(govMid2x.decisionForFrame(0, isSceneCut = false, forceLightweight = false).allowDnCnn)
        assertTrue(govMid2x.decisionForFrame(1, isSceneCut = false, forceLightweight = false).allowDnCnn)
        assertTrue(govMid2x.decisionForFrame(6, isSceneCut = false, forceLightweight = false).allowDnCnn)
    }

    @Test
    fun `pillar 3 motion smoothness interpolates moving scene more accurately than frame hold and preserves 1x duration`() {
        val w = 96
        val h = 64
        val frameT0 = synthesizeMovingScene(w, h, shiftX = 0, shiftY = 0)
        val frameT1GroundTruth = synthesizeMovingScene(w, h, shiftX = 3, shiftY = 2)
        val frameT2 = synthesizeMovingScene(w, h, shiftX = 6, shiftY = 4)

        val flowMid = MotionSmoothMath.synthesizeMidpointArgb(
            prevArgb = frameT0,
            nextArgb = frameT2,
            width = w,
            height = h,
            alpha = 0.5f,
            blockSize = 8,
            searchRadius = 8
        )

        val maeNaiveHold = frameMae(frameT0, frameT1GroundTruth)
        val maeOpticalFlow = frameMae(flowMid, frameT1GroundTruth)
        assertTrue(
            "Optical-flow midpoint MAE ($maeOpticalFlow) must dramatically beat naive hold ($maeNaiveHold)",
            maeOpticalFlow < maeNaiveHold * 0.55
        )

        // Cadence doubling checks
        assertTrue(MotionSmoothMath.shouldInterpolateCadence(15.0, allowInterpolation = true))
        assertTrue(MotionSmoothMath.shouldInterpolateCadence(24.0, allowInterpolation = true))
        assertFalse(MotionSmoothMath.shouldInterpolateCadence(30.0, allowInterpolation = true))
        assertEquals(48.0, MotionSmoothMath.targetOutputFps(24.0, cadenceInterpolated = true), 0.01)
        assertEquals(30.0, MotionSmoothMath.targetOutputFps(30.0, cadenceInterpolated = false), 0.01)

        // VFR dropped frame detection + PTS preservation (exact duration lock)
        val step30 = 33_333L
        assertEquals(0, MotionSmoothMath.detectDroppedFrameGap(0L, 33_333L, step30))
        assertEquals(1, MotionSmoothMath.detectDroppedFrameGap(33_333L, 100_000L, step30))
        assertEquals(2, MotionSmoothMath.detectDroppedFrameGap(33_333L, 133_333L, step30))

        assertTrue(MotionSmoothMath.isStuckDuplicateFrame(prevMad = 4.5f, currToPrevMad = 0.12f))
        assertFalse(MotionSmoothMath.isStuckDuplicateFrame(prevMad = 4.5f, currToPrevMad = 2.8f))

        val sourcePts = (0 until 10).map { it * 41_667L } // 24fps span ~ 375,003 us
        val interpPts = MotionSmoothMath.buildInterpolatedPtsSequence(sourcePts, 41_667L)
        assertEquals(19, interpPts.size)
        assertEquals(sourcePts.first(), interpPts.first())
        assertEquals(sourcePts.last(), interpPts.last())
        for (i in 1 until interpPts.size) {
            assertTrue("PTS must be strictly monotonic at $i", interpPts[i] > interpPts[i - 1])
        }

        println(
            String.format(
                Locale.US,
                "[PILLAR3_METRICS] naiveHoldMAE=%.2f, opticalFlowMidpointMAE=%.2f (%.0f%% error reduction), ptsSpan=%d->%d us (%d frames)",
                maeNaiveHold, maeOpticalFlow, ((maeNaiveHold - maeOpticalFlow) / maeNaiveHold) * 100.0,
                sourcePts.last(), interpPts.last(), interpPts.size
            )
        )
    }

    @Test
    fun `pillar 4 combined four-system enhancement and jvm microbenchmark complete cleanly inside speed ceilings`() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val profiles = EnhancementProfileRepository(context).loadProfiles()
        val caps = sampleCapabilities()
        val adjuster = SmartAutoAdjustmentEngine()

        val meta24fps = VideoMetadata(
            uri = Uri.EMPTY,
            displayName = "sample_24fps_720p.mp4",
            width = 1280,
            height = 720,
            rotationDegrees = 0,
            displayWidth = 1280,
            displayHeight = 720,
            fps = 24.0,
            durationMs = 60_000L,
            estimatedFrameCount = 1440,
            bitrateBps = 2_400_000L,
            videoCodec = "h264",
            hasAudio = true,
            audioCodec = "aac",
            audioSampleRate = 48000,
            audioChannels = 2,
            fileSizeBytes = 18_000_000L
        )

        // Verify Resolution Scaling + Neural Working Dimensions + Tile Planning
        val out4x = ResolutionCalculator.calculate(meta24fps, EnhancementMode.CLARITY_4X, profiles.getValue(EnhancementMode.CLARITY_4X), caps)
        assertEquals(1920 to 1080, out4x.width to out4x.height)
        val work4x = AiTilingMath.computeNeuralWorkingSize(1280, 720, out4x.width, out4x.height, out4x.aiTileSize, out4x.tilesPerFrame)
        val grid4x = AiTilingMath.planUpscaleTileGrid(work4x.first, work4x.second, out4x.width, out4x.height, out4x.aiTileSize, out4x.tilesPerFrame)
        assertTrue(grid4x.totalTiles in 1..4)

        val analysis = VideoQualityAnalysis(
            avgBrightness = 104f,
            darkAreaRatio = 0.18f,
            overexposedAreaRatio = 0.02f,
            contrastStdDev = 36f,
            dynamicRange = 125f,
            noiseLevel = 7.2f,
            blurVariance = 92f,
            saturationMean = 54f,
            whiteBalanceShiftR = 0.91f,
            whiteBalanceShiftB = 1.11f,
            compressionArtifactScore = 1.24f,
            edgeDensity = 0.15f,
            conditionTags = listOf("Soft / Blurry", "Faded Colors")
        )

        val params = adjuster.computeAdaptiveAdjustments(profiles.getValue(EnhancementMode.CLARITY_4X), analysis)
        val logger = EngineDiagnosticsLogger()
        val governor = SpeedBudgetGovernor(
            mode = EnhancementMode.CLARITY_4X,
            durationMs = meta24fps.durationMs,
            estimatedFrames = meta24fps.estimatedFrameCount,
            presetTileSize = out4x.aiTileSize,
            presetTilesCap = out4x.tilesPerFrame,
            measuredSrTileMs = 28L,
            deviceCapacityScore = caps.processingCapacityScore
        )

        // Run combined Color + Smoothness + Governor simulation across 15 motion frames (96x64)
        val w = 96
        val h = 64
        val t0 = System.nanoTime()
        var prevFrame: IntArray? = null
        var producedFrames = 0
        var interpCount = 0
        for (i in 0 until 15) {
            val currRaw = synthesizeMovingScene(w, h, shiftX = i * 2, shiftY = i)
            val currColor = ColorMath.enhanceArgbFrameCopy(currRaw, params)
            governor.decisionForFrame(i, isSceneCut = (i == 0), forceLightweight = false)
            if (prevFrame != null && MotionSmoothMath.shouldInterpolateCadence(meta24fps.fps, params.allowFrameInterpolation)) {
                val midRaw = MotionSmoothMath.synthesizeMidpointArgb(prevFrame, currRaw, w, h, alpha = 0.5f)
                val midColor = ColorMath.enhanceArgbFrameCopy(midRaw, params)
                assertEquals(w * h, midColor.size)
                interpCount++
                producedFrames++
            }
            prevFrame = currRaw
            producedFrames++
            assertEquals(w * h, currColor.size)
        }
        val elapsedMs = (System.nanoTime() - t0) / 1_000_000.0
        val report = logger.buildReport(
            totalMs = elapsedMs.roundToInt().toLong(),
            executionProvider = "SR:GPU • DN:NPU • Face:CPU",
            benchmarkSummary = "4-Pillar JVM Integration",
            codecUsed = "H.265 (Hardware)",
            initialTileSize = out4x.aiTileSize,
            finalTileSize = out4x.aiTileSize,
            framesProcessed = producedFrames,
            recoveredFrames = 0,
            analysisReusedFromCache = true,
            upscaleScaleLabel = "3.0x",
            interpolatedFramesCount = interpCount,
            stutterRepairsCount = 0,
            colorEnhancedFrames = producedFrames,
            smoothnessModeLabel = "24.0 -> 48.0 fps Optical Flow"
        )

        assertEquals(29, report.framesProcessed)
        assertEquals(14, report.interpolatedFramesCount)
        assertEquals(29, report.colorEnhancedFrames)
        assertTrue(elapsedMs < 3_000.0)

        println(
            String.format(
                Locale.US,
                "[PILLAR4_BENCHMARK] 15->29 frames (96x64 Color+OpticalFlow) completed in %.1f ms (%.2f ms/output-frame); 4K output=%dx%d (%d tiles)",
                elapsedMs, elapsedMs / producedFrames.toDouble(), out4x.width, out4x.height, grid4x.totalTiles
            )
        )
    }
}
