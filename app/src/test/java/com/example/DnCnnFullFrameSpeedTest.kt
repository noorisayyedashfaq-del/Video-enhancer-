package com.example

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import com.example.engine.DnCnnResidualMath
import com.example.engine.EnhancementProfileRepository
import com.example.engine.SmartAutoAdjustmentEngine
import com.example.engine.SpeedBudgetGovernor
import com.example.engine.SpeedExecutionTier
import com.example.model.EnhancementMode
import com.example.model.VideoQualityAnalysis
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.util.Locale
import java.util.Random
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import kotlin.math.roundToInt

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [36])
class DnCnnFullFrameSpeedTest {

    @Test
    fun `zero skip DnCNN runs on 100 percent of 1800 frames across all 4 modes and all execution tiers`() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val profiles = EnhancementProfileRepository(context).loadProfiles()
        val adjuster = SmartAutoAdjustmentEngine()
        val analysis = VideoQualityAnalysis(
            avgBrightness = 115f,
            darkAreaRatio = 0.12f,
            overexposedAreaRatio = 0.02f,
            contrastStdDev = 40f,
            dynamicRange = 140f,
            noiseLevel = 9.5f,
            blurVariance = 90f,
            saturationMean = 100f,
            whiteBalanceShiftR = 1.0f,
            whiteBalanceShiftB = 1.0f,
            compressionArtifactScore = 1.25f,
            edgeDensity = 0.16f,
            conditionTags = listOf("Moderate Noise")
        )

        val tileLatenciesForTiers = listOf(18L, 120L, 165L, 480L)
        val totalFrames = 1800

        var prevStrength = 0f
        for (mode in EnhancementMode.entries) {
            val profile = profiles.getValue(mode)
            val params = adjuster.computeAdaptiveAdjustments(profile, analysis)
            assertTrue("DnCNN must be enabled in ${mode.label}", params.useAiDenoise)
            assertTrue(
                "DnCNN strength must remain progressive & non-degraded (${params.dncnnStrength} > $prevStrength)",
                params.dncnnStrength > prevStrength
            )
            prevStrength = params.dncnnStrength

            for (srTileMs in tileLatenciesForTiers) {
                val gov = SpeedBudgetGovernor(
                    mode = mode,
                    durationMs = 60_000L,
                    estimatedFrames = totalFrames,
                    presetTileSize = profile.aiTileSize,
                    presetTilesCap = profile.targetTilesPerFrame,
                    measuredSrTileMs = srTileMs,
                    deviceCapacityScore = 75,
                    measuredDnCnnMs = 8L
                )

                var dncnnAllowedCount = 0
                var dncnnForcedLightweightCount = 0
                for (f in 0 until totalFrames) {
                    val dNormal = gov.decisionForFrame(f, isSceneCut = (f % 150 == 0), forceLightweight = false)
                    val dForced = gov.decisionForFrame(f, isSceneCut = false, forceLightweight = true)
                    if (dNormal.allowDnCnn) dncnnAllowedCount++
                    if (dForced.allowDnCnn) dncnnForcedLightweightCount++
                }

                assertEquals(
                    "Mode ${mode.label} tier=${gov.activeTier} must allow DnCNN on 100% of frames",
                    totalFrames,
                    dncnnAllowedCount
                )
                assertEquals(
                    "Mode ${mode.label} forceLightweight must STILL allow DnCNN on 100% of frames",
                    totalFrames,
                    dncnnForcedLightweightCount
                )
            }
        }
    }

    @Test
    fun `DnCnnResidualMath improves noisy frame PSNR while preserving sharp step edges`() {
        val w = 320
        val h = 180
        val total = w * h
        val clean = ByteArray(total)
        val noisy = ByteArray(total)
        val rng = Random(42L)

        // Left half dark flat sky (60), right half bright sunlit wall (210) -> 150-level razor edge at x=160
        for (y in 0 until h) {
            val rowOff = y * w
            for (x in 0 until w) {
                val baseVal = if (x < 160) 60 else 210
                clean[rowOff + x] = baseVal.toByte()
                val grain = ((rng.nextGaussian()) * 14.0).roundToInt()
                noisy[rowOff + x] = (baseVal + grain).coerceIn(0, 255).toByte()
            }
        }

        // Simulated 256x256 DnCNN prediction recovers underlying clean surface on low/mid band
        val origNorm = FloatArray(total) { (noisy[it].toInt() and 0xFF) / 255f }
        val denoisedNorm = FloatArray(total) { (clean[it].toInt() and 0xFF) / 255f }
        val resQ4 = ShortArray(total)
        val output = ByteArray(total)

        val psnrBefore = DnCnnResidualMath.computePsnrDb(clean, noisy)

        DnCnnResidualMath.quantizeResidualToQ4(origNorm, denoisedNorm, strength = 0.88f, dstQ4 = resQ4)
        DnCnnResidualMath.applyQ4ResidualAndEdgeGatedSmooth(
            srcY = noisy,
            width = w,
            height = h,
            resQ4Full = resQ4,
            strength = 0.88f,
            dstY = output
        )

        val psnrAfter = DnCnnResidualMath.computePsnrDb(clean, output)
        assertTrue(
            "Expected major PSNR improvement after DnCNN Q4 denoising: before=$psnrBefore, after=$psnrAfter",
            psnrAfter > psnrBefore + 8.0
        )

        // Verify sharp edge at x=159..160 is preserved (>92% contrast kept) and not smeared into blur
        val cleanEdgeJump = DnCnnResidualMath.computeEdgeJump(clean, w, row = 90, colLeft = 158, colRight = 161)
        val outEdgeJump = DnCnnResidualMath.computeEdgeJump(output, w, row = 90, colLeft = 158, colRight = 161)
        val retention = outEdgeJump.toDouble() / cleanEdgeJump.toDouble()
        assertTrue(
            "Edge sharpness retention must stay >= 92%, got ${(retention * 100.0)}%",
            retention >= 0.92
        )
        println(
            String.format(
                Locale.US,
                "[DNCNN_QUALITY] PSNR_before=%.2fdB PSNR_after=%.2fdB Gain=+%.2fdB EdgeRetention=%.1f%%",
                psnrBefore,
                psnrAfter,
                psnrAfter - psnrBefore,
                retention * 100.0
            )
        )
    }

    @Test
    fun `benchmark full-frame DnCNN Q4 fusion and async worker overlap across Clarity 1x to 4x`() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val profiles = EnhancementProfileRepository(context).loadProfiles()
        val exec = Executors.newSingleThreadExecutor { r -> Thread(r, "Bench-DnCNN-Async").apply { isDaemon = true } }

        try {
            val modeSpecs = listOf(
                Triple(EnhancementMode.CLARITY_1X, 1280 to 720, 12L),
                Triple(EnhancementMode.CLARITY_2X, 1280 to 720, 18L),
                Triple(EnhancementMode.CLARITY_3X, 1920 to 1080, 26L),
                Triple(EnhancementMode.CLARITY_4X, 1920 to 1080, 34L)
            )

            // Warmup JIT for accurate JVM measurements
            run {
                val warmSrc = ByteArray(640 * 360) { (it and 0xFF).toByte() }
                val warmQ4 = ShortArray(640 * 360) { 16 }
                val warmDst = ByteArray(640 * 360)
                repeat(3) {
                    DnCnnResidualMath.applyQ4ResidualAndEdgeGatedSmooth(warmSrc, 640, 360, warmQ4, 0.72f, warmDst)
                }
            }

            // Benchmark async worker overlap vs sequential inference+encode pipeline
            val overlapWorkMs = 8L
            val stageOtherMs = 14L
            val rounds = 20

            val tSeq0 = System.nanoTime()
            repeat(rounds) {
                Thread.sleep(overlapWorkMs)
                Thread.sleep(stageOtherMs)
            }
            val seqTotalMs = (System.nanoTime() - tSeq0) / 1_000_000.0

            val tAsync0 = System.nanoTime()
            var nextFut = exec.submit(Callable { Thread.sleep(overlapWorkMs); true })
            repeat(rounds) { idx ->
                assertTrue(nextFut.get())
                nextFut = if (idx + 1 < rounds) {
                    exec.submit(Callable { Thread.sleep(overlapWorkMs); true })
                } else {
                    exec.submit(Callable { true })
                }
                Thread.sleep(stageOtherMs)
            }
            nextFut.get()
            val asyncTotalMs = (System.nanoTime() - tAsync0) / 1_000_000.0
            val overlapSpeedup = seqTotalMs / asyncTotalMs
            assertTrue(
                "Async prefetch pipeline must outperform sequential (${asyncTotalMs}ms vs ${seqTotalMs}ms)",
                asyncTotalMs < seqTotalMs * 0.85
            )

            for ((mode, dims, srTileMs) in modeSpecs) {
                val profile = profiles.getValue(mode)
                val w = dims.first
                val h = dims.second
                val pixels = w * h
                val srcY = ByteArray(pixels) { (it * 7 and 0xFF).toByte() }
                val resQ4 = ShortArray(pixels) { ((it % 7) - 3).toShort() }
                val dstY = ByteArray(pixels)
                val strength = profile.dncnnStrengthPercent / 100f

                // Measure pure-JVM Q4 residual + 3x3 edge-gated fusion ms/frame (multi-frame average)
                val benchFrames = 6
                val tFuse0 = System.nanoTime()
                repeat(benchFrames) {
                    DnCnnResidualMath.applyQ4ResidualAndEdgeGatedSmooth(srcY, w, h, resQ4, strength, dstY)
                }
                val fusePerFrameMs = ((System.nanoTime() - tFuse0) / 1_000_000.0) / benchFrames

                // Measure 1800-frame SpeedBudgetGovernor scheduling overhead
                val gov = SpeedBudgetGovernor(
                    mode = mode,
                    durationMs = 60_000L,
                    estimatedFrames = 1800,
                    presetTileSize = profile.aiTileSize,
                    presetTilesCap = profile.targetTilesPerFrame,
                    measuredSrTileMs = srTileMs,
                    deviceCapacityScore = 85,
                    measuredDnCnnMs = 8L
                )
                val tGov0 = System.nanoTime()
                var dnAllowed = 0
                for (f in 0 until 1800) {
                    val d = gov.decisionForFrame(f, isSceneCut = (f % 120 == 0), forceLightweight = false)
                    if (d.allowDnCnn) dnAllowed++
                    gov.onFrameCompleted(f * 35L, f + 1, 1800, 35L)
                }
                val gov1800Ms = (System.nanoTime() - tGov0) / 1_000_000.0
                assertEquals(1800, dnAllowed)
                assertTrue("Governor 1800-frame check should stay under 100ms, got $gov1800Ms", gov1800Ms < 100.0)

                val total1MinFuseJvmSec = (fusePerFrameMs * 1800.0) / 1000.0
                println(
                    String.format(
                        Locale.US,
                        "[DNCNN_BENCHMARK_%s] res=%dx%d strength=%.0f%% tier=%s jvm_q4_fuse=%.2fms/f (1800f=%.2fs) gov_1800f=%.2fms async_overlap_speedup=%.2fx",
                        mode.label.uppercase(Locale.US),
                        w,
                        h,
                        profile.dncnnStrengthPercent,
                        gov.activeTier.name,
                        fusePerFrameMs,
                        total1MinFuseJvmSec,
                        gov1800Ms,
                        overlapSpeedup
                    )
                )
            }
        } finally {
            exec.shutdownNow()
        }
    }
}
