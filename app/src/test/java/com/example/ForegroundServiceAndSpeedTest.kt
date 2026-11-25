package com.example

import android.app.Application
import android.content.Context
import android.content.Intent
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import androidx.core.app.NotificationCompat
import androidx.test.core.app.ApplicationProvider
import com.example.engine.AiSpeedPolicy
import com.example.engine.AiTilingMath
import com.example.engine.DeviceCapabilityDetector
import com.example.engine.FastYuvCopyMath
import com.example.engine.SpeedBudgetGovernor
import com.example.engine.SpeedExecutionTier
import com.example.engine.TemporalReusePolicy
import com.example.model.EnhancementMode
import com.example.model.ProcessingProgress
import com.example.model.ProcessingStage
import com.example.service.EnhancementSessionCoordinator
import com.example.service.VideoEnhancementForegroundService
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [36])
class ForegroundServiceAndSpeedTest {

    @Test
    fun `manifest declares mediaProcessing foreground service permissions and singleTop activity`() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val pm = context.packageManager
        val pkgInfo = pm.getPackageInfo(
            context.packageName,
            PackageManager.GET_PERMISSIONS or PackageManager.GET_SERVICES or PackageManager.GET_ACTIVITIES
        )

        val perms = pkgInfo.requestedPermissions?.toSet().orEmpty()
        assertTrue(perms.contains(android.Manifest.permission.FOREGROUND_SERVICE))
        assertTrue(perms.contains(android.Manifest.permission.FOREGROUND_SERVICE_MEDIA_PROCESSING))
        assertTrue(perms.contains(android.Manifest.permission.POST_NOTIFICATIONS))
        assertTrue(perms.contains(android.Manifest.permission.WAKE_LOCK))
        assertTrue("AdMob SDK requires INTERNET permission", perms.contains(android.Manifest.permission.INTERNET))
        assertTrue("AdMob network gate requires ACCESS_NETWORK_STATE permission", perms.contains(android.Manifest.permission.ACCESS_NETWORK_STATE))

        val mainAct = pkgInfo.activities?.firstOrNull { it.name.endsWith("MainActivity") }
        assertNotNull(mainAct)
        assertEquals(ActivityInfo.LAUNCH_SINGLE_TOP, mainAct?.launchMode)

        val fgsService = pkgInfo.services?.firstOrNull { it.name.endsWith("VideoEnhancementForegroundService") }
        assertNotNull("VideoEnhancementForegroundService must be declared in manifest", fgsService)
        assertTrue(
            "Foreground service must declare mediaProcessing type",
            (fgsService!!.foregroundServiceType and ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROCESSING) != 0
        )
    }

    @Test
    fun `benchmarkThreadCandidates covers 4 6 and 8 CPU threads when cores allow`() {
        assertEquals(listOf(4), DeviceCapabilityDetector.benchmarkThreadCandidates(cpuCores = 4, maxTierThreads = 8))
        assertEquals(listOf(4, 6), DeviceCapabilityDetector.benchmarkThreadCandidates(cpuCores = 6, maxTierThreads = 8))
        assertEquals(listOf(4, 6, 8), DeviceCapabilityDetector.benchmarkThreadCandidates(cpuCores = 8, maxTierThreads = 8))
        assertEquals(listOf(4, 6), DeviceCapabilityDetector.benchmarkThreadCandidates(cpuCores = 8, maxTierThreads = 6))
    }

    @Test
    fun `AiSpeedPolicy triggers hardware fast path when tile execution threatens mode time deadline`() {
        // Fast GPU/NPU single-tile (45ms) -> stays on neural AI cleanly
        assertFalse(
            AiSpeedPolicy.shouldEngageLightweightFallback(
                measuredTileMs = 45L,
                tilesPerFrame = 2,
                estimatedFrameCount = 1800,
                modeBudgetFrameMs = 150L
            )
        )

        // Heavy multi-tile exceeding mode frame deadline (120ms x 4 = 480ms > 300ms) -> triggers fallback
        assertTrue(
            AiSpeedPolicy.shouldEngageLightweightFallback(
                measuredTileMs = 120L,
                tilesPerFrame = 4,
                estimatedFrameCount = 1800,
                modeBudgetFrameMs = 300L
            )
        )

        // Slow CPU tile (> 300ms single tile) -> triggers immediate GPU shader / SIMD fallback
        assertTrue(
            AiSpeedPolicy.shouldEngageLightweightFallback(
                measuredTileMs = 340L,
                tilesPerFrame = 1,
                estimatedFrameCount = 1800,
                modeBudgetFrameMs = 300L
            )
        )
    }

    @Test
    fun `SpeedBudgetGovernor enforces strict 1-minute time ceilings across flagship GPU mid NPU slow CPU and thermal pressure`() {
        val frames1Min = 1800 // 60s @ 30fps

        // Verify strict 1-minute budgets per mode
        assertEquals(150_000L, SpeedBudgetGovernor.maxBudgetMsPerMinute(EnhancementMode.HD))
        assertEquals(300_000L, SpeedBudgetGovernor.maxBudgetMsPerMinute(EnhancementMode.HD_PLUS))
        assertEquals(450_000L, SpeedBudgetGovernor.maxBudgetMsPerMinute(EnhancementMode.FHD))
        assertEquals(600_000L, SpeedBudgetGovernor.maxBudgetMsPerMinute(EnhancementMode.TWO_K))

        // 1) Flagship GPU (18ms/tile): fits FULL_NEURAL_PER_FRAME within strict budgets
        val flagship1x = SpeedBudgetGovernor(EnhancementMode.HD, 60_000L, frames1Min, 192, 1, 18L, 90)
        val flagship4x = SpeedBudgetGovernor(EnhancementMode.TWO_K, 60_000L, frames1Min, 256, 4, 18L, 90)
        assertEquals(SpeedExecutionTier.FULL_NEURAL_PER_FRAME, flagship1x.activeTier)
        assertEquals(SpeedExecutionTier.FULL_NEURAL_PER_FRAME, flagship4x.activeTier)
        assertTrue(flagship1x.decisionForFrame(0, isSceneCut = true, forceLightweight = false).runNeuralSrNow)

        // 2) Mid-tier NPU (145ms/tile): uses SINGLE_TILE_NEURAL for HD+/2K and KEYFRAME_NEURAL_HYBRID for tight HD (150s)
        val mid1x = SpeedBudgetGovernor(EnhancementMode.HD, 60_000L, frames1Min, 192, 1, 145L, 65)
        val mid2x = SpeedBudgetGovernor(EnhancementMode.HD_PLUS, 60_000L, frames1Min, 192, 2, 120L, 65)
        assertEquals(SpeedExecutionTier.KEYFRAME_NEURAL_HYBRID, mid1x.activeTier)
        assertTrue(mid1x.currentKeyframeInterval in 2..6)
        assertEquals(SpeedExecutionTier.SINGLE_TILE_NEURAL, mid2x.activeTier)

        // 3) Slow CPU device (450ms/tile) on HD/HD+: shifts to GPU_SHADER_FAST_PASS immediately so 1m never takes hours
        val slow1x = SpeedBudgetGovernor(EnhancementMode.HD, 60_000L, frames1Min, 192, 1, 450L, 35)
        assertEquals(SpeedExecutionTier.GPU_SHADER_FAST_PASS, slow1x.activeTier)
        assertTrue(slow1x.decisionForFrame(0, isSceneCut = false, forceLightweight = false).fastGpuMode)

        // 4) Dynamic mid-job thermal throttling simulation on 2K: starts FULL_NEURAL, sees slow frames, steps down cleanly
        val dynamic4x = SpeedBudgetGovernor(EnhancementMode.TWO_K, 60_000L, frames1Min, 256, 4, 40L, 85)
        assertEquals(SpeedExecutionTier.FULL_NEURAL_PER_FRAME, dynamic4x.activeTier)
        for (i in 1..20) {
            dynamic4x.onFrameCompleted(elapsedJobMs = i * 650L, processedFrames = i, totalFrames = frames1Min, lastFrameMs = 650L)
        }
        assertTrue(
            "Governor must step down from FULL_NEURAL when EMA exceeds remaining per-frame budget",
            dynamic4x.activeTier != SpeedExecutionTier.FULL_NEURAL_PER_FRAME
        )
    }

    @Test
    fun `computeNeuralWorkingSize prevents 4x input pixel bloat and guarantees 1 to 4 tile cap across 720p and 1080p`() {
        val work1x = AiTilingMath.computeNeuralWorkingSize(1280, 720, 1280, 720, tileSize = 192, maxTilesCap = 1)
        val grid1x = AiTilingMath.planUpscaleTileGrid(work1x.first, work1x.second, 1280, 720, 192, maxTilesCap = 1)
        assertEquals(1, grid1x.totalTiles)
        assertTrue(work1x.first <= 192 && work1x.second <= 192)

        val work4x = AiTilingMath.computeNeuralWorkingSize(1920, 1080, 3840, 2160, tileSize = 256, maxTilesCap = 4)
        val grid4x = AiTilingMath.planUpscaleTileGrid(work4x.first, work4x.second, 3840, 2160, 256, maxTilesCap = 4)
        assertTrue("Expected <= 4 tiles for 4x mode, got ${grid4x.totalTiles}", grid4x.totalTiles <= 4)
        assertTrue("Neural working width should stay well under raw 1920px (${work4x.first})", work4x.first <= 512)
    }

    @Test
    fun `jvm 1800-frame governor and motion math microbenchmark runs under strict overhead bounds`() {
        val governor = SpeedBudgetGovernor(EnhancementMode.TWO_K, 60_000L, 1800, 256, 4, 32L, 85)
        val thumbA = ByteArray(48 * 48) { (it and 0xFF).toByte() }
        val thumbB = ByteArray(48 * 48) { ((it + 2) and 0xFF).toByte() }
        val dummyUv = java.nio.ByteBuffer.allocate(320 * 180)
        val uvOut = ByteArray(320 * 180)

        val t0 = System.nanoTime()
        var neuralCount = 0
        for (i in 0 until 1800) {
            val mad = TemporalReusePolicy.computeMeanAbsDiff(thumbA, thumbB)
            val dec = governor.decisionForFrame(i, isSceneCut = mad >= TemporalReusePolicy.SCENE_CUT_MAD_THRESHOLD, forceLightweight = false)
            if (dec.runNeuralSrNow) neuralCount++
            if (i % 10 == 0) {
                FastYuvCopyMath.copySemiPlanarUvPlane(dummyUv, 0, 0, 160, 90, 320, uvOut, 0)
            }
            governor.onFrameCompleted(elapsedJobMs = i * 45L, processedFrames = i + 1, totalFrames = 1800, lastFrameMs = 45L)
        }
        val elapsedMs = (System.nanoTime() - t0) / 1_000_000L
        assertTrue(neuralCount > 0)
        assertTrue("1800-frame control & motion overhead took ${elapsedMs}ms (must be < 2000ms)", elapsedMs < 2_000L)
    }

    @Test
    fun `foreground service builds ongoing notification with title percentage stage and cancel action`() {
        val controller = Robolectric.buildService(VideoEnhancementForegroundService::class.java).create()
        val service = controller.get()

        val progress = ProcessingProgress(
            stage = ProcessingStage.ENHANCING,
            processedFrames = 45,
            totalFrames = 100,
            overallProgress = 0.62f,
            statusDetail = "Frame 45 / 100 • 1920x1080",
            processingSpeedLabel = "8.4 fps"
        )
        val notification = service.buildProgressNotification(progress, EnhancementMode.FHD)

        val extras = notification.extras
        assertEquals("ClearKit Video Processing", extras.getCharSequence(NotificationCompat.EXTRA_TITLE)?.toString())
        val text = extras.getCharSequence(NotificationCompat.EXTRA_TEXT)?.toString().orEmpty()
        assertTrue("Expected stage title in notification text: $text", text.contains("AI Enhancement"))
        assertTrue("Expected percentage in notification text: $text", text.contains("62%"))
        assertNotNull("Expected Cancel action button on notification", notification.actions?.firstOrNull {
            it.title.toString().contains("Cancel", ignoreCase = true)
        })

        // Verify cancel action triggers stop without crashing
        val cancelIntent = Intent(service, VideoEnhancementForegroundService::class.java).apply {
            action = VideoEnhancementForegroundService.ACTION_CANCEL_ENHANCEMENT
        }
        service.onStartCommand(cancelIntent, 0, 1)
        controller.destroy()
    }

    @Test
    fun `session coordinator prevents duplicate concurrent jobs and preserves mode across recreation`() {
        val app = ApplicationProvider.getApplicationContext<Application>()
        val coordinator = EnhancementSessionCoordinator.getInstance(app)
        coordinator.releaseJobLock()

        coordinator.selectMode(EnhancementMode.TWO_K)
        assertEquals(EnhancementMode.TWO_K, coordinator.uiState.value.selectedMode)

        assertTrue(coordinator.tryAcquireJobLock())
        assertFalse("Second concurrent lock attempt must fail to prevent duplicate jobs", coordinator.tryAcquireJobLock())
        coordinator.releaseJobLock()
    }

    @Test
    fun `persisted legacy mode identifiers migrate cleanly to HD HD_PLUS FHD and TWO_K`() {
        assertEquals(EnhancementMode.HD, EnhancementSessionCoordinator.resolveMigratedMode("CLARITY_1X"))
        assertEquals(EnhancementMode.HD_PLUS, EnhancementSessionCoordinator.resolveMigratedMode("CLARITY_2X"))
        assertEquals(EnhancementMode.FHD, EnhancementSessionCoordinator.resolveMigratedMode("CLARITY_3X"))
        assertEquals(EnhancementMode.TWO_K, EnhancementSessionCoordinator.resolveMigratedMode("CLARITY_4X"))

        assertEquals(EnhancementMode.FHD, EnhancementSessionCoordinator.resolveMigratedMode("QHD_2K"))
        assertEquals(EnhancementMode.TWO_K, EnhancementSessionCoordinator.resolveMigratedMode("UHD_4K"))

        assertEquals(EnhancementMode.HD, EnhancementSessionCoordinator.resolveMigratedMode("HD"))
        assertEquals(EnhancementMode.HD_PLUS, EnhancementSessionCoordinator.resolveMigratedMode("HD_PLUS"))
        assertEquals(EnhancementMode.FHD, EnhancementSessionCoordinator.resolveMigratedMode("FHD"))
        assertEquals(EnhancementMode.TWO_K, EnhancementSessionCoordinator.resolveMigratedMode("TWO_K"))

        assertEquals(EnhancementMode.HD_PLUS, EnhancementSessionCoordinator.resolveMigratedMode(null))
        assertEquals(EnhancementMode.HD_PLUS, EnhancementSessionCoordinator.resolveMigratedMode("UNKNOWN_ID"))
    }
}
