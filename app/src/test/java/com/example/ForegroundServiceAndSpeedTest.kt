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
import com.example.engine.DeviceCapabilityDetector
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
    fun `AiSpeedPolicy triggers automatic lightweight fallback only when AI latency exceeds speed threshold`() {
        // Fast GPU/NNAPI 2K (12 tiles x 30ms = 360ms/frame) -> stays on AI model
        assertFalse(
            AiSpeedPolicy.shouldEngageLightweightFallback(
                measuredTileMs = 30L,
                tilesPerFrame = 12,
                estimatedFrameCount = 150
            )
        )

        // Slow CPU 4K (25 tiles x 120ms = 3000ms/frame) -> triggers automatic lightweight speed fallback
        assertTrue(
            AiSpeedPolicy.shouldEngageLightweightFallback(
                measuredTileMs = 120L,
                tilesPerFrame = 25,
                estimatedFrameCount = 150
            )
        )

        // Heavy 2K tile > 220ms on multi-second clip -> triggers fast fallback
        assertTrue(
            AiSpeedPolicy.shouldEngageLightweightFallback(
                measuredTileMs = 240L,
                tilesPerFrame = 12,
                estimatedFrameCount = 120
            )
        )
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
        val notification = service.buildProgressNotification(progress, EnhancementMode.QHD_2K)

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

        coordinator.selectMode(EnhancementMode.UHD_4K)
        assertEquals(EnhancementMode.UHD_4K, coordinator.uiState.value.selectedMode)

        assertTrue(coordinator.tryAcquireJobLock())
        assertFalse("Second concurrent lock attempt must fail to prevent duplicate jobs", coordinator.tryAcquireJobLock())
        coordinator.releaseJobLock()
    }
}
