package com.example

import android.content.Context
import android.graphics.BitmapFactory
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performScrollTo
import androidx.core.content.ContextCompat
import androidx.test.core.app.ApplicationProvider
import com.example.ads.AdManager
import com.example.engine.DeviceCapabilityDetector
import com.example.model.EnhancementMode
import com.example.service.EnhancementSessionCoordinator
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.Shadows
import org.robolectric.annotation.Config
import android.os.Looper
import java.util.concurrent.TimeUnit

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [36])
class AppStartupStabilityTest {

    @get:Rule
    val composeRule = createAndroidComposeRule<MainActivity>()

    @Test
    fun `mainActivity launches Compose root and splash and survives full lifecycle`() {
        composeRule.waitForIdle()

        // Main UI interactive controls and splash compose cleanly on cold start
        composeRule.onNodeWithTag("select_video_button").assertExists()
        composeRule.onNodeWithTag("hardware_diagnostics_button").assertExists()
        composeRule.onNodeWithTag("mode_button_hd").assertExists()
        composeRule.onNodeWithTag("mode_button_hd_plus").assertExists()
        composeRule.onNodeWithTag("mode_button_fhd").assertExists()
        composeRule.onNodeWithTag("mode_button_two_k").assertExists()
        composeRule.onNodeWithTag("start_enhancement_button").assertExists()

        // Advance main looper past splash auto-dismiss (1000ms) and verify home studio controls are visible
        composeRule.mainClock.advanceTimeBy(1500L)
        composeRule.waitForIdle()
        composeRule.onNodeWithTag("select_video_button").assertIsDisplayed()
        composeRule.onNodeWithTag("start_enhancement_button")
            .performScrollTo()
            .assertIsDisplayed()
    }

    @Test
    fun `secondary MainActivity create-resume-destroy cycle runs without exception`() {
        val controller = Robolectric.buildActivity(MainActivity::class.java).setup()
        val activity = controller.get()
        Shadows.shadowOf(Looper.getMainLooper()).idleFor(1200L, TimeUnit.MILLISECONDS)
        assertFalse(activity.isFinishing)
        assertFalse(activity.isDestroyed)

        controller.pause().stop().destroy()
        assertTrue(activity.isDestroyed)
    }

    @Test
    fun `device capability detector completes cleanly without OEM NPEs`() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val caps = DeviceCapabilityDetector(context).detect()

        assertTrue(caps.tierName.isNotBlank())
        assertTrue(caps.totalRamMb >= 1024L)
        assertTrue(caps.cpuCores >= 2)
        assertTrue(caps.adaptiveTileSize >= 96)
        assertNotNull(caps.hardwareProbeSnapshot)
    }

    @Test
    fun `session coordinator and ad manager handle cold start and idle calls safely`() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val coordinator = EnhancementSessionCoordinator.getInstance(context)

        assertTrue(coordinator.uiState.value.selectedMode in EnhancementMode.entries)
        coordinator.clearError()
        coordinator.onTrimMemory()

        // Calling AdManager initialization & loadAdIfNeeded before SDK readiness must never crash
        assertFalse(AdManager.isInternetAvailable(context) && !AdManager.isSdkReady() && AdManager.isAdReady())
        AdManager.initialize(context)
        AdManager.loadAdIfNeeded(context)
    }

    @Test
    fun `launcher and splash drawables load safely within Canvas memory budget`() {
        val context = ApplicationProvider.getApplicationContext<Context>()

        assertNotNull(ContextCompat.getDrawable(context, R.drawable.splash_background))
        assertNotNull(ContextCompat.getDrawable(context, R.drawable.ic_launcher_foreground))

        val rawHeader = context.resources.openRawResource(R.drawable.video_enhancer_logo).use { input ->
            ByteArray(12).also { input.read(it) }
        }
        assertEquals("RIFF", String(rawHeader, 0, 4, Charsets.US_ASCII))
        assertEquals("WEBP", String(rawHeader, 8, 4, Charsets.US_ASCII))

        val logoBitmap = BitmapFactory.decodeResource(context.resources, R.drawable.video_enhancer_logo)
        assertNotNull("Logo bitmap should decode", logoBitmap)
        assertTrue("Logo allocation must stay safely below Canvas limit", logoBitmap.allocationByteCount < 16 * 1024 * 1024)
        assertTrue("Logo bitmap width must be positive", logoBitmap.width > 0)

        val bannerBitmap = BitmapFactory.decodeResource(context.resources, R.drawable.img_studio_banner)
        assertNotNull("Banner bitmap should decode", bannerBitmap)
        assertTrue("Banner allocation must stay safely below Canvas limit", bannerBitmap.allocationByteCount < 16 * 1024 * 1024)
        assertEquals(1376, bannerBitmap.width)
    }
}
