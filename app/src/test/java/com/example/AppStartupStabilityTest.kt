package com.example

import android.content.Context
import android.graphics.BitmapFactory
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.getUnclippedBoundsInRoot
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onAllNodesWithText
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
        composeRule.onNodeWithTag("settings_button").assertExists()
        composeRule.onNodeWithTag("hardware_diagnostics_button").assertDoesNotExist()
        composeRule.onNodeWithTag("mode_button_hd").assertExists()
        composeRule.onNodeWithTag("mode_button_hd_plus").assertExists()
        composeRule.onNodeWithTag("mode_button_fhd").assertExists()
        composeRule.onNodeWithTag("mode_button_two_k").assertExists()
        composeRule.onNodeWithTag("start_enhancement_button").assertExists()

        // Advance main looper past splash auto-dismiss (1000ms) and verify home studio controls are visible
        composeRule.mainClock.advanceTimeBy(1500L)
        composeRule.waitForIdle()
        composeRule.onNodeWithTag("settings_button").assertIsDisplayed()
        composeRule.onNodeWithTag("select_video_button").assertIsDisplayed()

        // Verify 4 equal quality buttons sit in a single horizontal row (HD -> HD+ -> FHD -> 2K) with single labels
        val hdBounds = composeRule.onNodeWithTag("mode_button_hd").performScrollTo().getUnclippedBoundsInRoot()
        val hdPlusBounds = composeRule.onNodeWithTag("mode_button_hd_plus").getUnclippedBoundsInRoot()
        val fhdBounds = composeRule.onNodeWithTag("mode_button_fhd").getUnclippedBoundsInRoot()
        val twoKBounds = composeRule.onNodeWithTag("mode_button_two_k").getUnclippedBoundsInRoot()

        assertEquals(hdBounds.top.value, hdPlusBounds.top.value, 1f)
        assertEquals(hdPlusBounds.top.value, fhdBounds.top.value, 1f)
        assertEquals(fhdBounds.top.value, twoKBounds.top.value, 1f)
        assertTrue(hdBounds.left < hdPlusBounds.left && hdPlusBounds.left < fhdBounds.left && fhdBounds.left < twoKBounds.left)
        assertEquals(hdBounds.right.value - hdBounds.left.value, hdPlusBounds.right.value - hdPlusBounds.left.value, 1f)
        assertEquals(fhdBounds.right.value - fhdBounds.left.value, twoKBounds.right.value - twoKBounds.left.value, 1f)

        composeRule.onAllNodesWithText("HD").fetchSemanticsNodes().let { assertEquals(1, it.size) }
        composeRule.onAllNodesWithText("HD+").fetchSemanticsNodes().let { assertEquals(1, it.size) }
        composeRule.onAllNodesWithText("FHD").fetchSemanticsNodes().let { assertEquals(1, it.size) }
        composeRule.onAllNodesWithText("2K").fetchSemanticsNodes().let { assertEquals(1, it.size) }

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

        val comparisonHeader = context.resources.openRawResource(R.drawable.img_video_comparison).use { input ->
            ByteArray(12).also { input.read(it) }
        }
        assertEquals("RIFF", String(comparisonHeader, 0, 4, Charsets.US_ASCII))
        assertEquals("WEBP", String(comparisonHeader, 8, 4, Charsets.US_ASCII))

        val comparisonBitmap = BitmapFactory.decodeResource(context.resources, R.drawable.img_video_comparison)
        assertNotNull("Comparison preview bitmap should decode", comparisonBitmap)
        assertTrue("Comparison allocation must stay safely below Canvas limit", comparisonBitmap.allocationByteCount < 16 * 1024 * 1024)
        assertTrue("Comparison preview bitmap width must be positive", comparisonBitmap.width > 0)
    }
}
