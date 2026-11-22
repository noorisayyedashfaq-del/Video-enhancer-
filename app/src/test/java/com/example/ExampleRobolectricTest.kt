package com.example

import android.graphics.BitmapFactory
import androidx.core.content.ContextCompat

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import androidx.test.core.app.ApplicationProvider
import com.example.ads.AdGateDecision
import com.example.ads.AdGatePolicy
import com.example.ads.AdManager
import com.example.engine.EnhancementProfileRepository
import com.example.model.EnhancementMode
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [36])
class ExampleRobolectricTest {

    @Test
    fun `app name and bundled offline AI assets and preset tile profiles are present`() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        assertEquals("ClearKit", context.getString(R.string.app_name))
        assertTrue(ContextCompat.getDrawable(context, R.drawable.splash_background) != null)
        assertTrue(ContextCompat.getDrawable(context, R.drawable.ic_launcher_foreground) != null)
        val logoRes = context.resources.openRawResource(R.drawable.img_clearkit_logo).use {
            BitmapFactory.decodeStream(it)
        }
        assertTrue("ClearKit logo bitmap must decode cleanly", logoRes != null && logoRes.width > 0)

        val requiredAssets = listOf(
            "ai_models/realesrgan/realesr_general_x4v3.onnx",
            "ai_models/denoise/dncnn.onnx",
            "ai_models/denoise/dncnn.data",
            "ai_models/face_enhancement/face_detection_yunet_2023mar.onnx",
            "processing/enhancement_profiles.json",
            "configuration/device_profiles.json"
        )
        for (asset in requiredAssets) {
            val bytes = context.assets.open(asset).use { it.readBytes() }
            assertTrue("Asset $asset should be non-empty", bytes.isNotEmpty())
        }

        // Verify all four Clarity enhancement presets load with exact colour, AI, and tile config
        val profiles = EnhancementProfileRepository(context).loadProfiles()
        assertEquals(4, profiles.size)

        val c1 = profiles.getValue(EnhancementMode.CLARITY_1X)
        assertTrue(c1.preserveOriginalResolution)
        assertEquals(1.0f, c1.targetScaleFactor, 0.01f)
        assertTrue(c1.useTiledInference)
        assertEquals(256, c1.aiTileSize)
        assertEquals(42f, c1.aiStrengthPercent, 0.01f)
        assertEquals(30f, c1.dncnnStrengthPercent, 0.01f)
        assertEquals(40f, c1.faceEnhancePercent, 0.01f)

        val c2 = profiles.getValue(EnhancementMode.CLARITY_2X)
        assertFalse(c2.preserveOriginalResolution)
        assertEquals(2.0f, c2.targetScaleFactor, 0.01f)
        assertEquals(2560, c2.maxOutputLongEdge)
        assertEquals(256, c2.aiTileSize)
        assertEquals(66f, c2.aiStrengthPercent, 0.01f)
        assertEquals(52f, c2.dncnnStrengthPercent, 0.01f)
        assertEquals(62f, c2.faceEnhancePercent, 0.01f)

        val c3 = profiles.getValue(EnhancementMode.CLARITY_3X)
        assertFalse(c3.preserveOriginalResolution)
        assertEquals(3.0f, c3.targetScaleFactor, 0.01f)
        assertEquals(3200, c3.maxOutputLongEdge)
        assertEquals(320, c3.aiTileSize)
        assertEquals(82f, c3.aiStrengthPercent, 0.01f)
        assertEquals(72f, c3.dncnnStrengthPercent, 0.01f)
        assertEquals(78f, c3.faceEnhancePercent, 0.01f)

        val c4 = profiles.getValue(EnhancementMode.CLARITY_4X)
        assertFalse(c4.preserveOriginalResolution)
        assertEquals(4.0f, c4.targetScaleFactor, 0.01f)
        assertEquals(3840, c4.maxOutputLongEdge)
        assertEquals(384, c4.aiTileSize)
        assertEquals(94f, c4.aiStrengthPercent, 0.01f)
        assertEquals(88f, c4.dncnnStrengthPercent, 0.01f)
        assertEquals(92f, c4.faceEnhancePercent, 0.01f)
        assertEquals(0f, c4.temperatureShift, 0.01f)
        assertEquals(0f, c4.hueShift, 0.01f)
    }

    @Test
    fun `admob manifest metadata permissions test unit id and ad gate rules are properly configured`() {
        val context = ApplicationProvider.getApplicationContext<Context>()

        assertEquals("ca-app-pub-3940256099942544/1033173712", AdManager.TEST_INTERSTITIAL_AD_UNIT_ID)

        // Verify INTERNET and ACCESS_NETWORK_STATE permissions are declared
        val packageInfo = context.packageManager.getPackageInfo(
            context.packageName,
            PackageManager.GET_PERMISSIONS
        )
        val permissions = packageInfo.requestedPermissions?.toList().orEmpty()
        assertTrue(permissions.contains(Manifest.permission.INTERNET))
        assertTrue(permissions.contains(Manifest.permission.ACCESS_NETWORK_STATE))

        // Verify APPLICATION_ID meta-data in AndroidManifest
        val appInfo = context.packageManager.getApplicationInfo(
            context.packageName,
            PackageManager.GET_META_DATA
        )
        assertEquals(
            "ca-app-pub-3940256099942544~3347511713",
            appInfo.metaData?.getString("com.google.android.gms.ads.APPLICATION_ID")
        )

        // Verify exact user-facing Ad flow & dialog messages
        assertEquals(
            "An Ad will display while your task being processed",
            context.getString(R.string.ad_notice_line_1)
        )
        assertEquals(
            "First watch the Ad, then processing will start",
            context.getString(R.string.ad_notice_line_2)
        )
        assertEquals(
            "Ad is loading, please tap again",
            context.getString(R.string.ad_loading_toast)
        )
        assertEquals(
            "Internet Required",
            context.getString(R.string.internet_required_title)
        )
        assertEquals(
            "To enhance video you need to watch a small Ad first. Please turn on Internet.",
            context.getString(R.string.internet_required_message)
        )

        // Verify gate policy blocks execution when offline or ad not loaded
        assertEquals(
            AdGateDecision.SHOW_INTERNET_DIALOG,
            AdGatePolicy.evaluate(isInternetAvailable = false, isAdLoaded = false)
        )
        assertEquals(
            AdGateDecision.SHOW_INTERNET_DIALOG,
            AdGatePolicy.evaluate(isInternetAvailable = false, isAdLoaded = true)
        )
        assertEquals(
            AdGateDecision.SHOW_AD_LOADING_TOAST,
            AdGatePolicy.evaluate(isInternetAvailable = true, isAdLoaded = false)
        )
        assertEquals(
            AdGateDecision.SHOW_AD_THEN_EXECUTE,
            AdGatePolicy.evaluate(isInternetAvailable = true, isAdLoaded = true)
        )
    }
}
