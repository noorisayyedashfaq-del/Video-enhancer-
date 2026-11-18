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

        // Verify all four enhancement mode presets load with exact tile config & preserve original resolution
        val profiles = EnhancementProfileRepository(context).loadProfiles()
        assertEquals(4, profiles.size)

        val hd = profiles.getValue(EnhancementMode.HD)
        assertTrue(hd.preserveOriginalResolution)
        assertFalse(hd.useTiledInference)
        assertEquals(0, hd.aiTileSize)
        assertEquals(0, hd.targetTilesPerFrame)

        val hdPlus = profiles.getValue(EnhancementMode.HD_PLUS)
        assertTrue(hdPlus.preserveOriginalResolution)
        assertTrue(hdPlus.useTiledInference)
        assertEquals(512, hdPlus.aiTileSize)
        assertEquals(6, hdPlus.targetTilesPerFrame)

        val qhd2k = profiles.getValue(EnhancementMode.QHD_2K)
        assertTrue(qhd2k.preserveOriginalResolution)
        assertTrue(qhd2k.useTiledInference)
        assertEquals(512, qhd2k.aiTileSize)
        assertEquals(12, qhd2k.targetTilesPerFrame)

        val uhd4k = profiles.getValue(EnhancementMode.UHD_4K)
        assertTrue(uhd4k.preserveOriginalResolution)
        assertTrue(uhd4k.useTiledInference)
        assertEquals(512, uhd4k.aiTileSize)
        assertEquals(25, uhd4k.targetTilesPerFrame)
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
