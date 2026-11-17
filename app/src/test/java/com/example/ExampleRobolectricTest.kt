package com.example

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import com.example.engine.EnhancementProfileRepository
import com.example.model.EnhancementMode
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [36])
class ExampleRobolectricTest {

    @Test
    fun `app name and bundled offline AI assets are present`() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        assertEquals("LuminaSR", context.getString(R.string.app_name))

        // Verify all AI model weights & config assets are bundled non-empty
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

        // Verify all four enhancement mode profiles load cleanly from JSON
        val profiles = EnhancementProfileRepository(context).loadProfiles()
        assertEquals(4, profiles.size)
        assertTrue(profiles[EnhancementMode.HD]?.preserveOriginalResolution == true)
        assertTrue(profiles[EnhancementMode.HD_PLUS]?.targetShortEdge == 720)
        assertTrue(profiles[EnhancementMode.QHD_2K]?.targetShortEdge == 1440)
        assertTrue(profiles[EnhancementMode.UHD_4K]?.targetShortEdge == 2160)
    }
}
