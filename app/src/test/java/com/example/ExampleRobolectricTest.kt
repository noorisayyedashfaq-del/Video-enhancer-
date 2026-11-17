package com.example

import android.content.Context
import androidx.test.core.app.ApplicationProvider
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
        assertEquals("LuminaSR", context.getString(R.string.app_name))

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
        assertEquals(384, qhd2k.aiTileSize)
        assertEquals(12, qhd2k.targetTilesPerFrame)

        val uhd4k = profiles.getValue(EnhancementMode.UHD_4K)
        assertTrue(uhd4k.preserveOriginalResolution)
        assertTrue(uhd4k.useTiledInference)
        assertEquals(256, uhd4k.aiTileSize)
        assertEquals(25, uhd4k.targetTilesPerFrame)
    }
}
