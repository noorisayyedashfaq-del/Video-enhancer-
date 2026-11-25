package com.example.engine

import android.content.Context
import com.example.model.EnhancementMode
import com.example.model.EnhancementProfile
import com.example.model.EnhancementProfilesRoot
import kotlinx.serialization.json.Json

class EnhancementProfileRepository(private val context: Context) {

    private val json = Json { ignoreUnknownKeys = true }

    fun loadProfiles(): Map<EnhancementMode, EnhancementProfile> {
        val loaded = try {
            val raw = context.assets.open("processing/enhancement_profiles.json").bufferedReader().use { it.readText() }
            val root = json.decodeFromString<EnhancementProfilesRoot>(raw)
            buildMap {
                EnhancementMode.entries.forEach { mode ->
                    root.modes[mode.id]?.let { put(mode, it) }
                }
            }
        } catch (_: Throwable) {
            emptyMap()
        }

        return if (loaded.size == EnhancementMode.entries.size) {
            loaded
        } else {
            defaultProfiles()
        }
    }

    private fun defaultProfiles(): Map<EnhancementMode, EnhancementProfile> = mapOf(
        EnhancementMode.HD to EnhancementProfile(
            id = "HD",
            displayName = "HD",
            description = "Natural AI: vibrant natural color, neural detail, smooth motion at HD (1024x576 / 576x1024)",
            targetLongEdge = 1024,
            targetShortEdge = 576,
            preserveOriginalResolution = false,
            targetScaleFactor = 1.0f,
            maxOutputLongEdge = 1024,
            useAiSuperResolution = true,
            useMultiStageUpscaling = false,
            useTiledInference = true,
            aiTileSize = 192,
            targetTilesPerFrame = 1,
            temporalConsistency = true,
            brightnessPercent = 1.5f,
            contrastPercent = 5.0f,
            sharpnessPercent = 12f,
            denoisePercent = 32f,
            saturationPercent = 6f,
            brilliancePercent = 6f,
            clarityPercent = 8f,
            highlightsPercent = -4f,
            shadowsPercent = 3f,
            whitesPercent = 3f,
            blacksPercent = -3f,
            temperatureShift = 0f,
            hueShift = 0f,
            aiStrengthPercent = 42f,
            detailRecoveryPercent = 46f,
            noiseReductionPercent = 32f,
            deblurPercent = 38f,
            aiSharpeningPercent = 22f,
            artifactReductionPercent = 34f,
            dncnnStrengthPercent = 30f,
            faceEnhancePercent = 40f,
            gamma = 1.0f,
            whiteBalance = "Neutral",
            colorCorrection = "Natural",
            deblocking = "Low",
            dehazingPercent = 2f,
            edgeEnhancementPercent = 5f,
            flickerReduction = "Automatic",
            preferredCodec = "H264",
            targetBpp = 0.18f
        ),
        EnhancementMode.HD_PLUS to EnhancementProfile(
            id = "HD_PLUS",
            displayName = "HD+",
            description = "HD+ AI: Real-ESRGAN detail reconstruction, vivid color & flow smoothness at HD+ (1152x648 / 648x1152)",
            targetLongEdge = 1152,
            targetShortEdge = 648,
            preserveOriginalResolution = false,
            targetScaleFactor = 2.0f,
            maxOutputLongEdge = 1152,
            useAiSuperResolution = true,
            useMultiStageUpscaling = false,
            useTiledInference = true,
            aiTileSize = 192,
            targetTilesPerFrame = 2,
            temporalConsistency = true,
            brightnessPercent = 2.5f,
            contrastPercent = 8.0f,
            sharpnessPercent = 18f,
            denoisePercent = 48f,
            saturationPercent = 10f,
            brilliancePercent = 10f,
            clarityPercent = 14f,
            highlightsPercent = -7f,
            shadowsPercent = 5f,
            whitesPercent = 4f,
            blacksPercent = -4f,
            temperatureShift = 0f,
            hueShift = 0f,
            aiStrengthPercent = 66f,
            detailRecoveryPercent = 68f,
            noiseReductionPercent = 50f,
            deblurPercent = 56f,
            aiSharpeningPercent = 30f,
            artifactReductionPercent = 52f,
            dncnnStrengthPercent = 52f,
            faceEnhancePercent = 62f,
            gamma = 1.0f,
            whiteBalance = "Neutral",
            colorCorrection = "Balanced",
            deblocking = "Medium",
            dehazingPercent = 4f,
            edgeEnhancementPercent = 8f,
            flickerReduction = "Automatic",
            preferredCodec = "H264",
            targetBpp = 0.20f
        ),
        EnhancementMode.FHD to EnhancementProfile(
            id = "FHD",
            displayName = "FHD",
            description = "FHD AI: neural texture reconstruction, balanced color & smooth motion at FHD (1440x810 / 810x1440)",
            targetLongEdge = 1440,
            targetShortEdge = 810,
            preserveOriginalResolution = false,
            targetScaleFactor = 3.0f,
            maxOutputLongEdge = 1440,
            useAiSuperResolution = true,
            useMultiStageUpscaling = true,
            useTiledInference = true,
            aiTileSize = 224,
            targetTilesPerFrame = 3,
            temporalConsistency = true,
            brightnessPercent = 3.5f,
            contrastPercent = 11.0f,
            sharpnessPercent = 26f,
            denoisePercent = 68f,
            saturationPercent = 13f,
            brilliancePercent = 14f,
            clarityPercent = 20f,
            highlightsPercent = -10f,
            shadowsPercent = 7f,
            whitesPercent = 6f,
            blacksPercent = -6f,
            temperatureShift = 0f,
            hueShift = 0f,
            aiStrengthPercent = 82f,
            detailRecoveryPercent = 84f,
            noiseReductionPercent = 70f,
            deblurPercent = 74f,
            aiSharpeningPercent = 38f,
            artifactReductionPercent = 68f,
            dncnnStrengthPercent = 72f,
            faceEnhancePercent = 78f,
            gamma = 1.0f,
            whiteBalance = "Neutral",
            colorCorrection = "Balanced",
            deblocking = "Medium",
            dehazingPercent = 6f,
            edgeEnhancementPercent = 12f,
            flickerReduction = "Automatic",
            preferredCodec = "H264",
            targetBpp = 0.22f
        ),
        EnhancementMode.TWO_K to EnhancementProfile(
            id = "TWO_K",
            displayName = "2K",
            description = "2K AI: super-resolution at 2K (1920x1080 / 1080x1920) with full color, detail & motion enhancement",
            targetLongEdge = 1920,
            targetShortEdge = 1080,
            preserveOriginalResolution = false,
            targetScaleFactor = 4.0f,
            maxOutputLongEdge = 1920,
            useAiSuperResolution = true,
            useMultiStageUpscaling = true,
            useTiledInference = true,
            aiTileSize = 256,
            targetTilesPerFrame = 4,
            temporalConsistency = true,
            brightnessPercent = 4.5f,
            contrastPercent = 14.0f,
            sharpnessPercent = 34f,
            denoisePercent = 84f,
            saturationPercent = 16f,
            brilliancePercent = 18f,
            clarityPercent = 26f,
            highlightsPercent = -13f,
            shadowsPercent = 9f,
            whitesPercent = 8f,
            blacksPercent = -8f,
            temperatureShift = 0f,
            hueShift = 0f,
            aiStrengthPercent = 94f,
            detailRecoveryPercent = 94f,
            noiseReductionPercent = 85f,
            deblurPercent = 88f,
            aiSharpeningPercent = 46f,
            artifactReductionPercent = 82f,
            dncnnStrengthPercent = 88f,
            faceEnhancePercent = 92f,
            gamma = 1.0f,
            whiteBalance = "Neutral",
            colorCorrection = "Balanced",
            deblocking = "High",
            dehazingPercent = 8f,
            edgeEnhancementPercent = 16f,
            flickerReduction = "Automatic",
            preferredCodec = "H264",
            targetBpp = 0.24f
        )
    )
}
