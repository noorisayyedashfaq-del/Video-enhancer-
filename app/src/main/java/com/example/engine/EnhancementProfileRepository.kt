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
        EnhancementMode.CLARITY_1X to EnhancementProfile(
            id = "CLARITY_1X",
            displayName = "Clarity 1x",
            description = "Natural AI: neural detail recovery & grain cleanup at original resolution",
            targetLongEdge = 0,
            targetShortEdge = 0,
            preserveOriginalResolution = true,
            targetScaleFactor = 1.0f,
            maxOutputLongEdge = 0,
            useAiSuperResolution = true,
            useMultiStageUpscaling = false,
            useTiledInference = true,
            aiTileSize = 256,
            targetTilesPerFrame = 6,
            temporalConsistency = true,
            brightnessPercent = 0.5f,
            contrastPercent = 1.5f,
            sharpnessPercent = 6f,
            denoisePercent = 28f,
            saturationPercent = 2f,
            brilliancePercent = 1.5f,
            clarityPercent = 3f,
            highlightsPercent = -2f,
            shadowsPercent = 1.5f,
            whitesPercent = 1f,
            blacksPercent = -1f,
            temperatureShift = 0f,
            hueShift = 0f,
            aiStrengthPercent = 42f,
            detailRecoveryPercent = 42f,
            noiseReductionPercent = 30f,
            deblurPercent = 35f,
            aiSharpeningPercent = 18f,
            artifactReductionPercent = 30f,
            dncnnStrengthPercent = 30f,
            faceEnhancePercent = 40f,
            gamma = 1.0f,
            whiteBalance = "Neutral",
            colorCorrection = "Natural",
            deblocking = "Low",
            dehazingPercent = 1f,
            edgeEnhancementPercent = 3f,
            flickerReduction = "Automatic",
            preferredCodec = "H264",
            targetBpp = 0.18f
        ),
        EnhancementMode.CLARITY_2X to EnhancementProfile(
            id = "CLARITY_2X",
            displayName = "Clarity 2x",
            description = "HD AI: 2x Real-ESRGAN super-resolution, DnCNN denoising & YuNet face detail",
            targetLongEdge = 2560,
            targetShortEdge = 1440,
            preserveOriginalResolution = false,
            targetScaleFactor = 2.0f,
            maxOutputLongEdge = 2560,
            useAiSuperResolution = true,
            useMultiStageUpscaling = false,
            useTiledInference = true,
            aiTileSize = 256,
            targetTilesPerFrame = 9,
            temporalConsistency = true,
            brightnessPercent = 1.0f,
            contrastPercent = 2.5f,
            sharpnessPercent = 10f,
            denoisePercent = 48f,
            saturationPercent = 3f,
            brilliancePercent = 2.5f,
            clarityPercent = 5f,
            highlightsPercent = -3f,
            shadowsPercent = 2f,
            whitesPercent = 1.5f,
            blacksPercent = -1.5f,
            temperatureShift = 0f,
            hueShift = 0f,
            aiStrengthPercent = 66f,
            detailRecoveryPercent = 66f,
            noiseReductionPercent = 50f,
            deblurPercent = 55f,
            aiSharpeningPercent = 28f,
            artifactReductionPercent = 50f,
            dncnnStrengthPercent = 52f,
            faceEnhancePercent = 62f,
            gamma = 1.0f,
            whiteBalance = "Neutral",
            colorCorrection = "Balanced",
            deblocking = "Medium",
            dehazingPercent = 2f,
            edgeEnhancementPercent = 5f,
            flickerReduction = "Automatic",
            preferredCodec = "H264",
            targetBpp = 0.20f
        ),
        EnhancementMode.CLARITY_3X to EnhancementProfile(
            id = "CLARITY_3X",
            displayName = "Clarity 3x",
            description = "Ultra AI: 3x neural reconstruction, advanced deblurring & facial refinement",
            targetLongEdge = 3200,
            targetShortEdge = 1800,
            preserveOriginalResolution = false,
            targetScaleFactor = 3.0f,
            maxOutputLongEdge = 3200,
            useAiSuperResolution = true,
            useMultiStageUpscaling = true,
            useTiledInference = true,
            aiTileSize = 320,
            targetTilesPerFrame = 12,
            temporalConsistency = true,
            brightnessPercent = 1.2f,
            contrastPercent = 3.5f,
            sharpnessPercent = 14f,
            denoisePercent = 68f,
            saturationPercent = 4f,
            brilliancePercent = 3.5f,
            clarityPercent = 7f,
            highlightsPercent = -4f,
            shadowsPercent = 2.5f,
            whitesPercent = 2f,
            blacksPercent = -2f,
            temperatureShift = 0f,
            hueShift = 0f,
            aiStrengthPercent = 82f,
            detailRecoveryPercent = 82f,
            noiseReductionPercent = 70f,
            deblurPercent = 72f,
            aiSharpeningPercent = 36f,
            artifactReductionPercent = 68f,
            dncnnStrengthPercent = 72f,
            faceEnhancePercent = 78f,
            gamma = 1.0f,
            whiteBalance = "Neutral",
            colorCorrection = "Balanced",
            deblocking = "Medium",
            dehazingPercent = 3f,
            edgeEnhancementPercent = 7f,
            flickerReduction = "Automatic",
            preferredCodec = "H265",
            targetBpp = 0.22f
        ),
        EnhancementMode.CLARITY_4X to EnhancementProfile(
            id = "CLARITY_4X",
            displayName = "Clarity 4x",
            description = "Maximum AI: full 4x super-resolution up to 4K with deep DnCNN & face AI restoration",
            targetLongEdge = 3840,
            targetShortEdge = 2160,
            preserveOriginalResolution = false,
            targetScaleFactor = 4.0f,
            maxOutputLongEdge = 3840,
            useAiSuperResolution = true,
            useMultiStageUpscaling = true,
            useTiledInference = true,
            aiTileSize = 384,
            targetTilesPerFrame = 16,
            temporalConsistency = true,
            brightnessPercent = 1.5f,
            contrastPercent = 4.0f,
            sharpnessPercent = 18f,
            denoisePercent = 84f,
            saturationPercent = 5f,
            brilliancePercent = 4.5f,
            clarityPercent = 9f,
            highlightsPercent = -5f,
            shadowsPercent = 3f,
            whitesPercent = 2.5f,
            blacksPercent = -2.5f,
            temperatureShift = 0f,
            hueShift = 0f,
            aiStrengthPercent = 94f,
            detailRecoveryPercent = 94f,
            noiseReductionPercent = 85f,
            deblurPercent = 88f,
            aiSharpeningPercent = 44f,
            artifactReductionPercent = 82f,
            dncnnStrengthPercent = 88f,
            faceEnhancePercent = 92f,
            gamma = 1.0f,
            whiteBalance = "Neutral",
            colorCorrection = "Balanced",
            deblocking = "High",
            dehazingPercent = 4f,
            edgeEnhancementPercent = 9f,
            flickerReduction = "Automatic",
            preferredCodec = "H265",
            targetBpp = 0.24f
        )
    )
}
