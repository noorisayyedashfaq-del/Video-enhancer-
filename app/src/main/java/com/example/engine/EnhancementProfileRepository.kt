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
        } catch (_: Exception) {
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
            description = "Natural Enhancement: 512px tile size, 4 tiles/frame at original resolution",
            targetLongEdge = 0,
            targetShortEdge = 0,
            preserveOriginalResolution = true,
            useAiSuperResolution = true,
            useMultiStageUpscaling = false,
            useTiledInference = true,
            aiTileSize = 512,
            targetTilesPerFrame = 4,
            temporalConsistency = true,
            brightnessPercent = 1f,
            contrastPercent = 3f,
            sharpnessPercent = 12f,
            denoisePercent = 25f,
            saturationPercent = 4f,
            brilliancePercent = 3f,
            clarityPercent = 5f,
            highlightsPercent = -3f,
            shadowsPercent = 2f,
            whitesPercent = 2f,
            blacksPercent = -2f,
            temperatureShift = 0f,
            hueShift = 0f,
            aiStrengthPercent = 25f,
            detailRecoveryPercent = 25f,
            noiseReductionPercent = 25f,
            deblurPercent = 20f,
            aiSharpeningPercent = 15f,
            artifactReductionPercent = 20f,
            gamma = 1.0f,
            whiteBalance = "Neutral",
            colorCorrection = "Natural",
            deblocking = "Low",
            dehazingPercent = 2f,
            edgeEnhancementPercent = 5f,
            flickerReduction = "Automatic",
            preferredCodec = "H264",
            targetBpp = 0.14f
        ),
        EnhancementMode.CLARITY_2X to EnhancementProfile(
            id = "CLARITY_2X",
            displayName = "Clarity 2x",
            description = "HD Enhancement: 512px tile size, 6 tiles/frame at original resolution",
            targetLongEdge = 0,
            targetShortEdge = 0,
            preserveOriginalResolution = true,
            useAiSuperResolution = true,
            useMultiStageUpscaling = false,
            useTiledInference = true,
            aiTileSize = 512,
            targetTilesPerFrame = 6,
            temporalConsistency = true,
            brightnessPercent = 2f,
            contrastPercent = 5f,
            sharpnessPercent = 22f,
            denoisePercent = 40f,
            saturationPercent = 7f,
            brilliancePercent = 6f,
            clarityPercent = 12f,
            highlightsPercent = -6f,
            shadowsPercent = 3f,
            whitesPercent = 3f,
            blacksPercent = -3f,
            temperatureShift = 0f,
            hueShift = 0f,
            aiStrengthPercent = 45f,
            detailRecoveryPercent = 45f,
            noiseReductionPercent = 40f,
            deblurPercent = 40f,
            aiSharpeningPercent = 30f,
            artifactReductionPercent = 40f,
            gamma = 1.0f,
            whiteBalance = "Neutral",
            colorCorrection = "Balanced",
            deblocking = "Medium",
            dehazingPercent = 4f,
            edgeEnhancementPercent = 10f,
            flickerReduction = "Automatic",
            preferredCodec = "H264",
            targetBpp = 0.16f
        ),
        EnhancementMode.CLARITY_3X to EnhancementProfile(
            id = "CLARITY_3X",
            displayName = "Clarity 3x",
            description = "Ultra Enhancement: 512px tile size, 12 tiles/frame at original resolution",
            targetLongEdge = 0,
            targetShortEdge = 0,
            preserveOriginalResolution = true,
            useAiSuperResolution = true,
            useMultiStageUpscaling = false,
            useTiledInference = true,
            aiTileSize = 512,
            targetTilesPerFrame = 12,
            temporalConsistency = true,
            brightnessPercent = 3f,
            contrastPercent = 8f,
            sharpnessPercent = 35f,
            denoisePercent = 60f,
            saturationPercent = 10f,
            brilliancePercent = 9f,
            clarityPercent = 20f,
            highlightsPercent = -9f,
            shadowsPercent = 5f,
            whitesPercent = 5f,
            blacksPercent = -5f,
            temperatureShift = 0f,
            hueShift = 0f,
            aiStrengthPercent = 70f,
            detailRecoveryPercent = 70f,
            noiseReductionPercent = 60f,
            deblurPercent = 60f,
            aiSharpeningPercent = 45f,
            artifactReductionPercent = 60f,
            gamma = 1.0f,
            whiteBalance = "Neutral",
            colorCorrection = "Balanced",
            deblocking = "Medium",
            dehazingPercent = 6f,
            edgeEnhancementPercent = 14f,
            flickerReduction = "Automatic",
            preferredCodec = "H265",
            targetBpp = 0.17f
        ),
        EnhancementMode.CLARITY_4X to EnhancementProfile(
            id = "CLARITY_4X",
            displayName = "Clarity 4x",
            description = "Maximum Enhancement: 512px tile size, 25 tiles/frame at original resolution",
            targetLongEdge = 0,
            targetShortEdge = 0,
            preserveOriginalResolution = true,
            useAiSuperResolution = true,
            useMultiStageUpscaling = false,
            useTiledInference = true,
            aiTileSize = 512,
            targetTilesPerFrame = 25,
            temporalConsistency = true,
            brightnessPercent = 4f,
            contrastPercent = 10f,
            sharpnessPercent = 45f,
            denoisePercent = 75f,
            saturationPercent = 12f,
            brilliancePercent = 12f,
            clarityPercent = 28f,
            highlightsPercent = -12f,
            shadowsPercent = 6f,
            whitesPercent = 7f,
            blacksPercent = -7f,
            temperatureShift = 0f,
            hueShift = 0f,
            aiStrengthPercent = 90f,
            detailRecoveryPercent = 90f,
            noiseReductionPercent = 75f,
            deblurPercent = 80f,
            aiSharpeningPercent = 60f,
            artifactReductionPercent = 75f,
            gamma = 1.0f,
            whiteBalance = "Neutral",
            colorCorrection = "Balanced",
            deblocking = "High",
            dehazingPercent = 8f,
            edgeEnhancementPercent = 18f,
            flickerReduction = "Automatic",
            preferredCodec = "H265",
            targetBpp = 0.18f
        )
    )
}
