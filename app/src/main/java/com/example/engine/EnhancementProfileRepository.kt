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
        EnhancementMode.HD to EnhancementProfile(
            id = "HD",
            displayName = "HD",
            description = "Improve quality without changing original resolution",
            targetLongEdge = 0,
            targetShortEdge = 0,
            preserveOriginalResolution = true,
            useAiSuperResolution = false,
            useMultiStageUpscaling = false,
            useTiledInference = false,
            temporalConsistency = true,
            brightnessPercent = 3f,
            contrastPercent = 8f,
            sharpnessPercent = 15f,
            denoisePercent = 10f,
            saturationPercent = 5f,
            gamma = 1.03f,
            whiteBalance = "Automatic",
            colorCorrection = "Automatic",
            deblocking = "Low",
            dehazingPercent = 5f,
            edgeEnhancementPercent = 10f,
            flickerReduction = "Automatic",
            preferredCodec = "H264",
            targetBpp = 0.16f
        ),
        EnhancementMode.HD_PLUS to EnhancementProfile(
            id = "HD_PLUS",
            displayName = "HD+",
            description = "Enhance quality and upscale to 720p",
            targetLongEdge = 1280,
            targetShortEdge = 720,
            preserveOriginalResolution = false,
            useAiSuperResolution = true,
            useMultiStageUpscaling = false,
            useTiledInference = true,
            temporalConsistency = true,
            brightnessPercent = 3f,
            contrastPercent = 10f,
            sharpnessPercent = 20f,
            denoisePercent = 15f,
            saturationPercent = 6f,
            gamma = 1.04f,
            whiteBalance = "Automatic",
            colorCorrection = "Automatic",
            deblocking = "Medium",
            dehazingPercent = 8f,
            edgeEnhancementPercent = 15f,
            flickerReduction = "Automatic",
            preferredCodec = "H264",
            targetBpp = 0.17f
        ),
        EnhancementMode.QHD_2K to EnhancementProfile(
            id = "QHD_2K",
            displayName = "2K",
            description = "Enhance quality and upscale to 1440p",
            targetLongEdge = 2560,
            targetShortEdge = 1440,
            preserveOriginalResolution = false,
            useAiSuperResolution = true,
            useMultiStageUpscaling = false,
            useTiledInference = true,
            temporalConsistency = true,
            brightnessPercent = 3f,
            contrastPercent = 12f,
            sharpnessPercent = 25f,
            denoisePercent = 20f,
            saturationPercent = 7f,
            gamma = 1.05f,
            whiteBalance = "Automatic",
            colorCorrection = "Automatic",
            deblocking = "Medium",
            dehazingPercent = 10f,
            edgeEnhancementPercent = 20f,
            flickerReduction = "Automatic",
            preferredCodec = "H265",
            targetBpp = 0.14f
        ),
        EnhancementMode.UHD_4K to EnhancementProfile(
            id = "UHD_4K",
            displayName = "4K",
            description = "Enhance quality and upscale to 2160p",
            targetLongEdge = 3840,
            targetShortEdge = 2160,
            preserveOriginalResolution = false,
            useAiSuperResolution = true,
            useMultiStageUpscaling = true,
            useTiledInference = true,
            temporalConsistency = true,
            brightnessPercent = 3f,
            contrastPercent = 12f,
            sharpnessPercent = 25f,
            denoisePercent = 20f,
            saturationPercent = 7f,
            gamma = 1.05f,
            whiteBalance = "Automatic",
            colorCorrection = "Automatic",
            deblocking = "Medium",
            dehazingPercent = 10f,
            edgeEnhancementPercent = 20f,
            flickerReduction = "Automatic",
            preferredCodec = "H265",
            targetBpp = 0.12f
        )
    )
}
