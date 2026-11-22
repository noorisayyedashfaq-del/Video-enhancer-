package com.example.engine

import com.example.model.DeviceCapabilities
import com.example.model.EnhancementMode
import com.example.model.EnhancementProfile
import com.example.model.OutputDimensions
import com.example.model.VideoMetadata
import kotlin.math.roundToInt

/**
 * Resolves output video configuration for the 4 Clarity levels (1x, 2x, 3x, 4x).
 * All presets keep original video resolution unchanged while configuring
 * 512px max tile size and 4 / 6 / 12 / 25 tiles per frame.
 */
object ResolutionCalculator {

    fun calculate(
        metadata: VideoMetadata,
        mode: EnhancementMode,
        profile: EnhancementProfile,
        capabilities: DeviceCapabilities?
    ): OutputDimensions {
        val evenInW = makeEven(metadata.displayWidth.coerceAtLeast(2))
        val evenInH = makeEven(metadata.displayHeight.coerceAtLeast(2))

        val preferH265 = profile.preferredCodec.equals("H265", ignoreCase = true) &&
            (capabilities?.hardwareH265Supported == true)

        val defaultTiles = when (mode) {
            EnhancementMode.CLARITY_1X -> 4
            EnhancementMode.CLARITY_2X -> 6
            EnhancementMode.CLARITY_3X -> 12
            EnhancementMode.CLARITY_4X -> 25
        }

        val tileSize = (if (profile.aiTileSize > 0) profile.aiTileSize else 512).coerceIn(128, 512)
        val tilesPerFrame = if (profile.targetTilesPerFrame > 0) profile.targetTilesPerFrame else defaultTiles
        val useAiTiles = tileSize > 0 && tilesPerFrame > 0

        val bitrate = computeTargetBitrate(evenInW, evenInH, metadata.fps, profile.targetBpp, preferH265)

        return OutputDimensions(
            width = evenInW,
            height = evenInH,
            scaleFactorX = 1f,
            scaleFactorY = 1f,
            needsAiSuperResolution = useAiTiles,
            needsMultiStageUpscaling = false,
            preserveOriginalResolution = true,
            targetBitrateBps = bitrate,
            useH265 = preferH265,
            aiTileSize = tileSize,
            tilesPerFrame = tilesPerFrame
        )
    }

    private fun makeEven(value: Int): Int {
        val aligned = (value / 2) * 2
        return if (aligned < 2) 2 else aligned
    }

    private fun computeTargetBitrate(
        width: Int,
        height: Int,
        fps: Double,
        bpp: Float,
        useH265: Boolean
    ): Int {
        val clampedFps = fps.coerceIn(15.0, 60.0)
        val codecEfficiency = if (useH265) 0.78 else 1.0
        val raw = (width.toDouble() * height.toDouble() * clampedFps * bpp * codecEfficiency).roundToInt()
        return raw.coerceIn(2_500_000, 48_000_000)
    }
}
