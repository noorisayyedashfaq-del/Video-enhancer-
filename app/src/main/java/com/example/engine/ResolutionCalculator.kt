package com.example.engine

import com.example.model.DeviceCapabilities
import com.example.model.EnhancementMode
import com.example.model.EnhancementProfile
import com.example.model.OutputDimensions
import com.example.model.VideoMetadata
import kotlin.math.roundToInt

/**
 * Resolves output video configuration for each enhancement preset.
 * All presets (HD, HD+, 2K, 4K) keep original video resolution unchanged while
 * configuring the preset's AI tile size and target tile count per frame.
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

        val (tileSize, tilesPerFrame) = when (mode) {
            EnhancementMode.HD -> 0 to 0
            EnhancementMode.HD_PLUS ->
                (if (profile.aiTileSize > 0) profile.aiTileSize else 512) to
                    (if (profile.targetTilesPerFrame > 0) profile.targetTilesPerFrame else 6)
            EnhancementMode.QHD_2K ->
                (if (profile.aiTileSize > 0) profile.aiTileSize else 512) to
                    (if (profile.targetTilesPerFrame > 0) profile.targetTilesPerFrame else 12)
            EnhancementMode.UHD_4K ->
                (if (profile.aiTileSize > 0) profile.aiTileSize else 512) to
                    (if (profile.targetTilesPerFrame > 0) profile.targetTilesPerFrame else 25)
        }

        val useAiTiles = mode != EnhancementMode.HD && tileSize > 0 && tilesPerFrame > 0
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
