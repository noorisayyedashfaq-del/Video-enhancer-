package com.example.engine

import com.example.model.DeviceCapabilities
import com.example.model.EnhancementMode
import com.example.model.EnhancementProfile
import com.example.model.OutputDimensions
import com.example.model.VideoMetadata
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Resolves progressive AI super-resolution output dimensions, smart aspect-preserving
 * tile counts, target bitrate, and codec selection across Clarity 1x - 4x.
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

        val rawScale = when {
            profile.preserveOriginalResolution -> 1.0f
            profile.targetScaleFactor > 1.0f -> profile.targetScaleFactor
            else -> when (mode) {
                EnhancementMode.CLARITY_1X -> 1.0f
                EnhancementMode.CLARITY_2X -> 2.0f
                EnhancementMode.CLARITY_3X -> 3.0f
                EnhancementMode.CLARITY_4X -> 4.0f
            }
        }

        val hardwareMaxLongEdge = (capabilities?.maxEncodeLongEdge ?: 3840).coerceAtLeast(1920)
        val profileMaxLongEdge = if (profile.maxOutputLongEdge > 0) {
            min(profile.maxOutputLongEdge, hardwareMaxLongEdge)
        } else {
            hardwareMaxLongEdge
        }

        val inLongEdge = max(evenInW, evenInH)
        val effectiveScale = if (profile.preserveOriginalResolution || rawScale <= 1.01f) {
            1.0f
        } else {
            val maxAllowedScale = profileMaxLongEdge.toFloat() / inLongEdge.toFloat()
            min(rawScale, maxAllowedScale).coerceAtLeast(1.0f)
        }

        val outW = if (effectiveScale <= 1.01f) {
            evenInW
        } else {
            makeEven((evenInW * effectiveScale).roundToInt())
        }
        val outH = if (effectiveScale <= 1.01f) {
            evenInH
        } else {
            makeEven((evenInH * effectiveScale).roundToInt())
        }

        val actualScaleX = outW.toFloat() / evenInW.toFloat()
        val actualScaleY = outH.toFloat() / evenInH.toFloat()
        val actualScale = (actualScaleX + actualScaleY) * 0.5f

        val baseTileSize = (if (profile.aiTileSize > 0) profile.aiTileSize else {
            when (mode) {
                EnhancementMode.CLARITY_1X, EnhancementMode.CLARITY_2X -> 192
                EnhancementMode.CLARITY_3X -> 224
                EnhancementMode.CLARITY_4X -> 256
            }
        }).coerceIn(128, 384)

        val tileBudgetCap = if (profile.targetTilesPerFrame > 0) profile.targetTilesPerFrame else {
            when (mode) {
                EnhancementMode.CLARITY_1X -> 1
                EnhancementMode.CLARITY_2X -> 2
                EnhancementMode.CLARITY_3X -> 3
                EnhancementMode.CLARITY_4X -> 4
            }
        }
        val neededTiles = computeSmartTileCount(
            inW = evenInW,
            inH = evenInH,
            tileSize = baseTileSize,
            maxTilesCap = tileBudgetCap
        )

        val bitrate = computeTargetBitrate(outW, outH, metadata.fps, profile.targetBpp, preferH265)

        return OutputDimensions(
            width = outW,
            height = outH,
            scaleFactorX = actualScaleX,
            scaleFactorY = actualScaleY,
            needsAiSuperResolution = profile.useAiSuperResolution && baseTileSize > 0 && neededTiles > 0,
            needsMultiStageUpscaling = actualScale > 2.05f,
            preserveOriginalResolution = effectiveScale <= 1.01f,
            targetBitrateBps = bitrate,
            useH265 = preferH265,
            aiTileSize = baseTileSize,
            tilesPerFrame = neededTiles,
            targetScaleFactor = actualScale
        )
    }

    /**
     * Computes only the tiles actually needed to cover neural working resolution cleanly
     * with strict [maxTilesCap] enforcement so processing stays within time budgets.
     */
    fun computeSmartTileCount(
        inW: Int,
        inH: Int,
        tileSize: Int,
        maxTilesCap: Int
    ): Int {
        val safeW = inW.coerceAtLeast(2)
        val safeH = inH.coerceAtLeast(2)
        val safeTile = tileSize.coerceIn(128, 384)
        val overlap = (safeTile * 0.10f).roundToInt().coerceIn(12, 32)
        val step = (safeTile - overlap).coerceAtLeast(64)

        val cols = if (safeW <= safeTile) 1 else ceil((safeW - overlap).toDouble() / step.toDouble()).toInt().coerceAtLeast(2)
        val rows = if (safeH <= safeTile) 1 else ceil((safeH - overlap).toDouble() / step.toDouble()).toInt().coerceAtLeast(2)
        val naturalTiles = (cols * rows).coerceAtLeast(1)
        return naturalTiles.coerceAtMost(maxTilesCap.coerceAtLeast(1))
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
        return raw.coerceIn(3_000_000, 65_000_000)
    }
}
