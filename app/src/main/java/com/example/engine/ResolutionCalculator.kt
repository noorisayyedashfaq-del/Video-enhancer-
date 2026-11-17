package com.example.engine

import com.example.model.DeviceCapabilities
import com.example.model.EnhancementMode
import com.example.model.EnhancementProfile
import com.example.model.OutputDimensions
import com.example.model.VideoMetadata
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

object ResolutionCalculator {

    fun calculate(
        metadata: VideoMetadata,
        mode: EnhancementMode,
        profile: EnhancementProfile,
        capabilities: DeviceCapabilities?
    ): OutputDimensions {
        val inW = metadata.displayWidth.coerceAtLeast(2)
        val inH = metadata.displayHeight.coerceAtLeast(2)

        val evenInW = makeEven(inW)
        val evenInH = makeEven(inH)

        val preferH265 = profile.preferredCodec.equals("H265", ignoreCase = true) &&
            (capabilities?.hardwareH265Supported == true)

        if (profile.preserveOriginalResolution || mode == EnhancementMode.HD) {
            val bitrate = computeTargetBitrate(evenInW, evenInH, metadata.fps, profile.targetBpp, preferH265)
            return OutputDimensions(
                width = evenInW,
                height = evenInH,
                scaleFactorX = 1f,
                scaleFactorY = 1f,
                needsAiSuperResolution = false,
                needsMultiStageUpscaling = false,
                preserveOriginalResolution = true,
                targetBitrateBps = bitrate,
                useH265 = preferH265
            )
        }

        // Preserve portrait vs landscape orientation and exact aspect ratio.
        val isPortrait = inH > inW
        val targetMaxW = if (isPortrait) profile.targetShortEdge else profile.targetLongEdge
        val targetMaxH = if (isPortrait) profile.targetLongEdge else profile.targetShortEdge

        // Never downscale if input already meets or exceeds selected target bounding box.
        val inLongEdge = max(evenInW, evenInH)
        val inShortEdge = min(evenInW, evenInH)
        val targetLong = max(targetMaxW, targetMaxH)
        val targetShort = min(targetMaxW, targetMaxH)

        if (inLongEdge >= targetLong || inShortEdge >= targetShort) {
            val bitrate = computeTargetBitrate(evenInW, evenInH, metadata.fps, profile.targetBpp, preferH265)
            return OutputDimensions(
                width = evenInW,
                height = evenInH,
                scaleFactorX = 1f,
                scaleFactorY = 1f,
                needsAiSuperResolution = false,
                needsMultiStageUpscaling = false,
                preserveOriginalResolution = true,
                targetBitrateBps = bitrate,
                useH265 = preferH265
            )
        }

        // Fit inside target bounding box while preserving original aspect ratio cleanly.
        val scaleW = targetMaxW.toDouble() / evenInW.toDouble()
        val scaleH = targetMaxH.toDouble() / evenInH.toDouble()
        val scale = min(scaleW, scaleH).coerceAtLeast(1.0)

        val outW = makeEven((evenInW * scale).roundToInt()).coerceAtLeast(evenInW)
        val outH = makeEven((evenInH * scale).roundToInt()).coerceAtLeast(evenInH)

        val actualScaleX = outW.toFloat() / evenInW.toFloat()
        val actualScaleY = outH.toFloat() / evenInH.toFloat()
        val maxScale = max(actualScaleX, actualScaleY)

        // Multi-stage upscaling when scale factor > 2.2x or 4K mode from <= 540p source
        val needsMultiStage = profile.useMultiStageUpscaling && (maxScale > 2.2f || inShortEdge <= 540)
        val bitrate = computeTargetBitrate(outW, outH, metadata.fps, profile.targetBpp, preferH265)

        return OutputDimensions(
            width = outW,
            height = outH,
            scaleFactorX = actualScaleX,
            scaleFactorY = actualScaleY,
            needsAiSuperResolution = profile.useAiSuperResolution && maxScale > 1.05f,
            needsMultiStageUpscaling = needsMultiStage,
            preserveOriginalResolution = false,
            targetBitrateBps = bitrate,
            useH265 = preferH265
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
