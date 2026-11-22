package com.example.engine

import android.content.Context
import android.graphics.Bitmap
import com.example.model.AdaptiveAdjustmentParams
import com.example.model.DeviceCapabilities
import com.example.model.OutputDimensions
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.File
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.pow
import kotlin.math.roundToInt

data class FrameProcessResult(
    val facesEnhanced: Int = 0,
    val lightweightFallbackTriggered: Boolean = false,
    val fallbackReason: String? = null,
    val activeAccelerationLabel: String = ""
)

class OpenCvFrameProcessor(
    private val context: Context,
    private val onnxEngine: OnnxAiInferenceEngine
) {
    private var openCvInitialized = false
    private var faceDetectorPath: String? = null

    // Persistent temporal & reusable scratch Mats (eliminates per-frame native heap churn)
    private var prevDenoisedFrame: Mat? = null
    private var prevLumaMean: Double? = null
    private var reusableRgbaMat: Mat? = null
    private var ycrcbScratch: Mat? = null
    private var hsvScratch: Mat? = null
    private var diffScratch: Mat? = null
    private var grayDiffScratch: Mat? = null
    private var staticMaskScratch: Mat? = null
    private var blendScratch: Mat? = null
    private var blurScratch: Mat? = null
    private var claheOutScratch: Mat? = null
    private var alignedScratch: Mat? = null
    private var skinMaskScratch: Mat? = null
    private var nonSkinMaskScratch: Mat? = null
    private var satBoostScratch: Mat? = null
    private var satHalfScratch: Mat? = null
    private var toneLutMat: Mat? = null
    private var cachedLutKey: String = ""
    private var satLutMat: Mat? = null
    private var halfSatLutMat: Mat? = null
    private var cachedSatKey: String = ""

    val isOpenCvReady: Boolean get() = openCvInitialized
    val isFaceDetectorReady: Boolean get() = faceDetectorPath != null

    @Synchronized
    fun ensureInitialized(): Boolean {
        if (!openCvInitialized) {
            openCvInitialized = try {
                OpenCVLoader.initLocal()
            } catch (_: Throwable) {
                false
            }
        }
        if (openCvInitialized && faceDetectorPath == null) {
            try {
                val dest = File(context.filesDir, "ai_models/face_enhancement/face_detection_yunet_2023mar.onnx")
                dest.parentFile?.mkdirs()
                if (!dest.exists() || dest.length() == 0L) {
                    context.assets.open("ai_models/face_enhancement/face_detection_yunet_2023mar.onnx").use { inp ->
                        dest.outputStream().use { out -> inp.copyTo(out) }
                    }
                }
                if (dest.exists() && dest.length() > 0L) {
                    faceDetectorPath = dest.absolutePath
                }
            } catch (_: Exception) {
            }
        }
        return openCvInitialized
    }

    fun resetTemporalState() {
        prevDenoisedFrame?.release(); prevDenoisedFrame = null
        prevLumaMean = null
        reusableRgbaMat?.release(); reusableRgbaMat = null
        ycrcbScratch?.release(); ycrcbScratch = null
        hsvScratch?.release(); hsvScratch = null
        diffScratch?.release(); diffScratch = null
        grayDiffScratch?.release(); grayDiffScratch = null
        staticMaskScratch?.release(); staticMaskScratch = null
        blendScratch?.release(); blendScratch = null
        blurScratch?.release(); blurScratch = null
        claheOutScratch?.release(); claheOutScratch = null
        alignedScratch?.release(); alignedScratch = null
        skinMaskScratch?.release(); skinMaskScratch = null
        nonSkinMaskScratch?.release(); nonSkinMaskScratch = null
        satBoostScratch?.release(); satBoostScratch = null
        satHalfScratch?.release(); satHalfScratch = null
        toneLutMat?.release(); toneLutMat = null
        cachedLutKey = ""
        satLutMat?.release(); satLutMat = null
        halfSatLutMat?.release(); halfSatLutMat = null
        cachedSatKey = ""
    }

    /**
     * Consolidated fast per-frame enhancement pipeline using fused YCrCb/HSV passes and
     * reusable scratch Mats, plus AI tile or lightweight GPU/SIMD enhancement.
     * Avoids double sharpening: when AI tiles run, neural restoration provides the clarity pass once.
     */
    fun processFrame(
        srcBgr: Mat,
        outDims: OutputDimensions,
        params: AdaptiveAdjustmentParams,
        capabilities: DeviceCapabilities,
        reusableOutBitmap: Bitmap,
        estimatedTotalFrames: Int,
        forceLightweightFallback: Boolean,
        tileSizeOverride: Int? = null,
        logger: EngineDiagnosticsLogger? = null,
        isCancelled: () -> Boolean,
        onUpscaleTileProgress: (done: Int, total: Int, tileSize: Int) -> Unit = { _, _, _ -> }
    ): FrameProcessResult {
        var workMat = srcBgr.clone()
        var usedFallback = forceLightweightFallback
        var fallbackReason: String? = null
        var accelLabel = onnxEngine.providerLabel

        try {
            // 1. Temporal Motion-Stabilized Noise Reduction + Gentle Artifact/Deblock Smoothing
            applyTemporalMotionStabilizedDenoise(workMat, params.denoiseStrength)
            if (params.deblockingStrength > 0.25f || params.denoiseStrength > 0.28f) {
                val blur = blurScratch ?: Mat().also { blurScratch = it }
                val smoothWeight = (0.14 + params.deblockingStrength * 0.22).coerceIn(0.14, 0.34)
                Imgproc.GaussianBlur(workMat, blur, Size(3.0, 3.0), 0.65)
                Core.addWeighted(blur, smoothWeight, workMat, 1.0 - smoothWeight, 0.0, workMat)
            }

            // 2. Single Fused Luma Pass: Brilliance Dehaze + Restrained Clarity CLAHE + Tone LUT + Flicker Reduction
            applyFusedYCrCbMastering(workMat, params)

            // 3. Neutral Temperature / White Balance & Skin-Safe Saturation
            if (abs(params.wbGainR - 1f) >= 0.01f || abs(params.wbGainB - 1f) >= 0.01f) {
                applyWhiteBalance(workMat, params)
            }
            if (abs(params.saturationGain - 1f) > 0.01f) {
                applySaturationBoost(workMat, params.saturationGain, params.skinToneProtection)
            }

            // 4. AI Tile Enhancement (existing Real-ESRGAN ONNX model, or single-pass GPU/SIMD fallback)
            if (outDims.needsAiSuperResolution && outDims.tilesPerFrame > 0 && outDims.aiTileSize > 0) {
                val tAi0 = System.currentTimeMillis()
                val activeTileSize = tileSizeOverride ?: outDims.aiTileSize
                val outcome = onnxEngine.enhanceBgrWithPresetTiles(
                    srcBgr = workMat,
                    tileSize = activeTileSize,
                    targetTilesPerFrame = outDims.tilesPerFrame,
                    estimatedTotalFrames = estimatedTotalFrames,
                    forceLightweightFallback = forceLightweightFallback,
                    aiStrength = params.aiStrength,
                    fallbackSharpnessBoost = params.sharpnessAmount,
                    isCancelled = isCancelled,
                    logger = logger,
                    onTileProgress = onUpscaleTileProgress
                )
                logger?.aiInferenceMs?.addAndGet(System.currentTimeMillis() - tAi0)
                usedFallback = outcome.usedLightweightFallback
                fallbackReason = outcome.fallbackReason
                accelLabel = outcome.activeAccelerationLabel
                workMat.release()
                workMat = outcome.outputBgr
            } else if (onnxEngine.isGlesGpuReady && (params.sharpnessAmount > 0.03f || params.edgeEnhancementAmount > 0.03f)) {
                val sharpness = (params.sharpnessAmount + params.edgeEnhancementAmount * 0.35f).coerceIn(0.05f, 0.58f)
                val gpuOk = onnxEngine.gpuProcessor.enhanceBgrInPlace(
                    bgr = workMat,
                    sharpness = sharpness,
                    contrast = 1.02f,
                    saturation = 1.0f
                )
                if (gpuOk) {
                    accelLabel = "GLES 3.1 GPU"
                } else {
                    applyFastUnsharpMask(workMat, params.sharpnessAmount, params.edgeEnhancementAmount)
                }
            } else if (params.sharpnessAmount > 0.03f || params.edgeEnhancementAmount > 0.03f) {
                applyFastUnsharpMask(workMat, params.sharpnessAmount, params.edgeEnhancementAmount)
            }

            // Ensure exact original dimensions (zero cropping)
            if (workMat.cols() != outDims.width || workMat.rows() != outDims.height) {
                val aligned = alignedScratch ?: Mat().also { alignedScratch = it }
                Imgproc.resize(
                    workMat,
                    aligned,
                    Size(outDims.width.toDouble(), outDims.height.toDouble()),
                    0.0,
                    0.0,
                    Imgproc.INTER_LINEAR
                )
                workMat.release()
                workMat = aligned.clone()
            }

            val rgbaMat = reusableRgbaMat?.takeIf { it.cols() == workMat.cols() && it.rows() == workMat.rows() }
                ?: Mat(workMat.rows(), workMat.cols(), CvType.CV_8UC4).also {
                    reusableRgbaMat?.release()
                    reusableRgbaMat = it
                }
            Imgproc.cvtColor(workMat, rgbaMat, Imgproc.COLOR_BGR2RGBA)
            Utils.matToBitmap(rgbaMat, reusableOutBitmap)
        } finally {
            workMat.release()
        }

        return FrameProcessResult(
            facesEnhanced = 0,
            lightweightFallbackTriggered = usedFallback,
            fallbackReason = fallbackReason,
            activeAccelerationLabel = accelLabel
        )
    }

    private fun applyTemporalMotionStabilizedDenoise(bgr: Mat, strength: Float) {
        val prev = prevDenoisedFrame
        if (prev == null || prev.cols() != bgr.cols() || prev.rows() != bgr.rows()) {
            prevDenoisedFrame?.release()
            prevDenoisedFrame = bgr.clone()
            return
        }

        val diff = diffScratch ?: Mat().also { diffScratch = it }
        val grayDiff = grayDiffScratch ?: Mat().also { grayDiffScratch = it }
        val staticMask = staticMaskScratch ?: Mat().also { staticMaskScratch = it }
        val blended = blendScratch ?: Mat().also { blendScratch = it }

        Core.absdiff(bgr, prev, diff)
        Imgproc.cvtColor(diff, grayDiff, Imgproc.COLOR_BGR2GRAY)
        Imgproc.threshold(grayDiff, staticMask, 16.0, 255.0, Imgproc.THRESH_BINARY_INV)
        val alpha = (0.08 + strength * 0.16).coerceIn(0.08, 0.24)
        Core.addWeighted(bgr, 1.0 - alpha, prev, alpha, 0.0, blended)
        blended.copyTo(bgr, staticMask)
        bgr.copyTo(prev)
    }

    /**
     * Single-pass YCrCb luminance mastering: preserves natural Cr/Cb chroma balance
     * while applying subtle brilliance black-lift, gentle clarity CLAHE (<= 24% blend),
     * calibrated tone LUT (brightness, contrast, highlights, shadows, whites, blacks),
     * and temporal flicker damping.
     */
    private fun applyFusedYCrCbMastering(bgr: Mat, params: AdaptiveAdjustmentParams) {
        val ycrcb = ycrcbScratch ?: Mat().also { ycrcbScratch = it }
        val channels = ArrayList<Mat>(3)
        try {
            Imgproc.cvtColor(bgr, ycrcb, Imgproc.COLOR_BGR2YCrCb)
            Core.split(ycrcb, channels)
            val yChan = channels[0]

            // Subtle brilliance black-point lift when washed hazy
            if (params.dehazingStrength > 0.02f) {
                val minVal = Core.minMaxLoc(yChan).minVal
                val blackLift = (minVal * params.dehazingStrength * 0.45).coerceIn(0.0, 12.0)
                if (blackLift > 0.5) {
                    val scale = 255.0 / (255.0 - blackLift)
                    yChan.convertTo(yChan, CvType.CV_8UC1, scale, -blackLift * scale)
                }
            }

            // Restrained local clarity via CLAHE (capped at 0.24 blend weight to prevent unnatural HDR look)
            if (params.claheClipLimit > 0.20) {
                val clahe = Imgproc.createCLAHE(params.claheClipLimit, Size(8.0, 8.0))
                val claheOut = claheOutScratch ?: Mat().also { claheOutScratch = it }
                clahe.apply(yChan, claheOut)
                Core.addWeighted(claheOut, 0.24, yChan, 0.76, 0.0, yChan)
            }

            // Combined monotonic LUT: brightness + contrast + highlights + shadows + whites + blacks + gamma
            val lut = getOrCreateToneLut(params)
            Core.LUT(yChan, lut, yChan)

            // Flicker stabilization inside the exact same Y channel
            if (params.flickerReductionAlpha > 0.01f) {
                val currentMean = Core.mean(yChan).`val`[0]
                val prev = prevLumaMean
                if (prev == null) {
                    prevLumaMean = currentMean
                } else {
                    val targetMean = prev * params.flickerReductionAlpha + currentMean * (1.0 - params.flickerReductionAlpha)
                    val delta = (targetMean - currentMean).coerceIn(-6.0, 6.0)
                    if (abs(delta) > 0.35) {
                        yChan.convertTo(yChan, CvType.CV_8UC1, 1.0, delta)
                    }
                    prevLumaMean = targetMean
                }
            }

            Core.merge(channels, ycrcb)
            Imgproc.cvtColor(ycrcb, bgr, Imgproc.COLOR_YCrCb2BGR)
        } finally {
            channels.forEach { it.release() }
        }
    }

    private fun getOrCreateToneLut(params: AdaptiveAdjustmentParams): Mat {
        val key = buildString {
            append(params.gamma).append('|')
            append(params.contrastGain).append('|')
            append(params.brightnessShift).append('|')
            append(params.exposureHighlightCompress).append('|')
            append(params.shadowsLift).append('|')
            append(params.whitesGain).append('|')
            append(params.blacksCrush)
        }
        val existing = toneLutMat
        if (existing != null && cachedLutKey == key) {
            return existing
        }
        val lut = existing ?: Mat(1, 256, CvType.CV_8UC1).also { toneLutMat = it }
        lut.put(0, 0, buildToneLutArray(params))
        cachedLutKey = key
        return lut
    }

    private fun applyWhiteBalance(bgr: Mat, params: AdaptiveAdjustmentParams) {
        val channels = ArrayList<Mat>(3)
        try {
            Core.split(bgr, channels) // B, G, R
            if (abs(params.wbGainB - 1f) >= 0.01f) {
                channels[0].convertTo(channels[0], CvType.CV_8UC1, params.wbGainB.toDouble(), 0.0)
            }
            if (abs(params.wbGainR - 1f) >= 0.01f) {
                channels[2].convertTo(channels[2], CvType.CV_8UC1, params.wbGainR.toDouble(), 0.0)
            }
            Core.merge(channels, bgr)
        } finally {
            channels.forEach { it.release() }
        }
    }

    /**
     * Soft highlight-safe saturation adjustment with natural skin-tone protection.
     * Uses cached 256-entry saturation LUTs to prevent color clipping and preserves skin hue range (H in 2..22).
     */
    private fun applySaturationBoost(bgr: Mat, satGain: Float, protectSkin: Boolean) {
        val hsv = hsvScratch ?: Mat().also { hsvScratch = it }
        val channels = ArrayList<Mat>(3)
        try {
            Imgproc.cvtColor(bgr, hsv, Imgproc.COLOR_BGR2HSV)
            Core.split(hsv, channels)
            ensureSatLuts(satGain)

            if (!protectSkin) {
                Core.LUT(channels[1], satLutMat!!, channels[1])
            } else {
                val skinMask = skinMaskScratch ?: Mat().also { skinMaskScratch = it }
                val nonSkinMask = nonSkinMaskScratch ?: Mat().also { nonSkinMaskScratch = it }
                val boostMat = satBoostScratch ?: Mat().also { satBoostScratch = it }
                val halfMat = satHalfScratch ?: Mat().also { satHalfScratch = it }

                // OpenCV Hue 0..179: natural skin tones fall in warm orange-red 2..22
                Core.inRange(channels[0], Scalar(2.0), Scalar(22.0), skinMask)
                Core.bitwise_not(skinMask, nonSkinMask)

                Core.LUT(channels[1], satLutMat!!, boostMat)
                Core.LUT(channels[1], halfSatLutMat!!, halfMat)
                boostMat.copyTo(channels[1], nonSkinMask)
                halfMat.copyTo(channels[1], skinMask)
            }

            Core.merge(channels, hsv)
            Imgproc.cvtColor(hsv, bgr, Imgproc.COLOR_HSV2BGR)
        } finally {
            channels.forEach { it.release() }
        }
    }

    private fun ensureSatLuts(satGain: Float) {
        val key = String.format(java.util.Locale.US, "%.3f", satGain)
        if (satLutMat != null && halfSatLutMat != null && cachedSatKey == key) return

        val fullLut = satLutMat ?: Mat(1, 256, CvType.CV_8UC1).also { satLutMat = it }
        val skinLut = halfSatLutMat ?: Mat(1, 256, CvType.CV_8UC1).also { halfSatLutMat = it }

        fullLut.put(0, 0, buildSaturationLutArray(satGain))
        val dampedGain = 1f + (satGain - 1f) * 0.42f
        skinLut.put(0, 0, buildSaturationLutArray(dampedGain))
        cachedSatKey = key
    }

    private fun applyFastUnsharpMask(bgr: Mat, sharpnessAmount: Float, edgeEnhanceAmount: Float) {
        val totalAmount = (sharpnessAmount + edgeEnhanceAmount * 0.35f).coerceIn(0.05f, 0.58f)
        val blurred = blurScratch ?: Mat().also { blurScratch = it }
        Imgproc.GaussianBlur(bgr, blurred, Size(0.0, 0.0), 1.05)
        Core.addWeighted(bgr, 1.0 + totalAmount, blurred, -totalAmount.toDouble(), 0.0, bgr)
    }

    companion object {
        /**
         * Builds a strictly monotonic 256-entry luma LUT combining gamma, contrast, brightness,
         * black-point anchor, shadow detail recovery, whites crispness, and highlight roll-off
         * without clipping or unnatural HDR tone inversion.
         */
        fun buildToneLutArray(params: AdaptiveAdjustmentParams): ByteArray {
            val lutData = ByteArray(256)
            val invGamma = params.gamma.toDouble().coerceIn(0.80, 1.25)
            val shadowsAmt = params.shadowsLift.toDouble().coerceIn(0.0, 0.25)
            val blacksAmt = params.blacksCrush.toDouble().coerceIn(0.0, 0.15)
            val whitesAmt = params.whitesGain.toDouble().coerceIn(0.0, 0.20)
            val hlCompress = params.exposureHighlightCompress.toDouble().coerceIn(0.0, 0.40)

            var prevOut = 0
            for (i in 0..255) {
                var v = (i / 255.0).pow(invGamma) * 255.0

                // 1. Blacks anchor (smoothly deepens deep blacks under ~40 without crushing mid-shadows)
                if (blacksAmt > 0.001 && i < 64) {
                    val weight = ((64.0 - i) / 64.0).pow(1.6)
                    v -= weight * blacksAmt * 28.0
                }

                // 2. Shadow detail lift (smooth Gaussian centered around i=64, preserving black=0)
                if (shadowsAmt > 0.001) {
                    val dist = (i - 64.0) / 36.0
                    val weight = exp(-0.5 * dist * dist) * (i / 24.0).coerceAtMost(1.0)
                    v += weight * shadowsAmt * 95.0
                }

                // 3. Contrast & Brightness around mid-gray 128
                v = ((v - 128.0) * params.contrastGain.toDouble()) + 128.0 + params.brightnessShift.toDouble()

                // 4. Whites crispness (smooth Gaussian centered near i=192, tapering to 0 at 255)
                if (whitesAmt > 0.001) {
                    val dist = (i - 192.0) / 30.0
                    val taper = ((255.0 - i) / 28.0).coerceIn(0.0, 1.0)
                    val weight = exp(-0.5 * dist * dist) * taper
                    v += weight * whitesAmt * 75.0
                }

                // 5. Highlight clipping roll-off above 195
                if (hlCompress > 0.005 && v > 195.0) {
                    val excess = v - 195.0
                    v = 195.0 + excess * (1.0 - hlCompress)
                }

                val clamped = v.roundToInt().coerceIn(prevOut, 255)
                prevOut = clamped
                lutData[i] = clamped.toByte()
            }
            return lutData
        }

        /**
         * Soft saturation LUT that prevents chroma clipping as saturation approaches 255.
         */
        fun buildSaturationLutArray(satGain: Float): ByteArray {
            val lutData = ByteArray(256)
            val gainMinusOne = (satGain - 1f).toDouble()
            for (s in 0..255) {
                val norm = s / 255.0
                val rolloff = 1.0 - norm.pow(1.55)
                val out = (s * (1.0 + gainMinusOne * rolloff)).roundToInt().coerceIn(0, 255)
                lutData[s] = out.toByte()
            }
            return lutData
        }
    }
}
