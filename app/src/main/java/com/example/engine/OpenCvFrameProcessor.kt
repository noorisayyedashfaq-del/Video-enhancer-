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
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.File
import kotlin.math.abs
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
    private var toneLutMat: Mat? = null
    private var cachedLutKey: String = ""

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
        toneLutMat?.release(); toneLutMat = null
        cachedLutKey = ""
    }

    /**
     * Consolidated fast per-frame enhancement pipeline using fused YCrCb/HSV passes and
     * reusable scratch Mats, plus AI tile or lightweight GPU/SIMD super-resolution pass.
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
            // 1. Fast Temporal Motion-Stabilized Denoise + Lightweight Spatial Smooth
            applyTemporalMotionStabilizedDenoise(workMat, params.denoiseStrength)
            if (params.deblockingStrength > 0.35f || params.denoiseStrength > 0.25f) {
                val blur = blurScratch ?: Mat().also { blurScratch = it }
                val smoothWeight = if (params.deblockingStrength > 0.35f) 0.32 else 0.22
                Imgproc.GaussianBlur(workMat, blur, Size(3.0, 3.0), 0.7)
                Core.addWeighted(blur, smoothWeight, workMat, 1.0 - smoothWeight, 0.0, workMat)
            }

            // 2. Single Fused Luma Pass: Dehaze + CLAHE + Contrast/Brightness/Gamma LUT + Flicker Reduction
            applyFusedYCrCbMastering(workMat, params)

            // 3. White Balance & Saturation
            if (abs(params.wbGainR - 1f) >= 0.01f || abs(params.wbGainB - 1f) >= 0.01f) {
                applyWhiteBalance(workMat, params)
            }
            if (abs(params.saturationGain - 1f) > 0.015f) {
                applySaturationBoost(workMat, params.saturationGain)
            }

            // 4. Preset AI Tile Enhancement (or automatic GLES GPU / SIMD Lightweight Fallback)
            if (outDims.needsAiSuperResolution && outDims.tilesPerFrame > 0 && outDims.aiTileSize > 0) {
                val tAi0 = System.currentTimeMillis()
                val activeTileSize = tileSizeOverride ?: outDims.aiTileSize
                val outcome = onnxEngine.enhanceBgrWithPresetTiles(
                    srcBgr = workMat,
                    tileSize = activeTileSize,
                    targetTilesPerFrame = outDims.tilesPerFrame,
                    estimatedTotalFrames = estimatedTotalFrames,
                    forceLightweightFallback = forceLightweightFallback,
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
                // HD fast GPU clarity pass
                val sharpness = (params.sharpnessAmount + params.edgeEnhancementAmount * 0.45f).coerceIn(0.05f, 0.75f)
                val gpuOk = onnxEngine.gpuProcessor.enhanceBgrInPlace(
                    bgr = workMat,
                    sharpness = sharpness,
                    contrast = 1.02f,
                    saturation = 1.01f
                )
                if (gpuOk) {
                    accelLabel = "GLES 3.1 GPU HD"
                } else {
                    applyFastUnsharpMask(workMat, params.sharpnessAmount, params.edgeEnhancementAmount)
                }
            } else if (params.sharpnessAmount > 0.03f || params.edgeEnhancementAmount > 0.03f) {
                applyFastUnsharpMask(workMat, params.sharpnessAmount, params.edgeEnhancementAmount)
            }

            // 5. Output Clarity Polish if lightweight fallback ran and sharpness > threshold
            if (usedFallback && outDims.needsAiSuperResolution && !onnxEngine.isGlesGpuReady &&
                (params.sharpnessAmount > 0.03f || params.edgeEnhancementAmount > 0.03f)
            ) {
                applyFastUnsharpMask(workMat, params.sharpnessAmount * 0.65f, params.edgeEnhancementAmount * 0.65f)
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
        val alpha = (0.10 + strength * 0.16).coerceIn(0.08, 0.26)
        Core.addWeighted(bgr, 1.0 - alpha, prev, alpha, 0.0, blended)
        blended.copyTo(bgr, staticMask)
        bgr.copyTo(prev)
    }

    /**
     * Single-pass YCrCb processing replacing three redundant full-frame BGR <-> YCrCb conversions.
     */
    private fun applyFusedYCrCbMastering(bgr: Mat, params: AdaptiveAdjustmentParams) {
        val ycrcb = ycrcbScratch ?: Mat().also { ycrcbScratch = it }
        val channels = ArrayList<Mat>(3)
        try {
            Imgproc.cvtColor(bgr, ycrcb, Imgproc.COLOR_BGR2YCrCb)
            Core.split(ycrcb, channels)
            val yChan = channels[0]

            // Subtle black-point dehaze on luma
            if (params.dehazingStrength > 0.03f) {
                val minVal = Core.minMaxLoc(yChan).minVal
                val blackLift = (minVal * params.dehazingStrength * 0.65).coerceIn(0.0, 20.0)
                if (blackLift > 0.5) {
                    val scale = 255.0 / (255.0 - blackLift)
                    yChan.convertTo(yChan, CvType.CV_8UC1, scale, -blackLift * scale)
                }
            }

            // Local CLAHE contrast
            if (params.claheClipLimit > 0.25) {
                val clahe = Imgproc.createCLAHE(params.claheClipLimit, Size(8.0, 8.0))
                val claheOut = claheOutScratch ?: Mat().also { claheOutScratch = it }
                clahe.apply(yChan, claheOut)
                Core.addWeighted(claheOut, 0.40, yChan, 0.60, 0.0, yChan)
            }

            // Combined LUT: gamma + contrast + brightness + highlight compression
            val lut = getOrCreateToneLut(params)
            Core.LUT(yChan, lut, yChan)

            // Flicker reduction inside the exact same Y channel
            if (params.flickerReductionAlpha > 0.01f) {
                val currentMean = Core.mean(yChan).`val`[0]
                val prev = prevLumaMean
                if (prev == null) {
                    prevLumaMean = currentMean
                } else {
                    val targetMean = prev * params.flickerReductionAlpha + currentMean * (1.0 - params.flickerReductionAlpha)
                    val delta = (targetMean - currentMean).coerceIn(-8.0, 8.0)
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
        val key = "${params.gamma}|${params.contrastGain}|${params.brightnessShift}|${params.exposureHighlightCompress}"
        val existing = toneLutMat
        if (existing != null && cachedLutKey == key) {
            return existing
        }
        val lut = existing ?: Mat(1, 256, CvType.CV_8UC1).also { toneLutMat = it }
        val lutData = ByteArray(256)
        val invGamma = params.gamma.toDouble()
        for (i in 0..255) {
            var v = (i / 255.0).pow(invGamma) * 255.0
            v = ((v - 128.0) * params.contrastGain) + 128.0 + params.brightnessShift
            if (params.exposureHighlightCompress > 0.01f && v > 195.0) {
                val excess = v - 195.0
                v = 195.0 + excess * (1.0 - params.exposureHighlightCompress)
            }
            lutData[i] = v.roundToInt().coerceIn(0, 255).toByte()
        }
        lut.put(0, 0, lutData)
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

    private fun applySaturationBoost(bgr: Mat, satGain: Float) {
        val hsv = hsvScratch ?: Mat().also { hsvScratch = it }
        val channels = ArrayList<Mat>(3)
        try {
            Imgproc.cvtColor(bgr, hsv, Imgproc.COLOR_BGR2HSV)
            Core.split(hsv, channels)
            channels[1].convertTo(channels[1], CvType.CV_8UC1, satGain.toDouble(), 0.0)
            Core.merge(channels, hsv)
            Imgproc.cvtColor(hsv, bgr, Imgproc.COLOR_HSV2BGR)
        } finally {
            channels.forEach { it.release() }
        }
    }

    private fun applyFastUnsharpMask(bgr: Mat, sharpnessAmount: Float, edgeEnhanceAmount: Float) {
        val totalAmount = (sharpnessAmount + edgeEnhanceAmount * 0.50f).coerceIn(0.05f, 0.85f)
        val blurred = blurScratch ?: Mat().also { blurScratch = it }
        Imgproc.GaussianBlur(bgr, blurred, Size(0.0, 0.0), 1.1)
        Core.addWeighted(bgr, 1.0 + totalAmount, blurred, -totalAmount.toDouble(), 0.0, bgr)
    }
}
