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
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.FaceDetectorYN
import java.io.File
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.roundToInt

data class FrameProcessResult(
    val facesEnhanced: Int = 0,
    val usedAiDenoise: Boolean = false,
    val dncnnInferenceMs: Long = 0L,
    val faceAiMs: Long = 0L,
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
    private var faceDetector: FaceDetectorYN? = null
    private var faceDetInputW: Int = -1
    private var faceDetInputH: Int = -1
    private var prevFaceBoxes: List<Rect> = emptyList()

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
    private var faceDetScaledMat: Mat? = null
    private var faceDetOutMat: Mat? = null
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
        prevFaceBoxes = emptyList()
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
        faceDetScaledMat?.release(); faceDetScaledMat = null
        faceDetOutMat?.release(); faceDetOutMat = null
        faceDetector = null
        faceDetInputW = -1
        faceDetInputH = -1
        toneLutMat?.release(); toneLutMat = null
        cachedLutKey = ""
        satLutMat?.release(); satLutMat = null
        halfSatLutMat?.release(); halfSatLutMat = null
        cachedSatKey = ""
    }

    /**
     * Genuine multi-stage AI enhancement pipeline per frame:
     * 1. Gentle temporal motion-masked grain suppression (zero spatial blur)
     * 2. DnCNN neural luminance denoising ([1,1,256,256] ONNX) + chroma deblocking + restrained tone LUT
     * 3. YuNet face detection with temporal bbox stabilization
     * 4. Real-ESRGAN x4v3 aspect-preserving tiled super-resolution + dedicated Face ROI AI SR -> [outDims.width x outDims.height]
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
        var facesEnhanced = 0
        var usedAiDenoise = false
        var dncnnMs = 0L
        var faceAiMs = 0L

        try {
            // 1. Temporal Motion-Stabilized Noise Reduction (never blurs pixels spatially before SR)
            applyTemporalMotionStabilizedDenoise(workMat, params.denoiseStrength)

            // 2. Single Fused Luma Pass: DnCNN Neural Denoise + Chroma Deblocking + Restrained CLAHE + Tone LUT
            val ycrcbOutcome = applyFusedYCrCbMastering(workMat, params, logger)
            usedAiDenoise = ycrcbOutcome.first
            dncnnMs = ycrcbOutcome.second

            // 3. Neutral Temperature / White Balance & Skin-Safe Saturation
            if (abs(params.wbGainR - 1f) >= 0.01f || abs(params.wbGainB - 1f) >= 0.01f) {
                applyWhiteBalance(workMat, params)
            }
            if (abs(params.saturationGain - 1f) > 0.01f) {
                applySaturationBoost(workMat, params.saturationGain, params.skinToneProtection)
            }

            // 4. YuNet Neural Face Detection in cleaned input space
            val rawFaces = detectFacesWithYuNet(workMat, logger)
            val stabilizedFaces = smoothFaceBoxes(prevFaceBoxes, rawFaces, workMat.cols(), workMat.rows())
            prevFaceBoxes = stabilizedFaces
            facesEnhanced = stabilizedFaces.size

            // 5. Real-ESRGAN x4v3 Tiled Super-Resolution + Dedicated Face ROI AI Enhancement
            if (outDims.needsAiSuperResolution && outDims.tilesPerFrame > 0 && outDims.aiTileSize > 0) {
                val tAi0 = System.currentTimeMillis()
                val activeTileSize = tileSizeOverride ?: outDims.aiTileSize
                val outcome = onnxEngine.enhanceBgrWithPresetTiles(
                    srcBgr = workMat,
                    outW = outDims.width,
                    outH = outDims.height,
                    tileSize = activeTileSize,
                    targetTilesPerFrame = outDims.tilesPerFrame,
                    estimatedTotalFrames = estimatedTotalFrames,
                    forceLightweightFallback = forceLightweightFallback,
                    aiStrength = params.aiStrength,
                    fallbackSharpnessBoost = params.sharpnessAmount,
                    faces = stabilizedFaces,
                    faceEnhanceStrength = params.faceEnhanceStrength,
                    isCancelled = isCancelled,
                    logger = logger,
                    onTileProgress = onUpscaleTileProgress
                )
                logger?.aiInferenceMs?.addAndGet(System.currentTimeMillis() - tAi0)
                faceAiMs = outcome.faceAiMs
                usedFallback = outcome.usedLightweightFallback
                fallbackReason = outcome.fallbackReason
                accelLabel = outcome.activeAccelerationLabel
                workMat.release()
                workMat = outcome.outputBgr
            } else {
                if (workMat.cols() != outDims.width || workMat.rows() != outDims.height) {
                    val aligned = alignedScratch ?: Mat().also { alignedScratch = it }
                    Imgproc.resize(
                        workMat,
                        aligned,
                        Size(outDims.width.toDouble(), outDims.height.toDouble()),
                        0.0,
                        0.0,
                        Imgproc.INTER_CUBIC
                    )
                    workMat.release()
                    workMat = aligned.clone()
                }
                if (onnxEngine.isGlesGpuReady && (params.sharpnessAmount > 0.03f || params.edgeEnhancementAmount > 0.03f)) {
                    val sharpness = (params.sharpnessAmount + params.edgeEnhancementAmount * 0.35f).coerceIn(0.05f, 0.45f)
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
            }

            // Ensure exact target output resolution
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
            facesEnhanced = facesEnhanced,
            usedAiDenoise = usedAiDenoise,
            dncnnInferenceMs = dncnnMs,
            faceAiMs = faceAiMs,
            lightweightFallbackTriggered = usedFallback,
            fallbackReason = fallbackReason,
            activeAccelerationLabel = accelLabel
        )
    }

    /**
     * Detects up to 3 primary human faces via bundled YuNet ONNX detector (face_detection_yunet_2023mar.onnx),
     * running on a <= 480px copy for speed and mapping exact bounding boxes back to full frame coordinates.
     */
    private fun detectFacesWithYuNet(srcBgr: Mat, logger: EngineDiagnosticsLogger?): List<Rect> {
        val modelPath = faceDetectorPath ?: return emptyList()
        val srcW = srcBgr.cols()
        val srcH = srcBgr.rows()
        if (srcW < 32 || srcH < 32) return emptyList()

        return try {
            val maxDetSide = 480.0
            val scale = if (max(srcW, srcH) > maxDetSide) maxDetSide / max(srcW, srcH).toDouble() else 1.0
            val detW = (srcW * scale).roundToInt().coerceAtLeast(32)
            val detH = (srcH * scale).roundToInt().coerceAtLeast(32)

            val detInput = if (scale < 0.99) {
                val scaled = faceDetScaledMat ?: Mat().also { faceDetScaledMat = it }
                Imgproc.resize(srcBgr, scaled, Size(detW.toDouble(), detH.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
                scaled
            } else {
                srcBgr
            }

            val detector = faceDetector ?: FaceDetectorYN.create(
                modelPath,
                "",
                Size(detW.toDouble(), detH.toDouble()),
                0.56f,
                0.30f,
                10
            ).also {
                faceDetector = it
                faceDetInputW = detW
                faceDetInputH = detH
            }

            if (faceDetInputW != detW || faceDetInputH != detH) {
                detector.inputSize = Size(detW.toDouble(), detH.toDouble())
                faceDetInputW = detW
                faceDetInputH = detH
            }

            val outFaces = faceDetOutMat ?: Mat().also { faceDetOutMat = it }
            detector.detect(detInput, outFaces)

            val rows = outFaces.rows()
            if (rows <= 0) return emptyList()

            val invScaleX = srcW.toDouble() / detW.toDouble()
            val invScaleY = srcH.toDouble() / detH.toDouble()
            val rowBuf = FloatArray(outFaces.cols().coerceAtLeast(15))
            val detected = ArrayList<Pair<Rect, Float>>(rows)

            for (r in 0 until rows) {
                outFaces.get(r, 0, rowBuf)
                val score = rowBuf[14]
                if (score < 0.56f) continue
                val x = (rowBuf[0] * invScaleX).roundToInt().coerceIn(0, srcW - 2)
                val y = (rowBuf[1] * invScaleY).roundToInt().coerceIn(0, srcH - 2)
                val w = (rowBuf[2] * invScaleX).roundToInt().coerceIn(8, srcW - x)
                val h = (rowBuf[3] * invScaleY).roundToInt().coerceIn(8, srcH - y)
                if (w >= 16 && h >= 16) {
                    detected.add(Rect(x, y, w, h) to (w * h).toFloat())
                }
            }

            detected.sortedByDescending { it.second }.take(3).map { it.first }
        } catch (t: Throwable) {
            logger?.warn("YuNet face detection skipped (${t.message})")
            emptyList()
        }
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
        Imgproc.threshold(grayDiff, staticMask, 14.0, 255.0, Imgproc.THRESH_BINARY_INV)
        val alpha = (0.06 + strength * 0.12).coerceIn(0.06, 0.18)
        Core.addWeighted(bgr, 1.0 - alpha, prev, alpha, 0.0, blended)
        blended.copyTo(bgr, staticMask)
        bgr.copyTo(prev)
    }

    /**
     * Single-pass YCrCb luminance mastering:
     * - Executes real DnCNN neural denoiser on Y channel (preserves edges/textures while cleaning noise)
     * - Applies light chroma median cleanup only for high compression blockiness
     * - Restrained tone LUT + ultra-gentle CLAHE (<= 0.16) + flicker damping
     */
    private fun applyFusedYCrCbMastering(
        bgr: Mat,
        params: AdaptiveAdjustmentParams,
        logger: EngineDiagnosticsLogger?
    ): Pair<Boolean, Long> {
        val ycrcb = ycrcbScratch ?: Mat().also { ycrcbScratch = it }
        val channels = ArrayList<Mat>(3)
        var usedDnCnn = false
        var dncnnMs = 0L
        try {
            Imgproc.cvtColor(bgr, ycrcb, Imgproc.COLOR_BGR2YCrCb)
            Core.split(ycrcb, channels)
            val yChan = channels[0]

            // Real DnCNN Neural Luminance Denoising (removes noise/grain cleanly without edge blur)
            if (params.useAiDenoise && onnxEngine.isDncnnReady) {
                val td0 = System.currentTimeMillis()
                usedDnCnn = onnxEngine.denoiseLuminanceYWithDnCnn(yChan, params.dncnnStrength, logger)
                dncnnMs = System.currentTimeMillis() - td0
            }

            // Gentle chroma-only median cleanup for color compression blocks without touching Y sharpness
            if (params.deblockingStrength > 0.28f) {
                Imgproc.medianBlur(channels[1], channels[1], 3)
                Imgproc.medianBlur(channels[2], channels[2], 3)
            }

            // Subtle brilliance black-point lift only when hazy
            if (params.dehazingStrength > 0.02f) {
                val minVal = Core.minMaxLoc(yChan).minVal
                val blackLift = (minVal * params.dehazingStrength * 0.40).coerceIn(0.0, 8.0)
                if (blackLift > 0.5) {
                    val scale = 255.0 / (255.0 - blackLift)
                    yChan.convertTo(yChan, CvType.CV_8UC1, scale, -blackLift * scale)
                }
            }

            // Restrained local clarity via CLAHE (capped at 0.16 weight so video never looks artificially HDR)
            if (params.claheClipLimit > 0.22) {
                val clahe = Imgproc.createCLAHE(params.claheClipLimit, Size(8.0, 8.0))
                val claheOut = claheOutScratch ?: Mat().also { claheOutScratch = it }
                clahe.apply(yChan, claheOut)
                Core.addWeighted(claheOut, 0.16, yChan, 0.84, 0.0, yChan)
            }

            val lut = getOrCreateToneLut(params)
            Core.LUT(yChan, lut, yChan)

            if (params.flickerReductionAlpha > 0.01f) {
                val currentMean = Core.mean(yChan).`val`[0]
                val prev = prevLumaMean
                if (prev == null) {
                    prevLumaMean = currentMean
                } else {
                    val targetMean = prev * params.flickerReductionAlpha + currentMean * (1.0 - params.flickerReductionAlpha)
                    val delta = (targetMean - currentMean).coerceIn(-5.0, 5.0)
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
        return usedDnCnn to dncnnMs
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
        val totalAmount = (sharpnessAmount + edgeEnhanceAmount * 0.35f).coerceIn(0.05f, 0.42f)
        val blurred = blurScratch ?: Mat().also { blurScratch = it }
        Imgproc.GaussianBlur(bgr, blurred, Size(0.0, 0.0), 1.0)
        Core.addWeighted(bgr, 1.0 + totalAmount, blurred, -totalAmount.toDouble(), 0.0, bgr)
    }

    companion object {
        /**
         * Temporally stabilizes detected face bounding boxes against nearest boxes in the
         * previous frame using IoU matching and exponential smoothing to prevent shimmer.
         */
        fun smoothFaceBoxes(
            prev: List<Rect>,
            curr: List<Rect>,
            frameW: Int,
            frameH: Int
        ): List<Rect> {
            if (prev.isEmpty() || curr.isEmpty()) return curr
            val alpha = 0.65f // new box weight
            return curr.map { box ->
                var bestMatch: Rect? = null
                var bestIou = 0.25f
                for (p in prev) {
                    val iou = rectIou(p, box)
                    if (iou > bestIou) {
                        bestIou = iou
                        bestMatch = p
                    }
                }
                if (bestMatch == null) {
                    box
                } else {
                    val sx = (bestMatch.x * (1f - alpha) + box.x * alpha).roundToInt().coerceIn(0, frameW - 2)
                    val sy = (bestMatch.y * (1f - alpha) + box.y * alpha).roundToInt().coerceIn(0, frameH - 2)
                    val sw = (bestMatch.width * (1f - alpha) + box.width * alpha).roundToInt().coerceIn(8, frameW - sx)
                    val sh = (bestMatch.height * (1f - alpha) + box.height * alpha).roundToInt().coerceIn(8, frameH - sy)
                    Rect(sx, sy, sw, sh)
                }
            }
        }

        private fun rectIou(a: Rect, b: Rect): Float {
            val ix0 = max(a.x, b.x)
            val iy0 = max(a.y, b.y)
            val ix1 = min(a.x + a.width, b.x + b.width)
            val iy1 = min(a.y + a.height, b.y + b.height)
            val iw = (ix1 - ix0).coerceAtLeast(0)
            val ih = (iy1 - iy0).coerceAtLeast(0)
            val inter = (iw * ih).toFloat()
            if (inter <= 0f) return 0f
            val union = (a.width * a.height + b.width * b.height - inter).coerceAtLeast(1f)
            return inter / union
        }

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

                if (blacksAmt > 0.001 && i < 64) {
                    val weight = ((64.0 - i) / 64.0).pow(1.6)
                    v -= weight * blacksAmt * 28.0
                }

                if (shadowsAmt > 0.001) {
                    val dist = (i - 64.0) / 36.0
                    val weight = exp(-0.5 * dist * dist) * (i / 24.0).coerceAtMost(1.0)
                    v += weight * shadowsAmt * 95.0
                }

                v = ((v - 128.0) * params.contrastGain.toDouble()) + 128.0 + params.brightnessShift.toDouble()

                if (whitesAmt > 0.001) {
                    val dist = (i - 192.0) / 30.0
                    val taper = ((255.0 - i) / 28.0).coerceIn(0.0, 1.0)
                    val weight = exp(-0.5 * dist * dist) * taper
                    v += weight * whitesAmt * 75.0
                }

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
