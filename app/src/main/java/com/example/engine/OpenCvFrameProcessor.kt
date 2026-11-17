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
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.roundToInt

class OpenCvFrameProcessor(
    private val context: Context,
    private val onnxEngine: OnnxAiInferenceEngine
) {
    private var openCvInitialized = false
    private var faceDetectorPath: String? = null
    private var faceDetector: FaceDetectorYN? = null
    private var prevDenoisedFrame: Mat? = null
    private var prevLumaMean: Double? = null
    private var reusableRgbaMat: Mat? = null

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
        prevDenoisedFrame?.release()
        prevDenoisedFrame = null
        prevLumaMean = null
        reusableRgbaMat?.release()
        reusableRgbaMat = null
    }

    /**
     * Executes the complete ordered per-frame enhancement pipeline on [srcBgr] and writes
     * the result into [reusableOutBitmap] of dimensions [outDims.width] x [outDims.height].
     * Returns number of faces enhanced in this frame.
     */
    fun processFrame(
        srcBgr: Mat,
        outDims: OutputDimensions,
        params: AdaptiveAdjustmentParams,
        capabilities: DeviceCapabilities,
        reusableOutBitmap: Bitmap,
        tileSizeOverride: Int? = null,
        logger: EngineDiagnosticsLogger? = null,
        isCancelled: () -> Boolean,
        onUpscaleTileProgress: (done: Int, total: Int, tileSize: Int) -> Unit = { _, _, _ -> }
    ): Int {
        var workMat = srcBgr.clone()
        val facesEnhanced = 0
        try {
            // 1. Temporal Motion-Stabilized Denoise + Lightweight Spatial Denoise (fast path)
            applyTemporalMotionStabilizedDenoise(workMat, params.denoiseStrength)
            if (params.deblockingStrength > 0.35f) {
                applyAdaptiveDeblocking(workMat, params.deblockingStrength)
            } else if (params.denoiseStrength > 0.25f) {
                val filtered = Mat()
                Imgproc.GaussianBlur(workMat, filtered, Size(3.0, 3.0), 0.75)
                Core.addWeighted(filtered, 0.30, workMat, 0.70, 0.0, workMat)
                filtered.release()
            }

            // 2. Color Correction (Dehazing + Brightness/Contrast/Gamma + White Balance + Saturation + Flicker)
            if (params.dehazingStrength > 0.03f) {
                applySubtleDehaze(workMat, params.dehazingStrength)
            }
            applyToneContrastAndGamma(workMat, params)
            applyWhiteBalance(workMat, params)
            if (abs(params.saturationGain - 1f) > 0.015f) {
                applySaturationBoost(workMat, params.saturationGain)
            }
            if (params.flickerReductionAlpha > 0.01f) {
                applyFlickerReduction(workMat, params.flickerReductionAlpha)
            }

            // 3. Preset AI Tile Enhancement (HD: 0 tiles, HD+: 512px x 6, 2K: 384px x 12, 4K: 256px x 25)
            // Original video resolution is kept unchanged.
            if (outDims.needsAiSuperResolution && outDims.tilesPerFrame > 0 && outDims.aiTileSize > 0 && onnxEngine.isRealEsrganReady) {
                val tAi0 = System.currentTimeMillis()
                val activeTileSize = tileSizeOverride ?: outDims.aiTileSize
                val enhanced = onnxEngine.enhanceBgrWithPresetTiles(
                    srcBgr = workMat,
                    tileSize = activeTileSize,
                    targetTilesPerFrame = outDims.tilesPerFrame,
                    isCancelled = isCancelled,
                    logger = logger,
                    onTileProgress = onUpscaleTileProgress
                )
                logger?.aiInferenceMs?.addAndGet(System.currentTimeMillis() - tAi0)
                workMat.release()
                workMat = enhanced
            }

            // 4. Controlled Edge-Masked Sharpening & Edge Enhancement
            if (params.sharpnessAmount > 0.03f || params.edgeEnhancementAmount > 0.03f) {
                applyEdgeMaskedSharpening(workMat, params.sharpnessAmount, params.edgeEnhancementAmount)
            }

            // Ensure output matches exact original even dimensions (no cropping/scaling surprises)
            if (workMat.cols() != outDims.width || workMat.rows() != outDims.height) {
                val aligned = Mat()
                Imgproc.resize(workMat, aligned, Size(outDims.width.toDouble(), outDims.height.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
                workMat.release()
                workMat = aligned
            }

            // Reuse single RGBA Mat for Bitmap transfer
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
        return facesEnhanced
    }

    private fun applyAdaptiveDeblocking(bgr: Mat, strength: Float) {
        val w = bgr.cols()
        val h = bgr.rows()
        if (w < 16 || h < 16) return
        val smooth = Mat()
        Imgproc.bilateralFilter(bgr, smooth, 5, 18.0 + strength * 22.0, 6.0)
        val origBytes = ByteArray(w * h * 3)
        val smoothBytes = ByteArray(w * h * 3)
        bgr.get(0, 0, origBytes)
        smooth.get(0, 0, smoothBytes)
        smooth.release()

        val alpha = (strength * 0.65f).coerceIn(0.1f, 0.65f)
        val invAlpha = 1f - alpha
        for (y in 1 until h - 1) {
            val isRowBorder = (y % 8 == 0 || y % 8 == 7)
            val rowOff = y * w
            if (isRowBorder) {
                for (x in 1 until w - 1) {
                    val idx = (rowOff + x) * 3
                    origBytes[idx] = ((origBytes[idx].toInt() and 0xFF) * invAlpha + (smoothBytes[idx].toInt() and 0xFF) * alpha).roundToInt().coerceIn(0, 255).toByte()
                    origBytes[idx + 1] = ((origBytes[idx + 1].toInt() and 0xFF) * invAlpha + (smoothBytes[idx + 1].toInt() and 0xFF) * alpha).roundToInt().coerceIn(0, 255).toByte()
                    origBytes[idx + 2] = ((origBytes[idx + 2].toInt() and 0xFF) * invAlpha + (smoothBytes[idx + 2].toInt() and 0xFF) * alpha).roundToInt().coerceIn(0, 255).toByte()
                }
            } else {
                var x = 7
                while (x < w - 1) {
                    for (bx in x..(x + 1)) {
                        if (bx in 1 until (w - 1)) {
                            val idx = (rowOff + bx) * 3
                            origBytes[idx] = ((origBytes[idx].toInt() and 0xFF) * invAlpha + (smoothBytes[idx].toInt() and 0xFF) * alpha).roundToInt().coerceIn(0, 255).toByte()
                            origBytes[idx + 1] = ((origBytes[idx + 1].toInt() and 0xFF) * invAlpha + (smoothBytes[idx + 1].toInt() and 0xFF) * alpha).roundToInt().coerceIn(0, 255).toByte()
                            origBytes[idx + 2] = ((origBytes[idx + 2].toInt() and 0xFF) * invAlpha + (smoothBytes[idx + 2].toInt() and 0xFF) * alpha).roundToInt().coerceIn(0, 255).toByte()
                        }
                    }
                    x += 8
                }
            }
        }
        bgr.put(0, 0, origBytes)
    }

    private fun applyTemporalMotionStabilizedDenoise(bgr: Mat, strength: Float) {
        val prev = prevDenoisedFrame
        if (prev == null || prev.cols() != bgr.cols() || prev.rows() != bgr.rows()) {
            prevDenoisedFrame?.release()
            prevDenoisedFrame = bgr.clone()
            return
        }

        val diff = Mat()
        val grayDiff = Mat()
        val staticMask = Mat()
        try {
            Core.absdiff(bgr, prev, diff)
            Imgproc.cvtColor(diff, grayDiff, Imgproc.COLOR_BGR2GRAY)
            Imgproc.threshold(grayDiff, staticMask, 16.0, 255.0, Imgproc.THRESH_BINARY_INV)
            val alpha = (0.10 + strength * 0.16).coerceIn(0.08, 0.26)
            val blended = Mat()
            Core.addWeighted(bgr, 1.0 - alpha, prev, alpha, 0.0, blended)
            blended.copyTo(bgr, staticMask)
            blended.release()

            prevDenoisedFrame?.release()
            prevDenoisedFrame = bgr.clone()
        } finally {
            diff.release()
            grayDiff.release()
            staticMask.release()
        }
    }

    private fun applySubtleDehaze(bgr: Mat, strength: Float) {
        val ycrcb = Mat()
        val channels = ArrayList<Mat>(3)
        try {
            Imgproc.cvtColor(bgr, ycrcb, Imgproc.COLOR_BGR2YCrCb)
            Core.split(ycrcb, channels)
            val minVal = Core.minMaxLoc(channels[0]).minVal
            val blackLift = (minVal * strength * 0.65).coerceIn(0.0, 20.0)
            if (blackLift > 0.5) {
                val scale = 255.0 / (255.0 - blackLift)
                channels[0].convertTo(channels[0], CvType.CV_8UC1, scale, -blackLift * scale)
                Core.merge(channels, ycrcb)
                Imgproc.cvtColor(ycrcb, bgr, Imgproc.COLOR_YCrCb2BGR)
            }
        } finally {
            channels.forEach { it.release() }
            ycrcb.release()
        }
    }

    private fun applyToneContrastAndGamma(bgr: Mat, params: AdaptiveAdjustmentParams) {
        val ycrcb = Mat()
        val channels = ArrayList<Mat>(3)
        val lut = Mat(1, 256, CvType.CV_8UC1)
        try {
            Imgproc.cvtColor(bgr, ycrcb, Imgproc.COLOR_BGR2YCrCb)
            Core.split(ycrcb, channels)

            if (params.claheClipLimit > 0.25) {
                val clahe = Imgproc.createCLAHE(params.claheClipLimit, Size(8.0, 8.0))
                val claheOut = Mat()
                clahe.apply(channels[0], claheOut)
                Core.addWeighted(claheOut, 0.45, channels[0], 0.55, 0.0, channels[0])
                claheOut.release()
            }

            // Combined LUT for gamma + brightness + contrast + highlight protection
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
            Core.LUT(channels[0], lut, channels[0])

            Core.merge(channels, ycrcb)
            Imgproc.cvtColor(ycrcb, bgr, Imgproc.COLOR_YCrCb2BGR)
        } finally {
            channels.forEach { it.release() }
            ycrcb.release()
            lut.release()
        }
    }

    private fun applyWhiteBalance(bgr: Mat, params: AdaptiveAdjustmentParams) {
        if (abs(params.wbGainR - 1f) < 0.01f && abs(params.wbGainB - 1f) < 0.01f) return
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
        val hsv = Mat()
        val channels = ArrayList<Mat>(3)
        try {
            Imgproc.cvtColor(bgr, hsv, Imgproc.COLOR_BGR2HSV)
            Core.split(hsv, channels)
            channels[1].convertTo(channels[1], CvType.CV_8UC1, satGain.toDouble(), 0.0)
            Core.merge(channels, hsv)
            Imgproc.cvtColor(hsv, bgr, Imgproc.COLOR_HSV2BGR)
        } finally {
            channels.forEach { it.release() }
            hsv.release()
        }
    }

    private fun applyFlickerReduction(bgr: Mat, alpha: Float) {
        val ycrcb = Mat()
        val channels = ArrayList<Mat>(3)
        try {
            Imgproc.cvtColor(bgr, ycrcb, Imgproc.COLOR_BGR2YCrCb)
            Core.split(ycrcb, channels)
            val currentMean = Core.mean(channels[0]).`val`[0]
            val prev = prevLumaMean
            if (prev == null) {
                prevLumaMean = currentMean
            } else {
                val targetMean = prev * alpha + currentMean * (1.0 - alpha)
                val delta = (targetMean - currentMean).coerceIn(-8.0, 8.0)
                if (abs(delta) > 0.35) {
                    channels[0].convertTo(channels[0], CvType.CV_8UC1, 1.0, delta)
                    Core.merge(channels, ycrcb)
                    Imgproc.cvtColor(ycrcb, bgr, Imgproc.COLOR_YCrCb2BGR)
                }
                prevLumaMean = targetMean
            }
        } finally {
            channels.forEach { it.release() }
            ycrcb.release()
        }
    }

    private fun applyEdgeMaskedSharpening(
        bgr: Mat,
        sharpnessAmount: Float,
        edgeEnhanceAmount: Float
    ) {
        val gray = Mat()
        val sobelX = Mat()
        val sobelY = Mat()
        val edgeMask = Mat()
        val blurred = Mat()
        val sharpened = Mat()
        try {
            Imgproc.cvtColor(bgr, gray, Imgproc.COLOR_BGR2GRAY)
            Imgproc.Sobel(gray, sobelX, CvType.CV_32F, 1, 0, 3)
            Imgproc.Sobel(gray, sobelY, CvType.CV_32F, 0, 1, 3)
            Core.magnitude(sobelX, sobelY, edgeMask)
            // Soft edge mask [0..1] suppresses flat-sky noise sharpening & prevents halo clipping
            edgeMask.convertTo(edgeMask, CvType.CV_8UC1, 1.4, 0.0)
            Imgproc.GaussianBlur(edgeMask, edgeMask, Size(5.0, 5.0), 1.2)

            val totalAmount = (sharpnessAmount + edgeEnhanceAmount * 0.55f).coerceIn(0.05f, 0.95f)
            Imgproc.GaussianBlur(bgr, blurred, Size(0.0, 0.0), 1.15)
            Core.addWeighted(bgr, 1.0 + totalAmount, blurred, -totalAmount.toDouble(), 0.0, sharpened)

            // Blend sharpened pixels strongly on structural edges, gently elsewhere
            val gentle = Mat()
            Core.addWeighted(bgr, 0.72, sharpened, 0.28, 0.0, gentle)
            sharpened.copyTo(gentle, edgeMask)
            gentle.copyTo(bgr)
            gentle.release()
        } finally {
            gray.release()
            sobelX.release()
            sobelY.release()
            edgeMask.release()
            blurred.release()
            sharpened.release()
        }
    }

    private fun enhanceDetectedFaces(bgr: Mat, strength: Float): Int {
        val path = faceDetectorPath ?: return 0
        val detW = min(640, bgr.cols()).coerceAtLeast(64)
        val scale = detW.toDouble() / bgr.cols().toDouble()
        val detH = (bgr.rows() * scale).roundToInt().coerceAtLeast(64)

        val resizedForDet = Mat()
        val faces = Mat()
        var enhancedCount = 0
        try {
            Imgproc.resize(bgr, resizedForDet, Size(detW.toDouble(), detH.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
            val detector = faceDetector ?: FaceDetectorYN.create(path, "", Size(detW.toDouble(), detH.toDouble()), 0.65f, 0.3f, 20).also {
                faceDetector = it
            }
            detector.setInputSize(Size(detW.toDouble(), detH.toDouble()))
            detector.detect(resizedForDet, faces)

            val numFaces = faces.rows()
            if (numFaces <= 0) return 0

            val invScale = 1.0 / scale
            for (i in 0 until numFaces) {
                val x = (faces.get(i, 0)[0] * invScale).roundToInt().coerceIn(0, bgr.cols() - 2)
                val y = (faces.get(i, 1)[0] * invScale).roundToInt().coerceIn(0, bgr.rows() - 2)
                val fw = (faces.get(i, 2)[0] * invScale).roundToInt().coerceAtLeast(16)
                val fh = (faces.get(i, 3)[0] * invScale).roundToInt().coerceAtLeast(16)

                val padX = (fw * 0.12).roundToInt()
                val padY = (fh * 0.12).roundToInt()
                val rx = max(0, x - padX)
                val ry = max(0, y - padY)
                val rw = min(bgr.cols() - rx, fw + 2 * padX)
                val rh = min(bgr.rows() - ry, fh + 2 * padY)
                if (rw < 24 || rh < 24) continue

                val roi = Mat(bgr, Rect(rx, ry, rw, rh))
                enhanceSingleFaceRoi(roi, strength)
                roi.release()
                enhancedCount++
            }
        } catch (_: Exception) {
            // Never fail frame processing if face detector hits an edge-case frame
        } finally {
            resizedForDet.release()
            faces.release()
        }
        return enhancedCount
    }

    /**
     * Enhances facial clarity (eyes, brows, hair edges) while preserving natural skin texture
     * via high-pass detail extraction + gentle edge-preserving bilateral + elliptical feathered blend.
     */
    private fun enhanceSingleFaceRoi(roi: Mat, strength: Float) {
        val smoothedSkin = Mat()
        val detail = Mat()
        val enhanced = Mat()
        val mask = Mat.zeros(roi.rows(), roi.cols(), CvType.CV_8UC1)
        try {
            // Preserve skin texture: subtle bilateral + recover crisp high-frequency pores/eyes
            Imgproc.bilateralFilter(roi, smoothedSkin, 5, 16.0, 5.0)
            Core.addWeighted(roi, 1.0 + strength * 0.45, smoothedSkin, -strength * 0.45, 0.0, detail)

            // Gentle local contrast on luminance
            val ycrcb = Mat()
            val ch = ArrayList<Mat>(3)
            Imgproc.cvtColor(detail, ycrcb, Imgproc.COLOR_BGR2YCrCb)
            Core.split(ycrcb, ch)
            val clahe = Imgproc.createCLAHE(0.75, Size(4.0, 4.0))
            val claheY = Mat()
            clahe.apply(ch[0], claheY)
            Core.addWeighted(claheY, 0.25, ch[0], 0.75, 0.0, ch[0])
            claheY.release()
            Core.merge(ch, ycrcb)
            ch.forEach { it.release() }
            Imgproc.cvtColor(ycrcb, enhanced, Imgproc.COLOR_YCrCb2BGR)
            ycrcb.release()

            // Elliptical feathered mask so boundaries blend seamlessly with zero visible seams
            Imgproc.ellipse(
                mask,
                org.opencv.core.Point(roi.cols() / 2.0, roi.rows() / 2.0),
                Size(roi.cols() * 0.42, roi.rows() * 0.44),
                0.0, 0.0, 360.0,
                Scalar(255.0),
                -1
            )
            Imgproc.GaussianBlur(mask, mask, Size(15.0, 15.0), 5.0)

            val blended = Mat()
            Core.addWeighted(roi, 1.0 - strength * 0.75, enhanced, strength * 0.75, 0.0, blended)
            blended.copyTo(roi, mask)
            blended.release()
        } finally {
            smoothedSkin.release()
            detail.release()
            enhanced.release()
            mask.release()
        }
    }
}
