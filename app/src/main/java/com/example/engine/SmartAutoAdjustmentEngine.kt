package com.example.engine

import com.example.model.AdaptiveAdjustmentParams
import com.example.model.EnhancementProfile
import com.example.model.VideoQualityAnalysis
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

class SmartAutoAdjustmentEngine {

    /**
     * Analyzes sampled BGR frames using OpenCV to measure brightness, exposure clipping,
     * contrast, Immerkær noise sigma, Laplacian blur variance, HSV saturation,
     * gray-world white balance shift, 8x8 DCT blockiness, and Sobel edge density.
     */
    fun analyzeSampleFiles(frameFiles: List<File>): VideoQualityAnalysis {
        if (frameFiles.isEmpty()) {
            return defaultBalancedAnalysis()
        }

        var sumBrightness = 0.0
        var sumDarkRatio = 0.0
        var sumOverRatio = 0.0
        var sumContrastStd = 0.0
        var sumDynamicRange = 0.0
        var sumNoise = 0.0
        var sumBlur = 0.0
        var sumSaturation = 0.0
        var sumWbR = 0.0
        var sumWbB = 0.0
        var sumBlockiness = 0.0
        var sumEdgeDensity = 0.0
        var validCount = 0

        for (file in frameFiles) {
            val bgr = Imgcodecs.imread(file.absolutePath)
            if (bgr.empty()) {
                bgr.release()
                continue
            }
            val analysisMat = Mat()
            try {
                // Resize to analysis resolution <= 480px wide for consistent metrics & fast evaluation
                val maxAnalyticW = 480.0
                if (bgr.cols() > maxAnalyticW) {
                    val scale = maxAnalyticW / bgr.cols()
                    Imgproc.resize(bgr, analysisMat, Size(maxAnalyticW, (bgr.rows() * scale).coerceAtLeast(32.0)), 0.0, 0.0, Imgproc.INTER_AREA)
                } else {
                    bgr.copyTo(analysisMat)
                }

                val metrics = analyzeMat(analysisMat)
                sumBrightness += metrics.avgBrightness
                sumDarkRatio += metrics.darkAreaRatio
                sumOverRatio += metrics.overexposedAreaRatio
                sumContrastStd += metrics.contrastStdDev
                sumDynamicRange += metrics.dynamicRange
                sumNoise += metrics.noiseLevel
                sumBlur += metrics.blurVariance
                sumSaturation += metrics.saturationMean
                sumWbR += metrics.whiteBalanceShiftR
                sumWbB += metrics.whiteBalanceShiftB
                sumBlockiness += metrics.compressionArtifactScore
                sumEdgeDensity += metrics.edgeDensity
                validCount++
            } finally {
                analysisMat.release()
                bgr.release()
            }
        }

        if (validCount == 0) {
            return defaultBalancedAnalysis()
        }

        val n = validCount.toFloat()
        val avgB = (sumBrightness / n).toFloat()
        val darkR = (sumDarkRatio / n).toFloat()
        val overR = (sumOverRatio / n).toFloat()
        val contrastStd = (sumContrastStd / n).toFloat()
        val dynRange = (sumDynamicRange / n).toFloat()
        val noise = (sumNoise / n).toFloat()
        val blur = (sumBlur / n).toFloat()
        val sat = (sumSaturation / n).toFloat()
        val wbR = (sumWbR / n).toFloat()
        val wbB = (sumWbB / n).toFloat()
        val blockiness = (sumBlockiness / n).toFloat()
        val edgeDens = (sumEdgeDensity / n).toFloat()

        val tags = buildList {
            if (avgB < 78f || darkR > 0.35f) add("Underexposed / Dark")
            if (avgB > 188f || overR > 0.16f) add("Overexposed / Bright")
            if (contrastStd < 38f) add("Low Contrast")
            if (noise > 8.5f) add("High Noise")
            if (blur < 95f) add("Soft / Blurry")
            if (blur > 420f && noise < 6f) add("Already Sharp")
            if (sat < 52f) add("Faded Colors")
            if (blockiness > 1.24f) add("Compression Artifacts")
            if (isEmpty()) add("Balanced Source")
        }

        return VideoQualityAnalysis(
            avgBrightness = avgB,
            darkAreaRatio = darkR,
            overexposedAreaRatio = overR,
            contrastStdDev = contrastStd,
            dynamicRange = dynRange,
            noiseLevel = noise,
            blurVariance = blur,
            saturationMean = sat,
            whiteBalanceShiftR = wbR,
            whiteBalanceShiftB = wbB,
            compressionArtifactScore = blockiness,
            edgeDensity = edgeDens,
            conditionTags = tags
        )
    }

    /**
     * Computes adaptive adjustment parameters by modulating the target EnhancementProfile
     * based on the measured video quality metrics.
     */
    fun computeAdaptiveAdjustments(
        profile: EnhancementProfile,
        analysis: VideoQualityAnalysis
    ): AdaptiveAdjustmentParams {
        val notes = mutableListOf<String>()

        // Detect if video is already clean & high quality -> apply minimal refinement
        val isHighQualitySource =
            analysis.avgBrightness in 88f..175f &&
                analysis.overexposedAreaRatio < 0.08f &&
                analysis.darkAreaRatio < 0.25f &&
                analysis.noiseLevel < 5.5f &&
                analysis.blurVariance > 260f &&
                analysis.compressionArtifactScore < 1.15f

        val qualityDampener = if (isHighQualitySource) 0.55f else 1.0f
        if (isHighQualitySource) {
            notes.add("High-quality source: subtle refinement")
        }

        // 1. Brightness & Exposure & Gamma
        var brightnessShift = (profile.brightnessPercent / 100f) * 65f * qualityDampener
        var gamma = profile.gamma
        var highlightCompress = 0f

        when {
            analysis.avgBrightness < 80f || analysis.darkAreaRatio > 0.35f -> {
                val darkSeverity = ((85f - analysis.avgBrightness).coerceIn(0f, 55f) / 55f)
                brightnessShift += 6f + darkSeverity * 14f
                gamma = (0.94f - darkSeverity * 0.12f).coerceIn(0.80f, 0.96f) // lifts shadows
                notes.add("Lifted dark shadows")
            }
            analysis.avgBrightness > 188f || analysis.overexposedAreaRatio > 0.15f -> {
                val overSeverity = ((analysis.avgBrightness - 180f).coerceIn(0f, 60f) / 60f)
                brightnessShift = -4f - overSeverity * 12f
                gamma = (1.06f + overSeverity * 0.12f).coerceIn(1.04f, 1.22f)
                highlightCompress = (0.14f + analysis.overexposedAreaRatio * 0.65f).coerceIn(0.10f, 0.40f)
                notes.add("Protected highlights")
            }
        }

        // 2. Contrast & Dehazing
        var contrastGain = 1f + (profile.contrastPercent / 100f) * 0.85f * qualityDampener
        var claheLimit = (0.7 + (profile.contrastPercent / 10f) * 0.45) * qualityDampener
        var dehazing = (profile.dehazingPercent / 100f) * qualityDampener

        if (analysis.contrastStdDev < 38f) {
            contrastGain = (contrastGain + 0.10f).coerceAtMost(1.32f)
            claheLimit = (claheLimit + 0.55).coerceAtMost(2.2)
            dehazing = (dehazing + 0.08f).coerceAtMost(0.32f)
            notes.add("Boosted flat contrast")
        } else if (analysis.contrastStdDev > 72f) {
            contrastGain = (1f + (contrastGain - 1f) * 0.5f)
            claheLimit *= 0.6
        }

        // 3. Denoise
        var denoiseStrength = (profile.denoisePercent / 100f) * 1.4f * qualityDampener
        if (analysis.noiseLevel > 8.5f) {
            val noiseExtra = ((analysis.noiseLevel - 8f) / 14f).coerceIn(0.1f, 0.55f)
            denoiseStrength = (denoiseStrength + noiseExtra).coerceIn(0.15f, 0.92f)
            notes.add("Increased denoising")
        } else if (analysis.noiseLevel < 4.0f) {
            denoiseStrength *= 0.55f
        }
        val useAiDenoise = denoiseStrength >= 0.22f || analysis.noiseLevel >= 7.5f

        // 4. Deblocking
        val baseDeblock = when (profile.deblocking.lowercase()) {
            "high" -> 0.65f
            "medium" -> 0.40f
            "low" -> 0.22f
            else -> 0.25f
        } * qualityDampener
        val deblockingStrength = if (analysis.compressionArtifactScore > 1.20f) {
            val extra = ((analysis.compressionArtifactScore - 1.15f) * 0.85f).coerceIn(0.10f, 0.45f)
            notes.add("Applied deblocking")
            (baseDeblock + extra).coerceIn(0.15f, 0.90f)
        } else {
            baseDeblock
        }

        // 5. Sharpness & Edge Enhancement (reduce if already sharp or very noisy)
        var sharpnessAmount = (profile.sharpnessPercent / 100f) * 1.8f * qualityDampener
        var edgeEnhancement = (profile.edgeEnhancementPercent / 100f) * 1.5f * qualityDampener
        if (analysis.blurVariance > 380f) {
            sharpnessAmount *= 0.45f
            edgeEnhancement *= 0.55f
            notes.add("Reduced sharpening (already sharp)")
        } else if (analysis.blurVariance < 100f && analysis.noiseLevel < 9f) {
            sharpnessAmount = (sharpnessAmount + 0.14f).coerceAtMost(0.75f)
            edgeEnhancement = (edgeEnhancement + 0.10f).coerceAtMost(0.55f)
            notes.add("Restored edge clarity")
        }
        if (analysis.noiseLevel > 11f) {
            // Avoid sharpening film noise
            sharpnessAmount *= 0.6f
        }

        // 6. Saturation & White Balance
        var saturationGain = 1f + (profile.saturationPercent / 100f) * 1.6f * qualityDampener
        if (analysis.saturationMean < 55f) {
            saturationGain = (saturationGain + 0.16f).coerceAtMost(1.38f)
            notes.add("Restored faded colors")
        } else if (analysis.saturationMean > 145f) {
            saturationGain = 1f + (saturationGain - 1f) * 0.4f
        }

        // Damped gray-world white balance correction (clamped to prevent unnatural color casts)
        val wbStrength = if (isHighQualitySource) 0.25f else 0.45f
        val wbGainR = (1f + (analysis.whiteBalanceShiftR - 1f) * wbStrength).coerceIn(0.88f, 1.14f)
        val wbGainB = (1f + (analysis.whiteBalanceShiftB - 1f) * wbStrength).coerceIn(0.88f, 1.14f)

        val flickerAlpha = if (profile.flickerReduction.equals("Automatic", ignoreCase = true)) {
            if (isHighQualitySource) 0.10f else 0.18f
        } else 0f

        val faceEnhance = if (isHighQualitySource) 0.22f else 0.36f

        return AdaptiveAdjustmentParams(
            profileId = profile.id,
            brightnessShift = brightnessShift.coerceIn(-22f, 24f),
            exposureHighlightCompress = highlightCompress,
            contrastGain = contrastGain.coerceIn(1.0f, 1.35f),
            claheClipLimit = claheLimit.coerceIn(0.3, 2.2),
            gamma = gamma.coerceIn(0.80f, 1.22f),
            saturationGain = saturationGain.coerceIn(0.95f, 1.40f),
            wbGainR = wbGainR,
            wbGainG = 1.0f,
            wbGainB = wbGainB,
            denoiseStrength = denoiseStrength.coerceIn(0.05f, 0.92f),
            useAiDenoise = useAiDenoise,
            deblockingStrength = deblockingStrength.coerceIn(0.05f, 0.90f),
            dehazingStrength = dehazing.coerceIn(0f, 0.32f),
            sharpnessAmount = sharpnessAmount.coerceIn(0.05f, 0.78f),
            edgeEnhancementAmount = edgeEnhancement.coerceIn(0.05f, 0.55f),
            flickerReductionAlpha = flickerAlpha,
            faceEnhanceStrength = faceEnhance,
            adjustmentSummary = if (notes.isEmpty()) "Balanced adaptive mastering" else notes.joinToString(" • ")
        )
    }

    private fun analyzeMat(bgr: Mat): VideoQualityAnalysis {
        val gray = Mat()
        val hsv = Mat()
        val lap = Mat()
        val sobelX = Mat()
        val sobelY = Mat()
        val sobelMag = Mat()
        val noiseMat = Mat()
        val kernel = Mat(3, 3, CvType.CV_32F)
        try {
            Imgproc.cvtColor(bgr, gray, Imgproc.COLOR_BGR2GRAY)
            Imgproc.cvtColor(bgr, hsv, Imgproc.COLOR_BGR2HSV)

            val totalPixels = (gray.rows() * gray.cols()).toDouble().coerceAtLeast(1.0)

            // Brightness & Contrast
            val meanMat = MatOfDouble()
            val stdMat = MatOfDouble()
            Core.meanStdDev(gray, meanMat, stdMat)
            val avgBrightness = meanMat.get(0, 0)[0].toFloat()
            val contrastStd = stdMat.get(0, 0)[0].toFloat()
            meanMat.release()
            stdMat.release()

            // Dark (< 24) & Overexposed (> 238) ratios
            val darkMask = Mat()
            val overMask = Mat()
            Imgproc.threshold(gray, darkMask, 24.0, 255.0, Imgproc.THRESH_BINARY_INV)
            Imgproc.threshold(gray, overMask, 238.0, 255.0, Imgproc.THRESH_BINARY)
            val darkRatio = (Core.countNonZero(darkMask) / totalPixels).toFloat()
            val overRatio = (Core.countNonZero(overMask) / totalPixels).toFloat()
            darkMask.release()
            overMask.release()

            // Dynamic range approximation from std
            val dynamicRange = (contrastStd * 3.1f).coerceIn(0f, 255f)

            // Blur via Variance of Laplacian
            Imgproc.Laplacian(gray, lap, CvType.CV_64F)
            val lapMean = MatOfDouble()
            val lapStd = MatOfDouble()
            Core.meanStdDev(lap, lapMean, lapStd)
            val lapSigma = lapStd.get(0, 0)[0]
            val blurVariance = (lapSigma * lapSigma).toFloat()
            lapMean.release()
            lapStd.release()

            // Immerkær fast noise estimation kernel
            kernel.put(
                0, 0,
                1.0, -2.0, 1.0,
                -2.0, 4.0, -2.0,
                1.0, -2.0, 1.0
            )
            Imgproc.filter2D(gray, noiseMat, CvType.CV_32F, kernel)
            Core.absdiff(noiseMat, Scalar(0.0), noiseMat)
            val sumNoise = Core.sumElems(noiseMat).`val`[0]
            val w = max(1, gray.cols() - 2)
            val h = max(1, gray.rows() - 2)
            val noiseSigma = ((sumNoise * sqrt(0.5 * PI)) / (6.0 * w * h)).toFloat()

            // HSV Saturation mean
            val hsvChannels = ArrayList<Mat>(3)
            Core.split(hsv, hsvChannels)
            val satMean = Core.mean(hsvChannels[1]).`val`[0].toFloat()
            hsvChannels.forEach { it.release() }

            // Gray-world white balance ratio
            val bgrMean = Core.mean(bgr).`val`
            val meanB = bgrMean[0].coerceAtLeast(1.0)
            val meanG = bgrMean[1].coerceAtLeast(1.0)
            val meanR = bgrMean[2].coerceAtLeast(1.0)
            val wbShiftR = (meanG / meanR).toFloat().coerceIn(0.6f, 1.5f)
            val wbShiftB = (meanG / meanB).toFloat().coerceIn(0.6f, 1.5f)

            // Sobel edge density
            Imgproc.Sobel(gray, sobelX, CvType.CV_32F, 1, 0, 3)
            Imgproc.Sobel(gray, sobelY, CvType.CV_32F, 0, 1, 3)
            Core.magnitude(sobelX, sobelY, sobelMag)
            val edgeMask = Mat()
            Imgproc.threshold(sobelMag, edgeMask, 45.0, 255.0, Imgproc.THRESH_BINARY)
            val edgeDensity = (Core.countNonZero(edgeMask) / totalPixels).toFloat()
            edgeMask.release()

            // 8x8 macroblock boundary discontinuity score
            val blockiness = measureBlockiness(gray)

            return VideoQualityAnalysis(
                avgBrightness = avgBrightness,
                darkAreaRatio = darkRatio,
                overexposedAreaRatio = overRatio,
                contrastStdDev = contrastStd,
                dynamicRange = dynamicRange,
                noiseLevel = noiseSigma,
                blurVariance = blurVariance,
                saturationMean = satMean,
                whiteBalanceShiftR = wbShiftR,
                whiteBalanceShiftB = wbShiftB,
                compressionArtifactScore = blockiness,
                edgeDensity = edgeDensity,
                conditionTags = emptyList()
            )
        } finally {
            gray.release()
            hsv.release()
            lap.release()
            sobelX.release()
            sobelY.release()
            sobelMag.release()
            noiseMat.release()
            kernel.release()
        }
    }

    private fun measureBlockiness(gray: Mat): Float {
        val rows = gray.rows()
        val cols = gray.cols()
        if (rows < 24 || cols < 24) return 1.0f
        val bytes = ByteArray(rows * cols)
        gray.get(0, 0, bytes)

        var boundarySum = 0.0
        var boundaryCount = 0
        var interiorSum = 0.0
        var interiorCount = 0

        for (r in 1 until rows - 1 step 2) {
            val rowOffset = r * cols
            for (c in 1 until cols - 1) {
                val diff = abs((bytes[rowOffset + c].toInt() and 0xFF) - (bytes[rowOffset + c - 1].toInt() and 0xFF))
                if (c % 8 == 0) {
                    boundarySum += diff
                    boundaryCount++
                } else {
                    interiorSum += diff
                    interiorCount++
                }
            }
        }
        val bAvg = if (boundaryCount > 0) boundarySum / boundaryCount else 0.0
        val iAvg = if (interiorCount > 0) interiorSum / interiorCount else 1.0
        return (bAvg / max(iAvg, 0.5)).toFloat().coerceIn(0.8f, 3.0f)
    }

    private fun defaultBalancedAnalysis() = VideoQualityAnalysis(
        avgBrightness = 120f,
        darkAreaRatio = 0.1f,
        overexposedAreaRatio = 0.02f,
        contrastStdDev = 50f,
        dynamicRange = 155f,
        noiseLevel = 4.5f,
        blurVariance = 220f,
        saturationMean = 95f,
        whiteBalanceShiftR = 1.0f,
        whiteBalanceShiftB = 1.0f,
        compressionArtifactScore = 1.05f,
        edgeDensity = 0.18f,
        conditionTags = listOf("Balanced Source")
    )
}
