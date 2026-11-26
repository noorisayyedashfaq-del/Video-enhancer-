package com.example.engine

import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.os.Build
import com.example.model.AdaptiveAdjustmentParams
import com.example.model.EnhancementProfile
import com.example.model.VideoMetadata
import com.example.model.VideoQualityAnalysis
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.roundToInt

data class AnalysisOutcome(
    val analysis: VideoQualityAnalysis,
    val reusedFromCache: Boolean,
    val sampledFrames: Int,
    val durationMs: Long,
    val warning: String? = null
)

class SmartAutoAdjustmentEngine {

    private val analysisCache = ConcurrentHashMap<String, VideoQualityAnalysis>()

    fun cacheKeyFor(metadata: VideoMetadata): String {
        return "${metadata.uri}|${metadata.fileSizeBytes}|${metadata.durationMs}|${metadata.displayWidth}x${metadata.displayHeight}"
    }

    fun clearCache() {
        analysisCache.clear()
    }

    /**
     * Fast representative frame sampling directly from [videoFile] before full frame extraction.
     * Samples up to 3 evenly-spaced keyframes via MediaMetadataRetriever at <= 320px width,
     * falling back to targeted MediaExtractor + MediaCodec single-frame seeks if needed, with strict timeout recovery.
     */
    suspend fun analyzeVideoRepresentativeFrames(
        videoFile: File,
        metadata: VideoMetadata,
        tempDir: File,
        timeoutMs: Long = 6_000L,
        logger: EngineDiagnosticsLogger? = null
    ): AnalysisOutcome = withContext(Dispatchers.IO) {
        val startMs = System.currentTimeMillis()
        val key = cacheKeyFor(metadata)
        analysisCache[key]?.let { cached ->
            val elapsed = System.currentTimeMillis() - startMs
            logger?.info("Video quality analysis reused from cache in ${elapsed}ms (${cached.conditionTags.joinToString()})")
            return@withContext AnalysisOutcome(
                analysis = cached,
                reusedFromCache = true,
                sampledFrames = 3,
                durationMs = elapsed
            )
        }

        val timedOutcome = withTimeoutOrNull(timeoutMs) {
            var warningMsg: String? = null
            val metricsList = mutableListOf<VideoQualityAnalysis>()

            // Fast sampler: 3 evenly-spaced sync frames decoded directly at 320px
            val retriever = MediaMetadataRetriever()
            try {
                retriever.setDataSource(videoFile.absolutePath)
                val durUs = (metadata.durationMs.coerceAtLeast(300L)) * 1000L
                val fractions = listOf(0.15, 0.50, 0.85)
                for (frac in fractions) {
                    val timeUs = (durUs * frac).toLong().coerceAtLeast(0L)
                    val bmp: Bitmap? = try {
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
                            retriever.getScaledFrameAtTime(
                                timeUs,
                                MediaMetadataRetriever.OPTION_CLOSEST_SYNC,
                                320,
                                320
                            )
                        } else {
                            retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
                        }
                    } catch (_: Throwable) {
                        null
                    }
                    if (bmp != null) {
                        val rgbaMat = Mat()
                        val bgrMat = Mat()
                        val scaledMat = Mat()
                        try {
                            Utils.bitmapToMat(bmp, rgbaMat)
                            Imgproc.cvtColor(rgbaMat, bgrMat, Imgproc.COLOR_RGBA2BGR)
                            resizeForAnalysis(bgrMat, scaledMat)
                            metricsList.add(analyzeMat(scaledMat))
                        } finally {
                            rgbaMat.release()
                            bgrMat.release()
                            scaledMat.release()
                            bmp.recycle()
                        }
                    }
                }
            } catch (t: Throwable) {
                warningMsg = "Fast frame sampler fallback triggered: ${t.message}"
                logger?.warn(warningMsg)
            } finally {
                try {
                    retriever.release()
                } catch (_: Throwable) {
                }
            }

            // Fallback: MediaExtractor + MediaCodec targeted keyframe seeks (zero temp disk files)
            if (metricsList.isEmpty()) {
                val fallbackMats = AndroidMediaToolkit.sampleRepresentativeFramesViaMediaCodec(
                    videoFile = videoFile,
                    durationMs = metadata.durationMs,
                    rotationDegrees = metadata.rotationDegrees,
                    fractions = listOf(0.15, 0.50, 0.85),
                    maxAnalyticWidth = 320
                )
                try {
                    for (mat in fallbackMats) {
                        if (!mat.empty()) {
                            metricsList.add(analyzeMat(mat))
                        }
                    }
                } finally {
                    fallbackMats.forEach { it.release() }
                }
                if (metricsList.isNotEmpty()) {
                    val fallbackAnalysis = aggregateMetrics(metricsList)
                    analysisCache[key] = fallbackAnalysis
                    val elapsed = System.currentTimeMillis() - startMs
                    logger?.info("Representative analysis via MediaCodec keyframe seek completed in ${elapsed}ms (${metricsList.size} frames)")
                    return@withTimeoutOrNull AnalysisOutcome(
                        analysis = fallbackAnalysis,
                        reusedFromCache = false,
                        sampledFrames = metricsList.size,
                        durationMs = elapsed,
                        warning = warningMsg
                    )
                }
                warningMsg = (warningMsg?.let { "$it; " } ?: "") + "No sample frames could be decoded; using balanced profile defaults"
                logger?.warn(warningMsg)
                val balanced = defaultBalancedAnalysis()
                return@withTimeoutOrNull AnalysisOutcome(
                    analysis = balanced,
                    reusedFromCache = false,
                    sampledFrames = 0,
                    durationMs = System.currentTimeMillis() - startMs,
                    warning = warningMsg
                )
            }

            val aggregated = aggregateMetrics(metricsList)
            analysisCache[key] = aggregated
            val elapsed = System.currentTimeMillis() - startMs
            logger?.info("Representative video analysis completed in ${elapsed}ms (${metricsList.size} frames): ${aggregated.conditionTags.joinToString()}")
            AnalysisOutcome(
                analysis = aggregated,
                reusedFromCache = false,
                sampledFrames = metricsList.size,
                durationMs = elapsed,
                warning = warningMsg
            )
        }

        if (timedOutcome != null) {
            return@withContext timedOutcome
        }

        val elapsed = System.currentTimeMillis() - startMs
        val timeoutWarn = "Representative frame analysis timed out after ${timeoutMs}ms; using balanced profile defaults"
        logger?.warn(timeoutWarn)
        return@withContext AnalysisOutcome(
            analysis = defaultBalancedAnalysis(),
            reusedFromCache = false,
            sampledFrames = 0,
            durationMs = elapsed,
            warning = timeoutWarn
        )
    }

    /**
     * Analyzes sampled BGR frame files using OpenCV to measure brightness, exposure clipping,
     * contrast, Immerkær noise sigma, Laplacian blur variance, HSV saturation,
     * gray-world white balance shift, 8x8 DCT blockiness, and Sobel edge density.
     */
    fun analyzeSampleFiles(frameFiles: List<File>): VideoQualityAnalysis {
        if (frameFiles.isEmpty()) {
            return defaultBalancedAnalysis()
        }

        val metricsList = mutableListOf<VideoQualityAnalysis>()
        for (file in frameFiles) {
            val bgr = Imgcodecs.imread(file.absolutePath)
            if (bgr.empty()) {
                bgr.release()
                continue
            }
            val analysisMat = Mat()
            try {
                resizeForAnalysis(bgr, analysisMat)
                metricsList.add(analyzeMat(analysisMat))
            } finally {
                analysisMat.release()
                bgr.release()
            }
        }

        return if (metricsList.isEmpty()) defaultBalancedAnalysis() else aggregateMetrics(metricsList)
    }

    private fun resizeForAnalysis(srcBgr: Mat, dstBgr: Mat) {
        val maxAnalyticW = 320.0
        if (srcBgr.cols() > maxAnalyticW) {
            val scale = maxAnalyticW / srcBgr.cols()
            Imgproc.resize(
                srcBgr,
                dstBgr,
                Size(maxAnalyticW, (srcBgr.rows() * scale).coerceAtLeast(32.0)),
                0.0,
                0.0,
                Imgproc.INTER_AREA
            )
        } else {
            srcBgr.copyTo(dstBgr)
        }
    }

    private fun aggregateMetrics(samples: List<VideoQualityAnalysis>): VideoQualityAnalysis {
        if (samples.isEmpty()) return defaultBalancedAnalysis()
        val n = samples.size.toFloat()
        val avgB = samples.sumOf { it.avgBrightness.toDouble() }.toFloat() / n
        val darkR = samples.sumOf { it.darkAreaRatio.toDouble() }.toFloat() / n
        val overR = samples.sumOf { it.overexposedAreaRatio.toDouble() }.toFloat() / n
        val contrastStd = samples.sumOf { it.contrastStdDev.toDouble() }.toFloat() / n
        val dynRange = samples.sumOf { it.dynamicRange.toDouble() }.toFloat() / n
        val noise = samples.sumOf { it.noiseLevel.toDouble() }.toFloat() / n
        val blur = samples.sumOf { it.blurVariance.toDouble() }.toFloat() / n
        val sat = samples.sumOf { it.saturationMean.toDouble() }.toFloat() / n
        val wbR = samples.sumOf { it.whiteBalanceShiftR.toDouble() }.toFloat() / n
        val wbB = samples.sumOf { it.whiteBalanceShiftB.toDouble() }.toFloat() / n
        val blockiness = samples.sumOf { it.compressionArtifactScore.toDouble() }.toFloat() / n
        val edgeDens = samples.sumOf { it.edgeDensity.toDouble() }.toFloat() / n

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
     * Computes calibrated adaptive adjustment parameters from the fixed HD, HD+, FHD, and 2K preset
     * while preserving natural skin tones, highlight/shadow detail, and avoiding double sharpening.
     */
    fun computeAdaptiveAdjustments(
        profile: EnhancementProfile,
        analysis: VideoQualityAnalysis
    ): AdaptiveAdjustmentParams {
        val notes = mutableListOf<String>()

        val isHighQualitySource =
            analysis.avgBrightness in 88f..175f &&
                analysis.overexposedAreaRatio < 0.08f &&
                analysis.darkAreaRatio < 0.25f &&
                analysis.noiseLevel < 5.5f &&
                analysis.blurVariance > 260f &&
                analysis.compressionArtifactScore < 1.15f

        val toneScale = if (isHighQualitySource) 0.88f else 1.0f
        if (isHighQualitySource) {
            notes.add("Clean source: calibrated clarity pass")
        }

        // 1. Brightness, Highlights, Shadows, Whites, Blacks & Gamma
        var brightnessShift = profile.brightnessPercent * toneScale
        var gamma = profile.gamma
        var highlightCompress = (abs(profile.highlightsPercent) / 100f).coerceIn(0f, 0.30f)
        var shadowsLift = (profile.shadowsPercent / 100f).coerceIn(0f, 0.20f) * toneScale
        val whitesGain = (profile.whitesPercent / 100f).coerceIn(0f, 0.16f) * toneScale
        val blacksCrush = (abs(profile.blacksPercent) / 100f).coerceIn(0f, 0.14f) * toneScale

        when {
            analysis.avgBrightness < 78f || analysis.darkAreaRatio > 0.35f -> {
                val darkSeverity = ((82f - analysis.avgBrightness).coerceIn(0f, 50f) / 50f)
                brightnessShift += 4f + darkSeverity * 10f
                gamma = (0.94f - darkSeverity * 0.10f).coerceIn(0.84f, 0.96f)
                shadowsLift = (shadowsLift + darkSeverity * 0.08f).coerceIn(0f, 0.24f)
                notes.add("Recovered shadow detail")
            }
            analysis.avgBrightness > 188f || analysis.overexposedAreaRatio > 0.15f -> {
                val overSeverity = ((analysis.avgBrightness - 180f).coerceIn(0f, 60f) / 60f)
                brightnessShift = (-3f - overSeverity * 8f).coerceAtLeast(-14f)
                gamma = (1.05f + overSeverity * 0.10f).coerceIn(1.03f, 1.18f)
                highlightCompress = (highlightCompress + 0.08f + analysis.overexposedAreaRatio * 0.45f).coerceIn(0.06f, 0.38f)
                notes.add("Protected highlights")
            }
        }

        // 2. Contrast, Brilliance & Clarity
        var contrastGain = 1f + (profile.contrastPercent / 100f) * toneScale
        val brillianceStrength = (profile.brilliancePercent / 100f).coerceIn(0f, 0.24f) * toneScale
        val clarityStrength = (profile.clarityPercent / 100f).coerceIn(0f, 0.40f) * toneScale
        var claheLimit = (0.25 + profile.clarityPercent * 0.032 + profile.brilliancePercent * 0.020) * toneScale
        var dehazing = (brillianceStrength * 0.60f + (profile.dehazingPercent / 100f) * 0.45f).coerceIn(0f, 0.20f)

        if (analysis.contrastStdDev < 42f) {
            val flatBoost = ((44f - analysis.contrastStdDev) / 44f).coerceIn(0.15f, 1.0f)
            contrastGain = (contrastGain + 0.06f * flatBoost).coerceAtMost(1.24f)
            claheLimit = (claheLimit + 0.38 * flatBoost).coerceAtMost(1.65)
            dehazing = (dehazing + 0.05f * flatBoost).coerceAtMost(0.22f)
            notes.add("Balanced flat contrast")
        } else if (analysis.contrastStdDev > 76f) {
            contrastGain = 1f + (contrastGain - 1f) * 0.55f
            claheLimit *= 0.65
        }

        // 3. Neural DnCNN Denoise + Temporal/Artifact Cleanup
        var denoiseStrength = (profile.noiseReductionPercent / 100f) * 0.60f * toneScale
        val baseDncnn = (profile.dncnnStrengthPercent / 100f).coerceIn(0.18f, 0.94f)
        var dncnnBoostFactor = 0f
        if (analysis.noiseLevel > 7.5f) {
            val noiseExtra = ((analysis.noiseLevel - 7f) / 14f).coerceIn(0.06f, 0.28f)
            denoiseStrength = (denoiseStrength + noiseExtra).coerceIn(0.15f, 0.88f)
            dncnnBoostFactor += ((analysis.noiseLevel - 7f) / 14f).coerceIn(0.08f, 0.32f)
            notes.add("DnCNN neural denoise")
        } else if (analysis.noiseLevel < 3.5f) {
            denoiseStrength *= 0.65f
        }

        val baseDeblock = (profile.artifactReductionPercent / 100f) * 0.55f * toneScale
        val deblockingStrength = if (analysis.compressionArtifactScore > 1.18f) {
            val extra = ((analysis.compressionArtifactScore - 1.12f) * 0.65f).coerceIn(0.06f, 0.28f)
            dncnnBoostFactor += ((analysis.compressionArtifactScore - 1.12f) * 0.35f).coerceIn(0.05f, 0.16f)
            notes.add("Deblocked artifacts")
            (baseDeblock + extra).coerceIn(0.10f, 0.80f)
        } else {
            baseDeblock.coerceIn(0.08f, 0.65f)
        }

        val clampedBoost = dncnnBoostFactor.coerceIn(0f, 0.45f)
        var dncnnStrength = baseDncnn + clampedBoost * (0.96f - baseDncnn)
        if (analysis.noiseLevel < 3.5f && analysis.compressionArtifactScore < 1.15f) {
            dncnnStrength *= 0.85f
        }
        val useAiDenoise = dncnnStrength >= 0.18f

        // 4. Detail Reconstruction, Deblur & Halo-Clamped AI Sharpening
        var sharpnessAmount = (profile.sharpnessPercent / 100f) * 0.72f * toneScale
        var edgeEnhancement = (profile.edgeEnhancementPercent / 100f) * 0.70f * toneScale
        val detailRecovery = (profile.detailRecoveryPercent / 100f).coerceIn(0.25f, 0.96f)
        var deblurStrength = (profile.deblurPercent / 100f).coerceIn(0.20f, 0.94f)
        var aiSharpening = (profile.aiSharpeningPercent / 100f).coerceIn(0.15f, 0.65f)

        if (analysis.blurVariance > 450f) {
            sharpnessAmount *= 0.60f
            edgeEnhancement *= 0.60f
            deblurStrength *= 0.72f
        } else if (analysis.blurVariance < 115f && analysis.noiseLevel < 11f) {
            deblurStrength = (deblurStrength + 0.12f).coerceAtMost(0.95f)
            aiSharpening = (aiSharpening + 0.08f).coerceAtMost(0.68f)
            notes.add("AI deblur & texture recovery")
        }
        if (analysis.noiseLevel > 11f) {
            sharpnessAmount *= 0.65f
        }

        // 5. Saturation, Selective Vibrance & White Balance (Skin-Protected, No Oversaturation)
        var saturationGain = 1f + (profile.saturationPercent / 100f) * toneScale
        var vibranceGain = (0.16f + profile.brilliancePercent * 0.012f + profile.saturationPercent * 0.010f) * toneScale
        if (analysis.saturationMean < 75f) {
            val dullSeverity = ((75f - analysis.saturationMean) / 55f).coerceIn(0.15f, 1.0f)
            saturationGain = (saturationGain + 0.05f * dullSeverity).coerceAtMost(1.22f)
            vibranceGain = (vibranceGain + 0.18f * dullSeverity).coerceAtMost(0.65f)
            notes.add("Restored vibrant color depth")
        } else if (analysis.saturationMean > 148f) {
            saturationGain = 1f + (saturationGain - 1f) * 0.45f
            vibranceGain *= 0.45f
        }

        val (wbGainR, wbGainB) = ColorMath.computeAutoWhiteBalanceGains(
            wbShiftR = analysis.whiteBalanceShiftR,
            wbShiftB = analysis.whiteBalanceShiftB,
            tempShift = profile.temperatureShift
        )
        if (abs(wbGainR - 1f) > 0.025f || abs(wbGainB - 1f) > 0.025f) {
            notes.add("Balanced white point")
        }

        val flickerAlpha = if (profile.flickerReduction.equals("Automatic", ignoreCase = true)) {
            if (isHighQualitySource) 0.10f else 0.20f
        } else 0f
        val motionSmoothStrength = if (profile.temporalConsistency) {
            (0.28f + (profile.aiStrengthPercent / 100f) * 0.35f).coerceIn(0.25f, 0.70f)
        } else 0f

        val aiStrength = (profile.aiStrengthPercent / 100f).coerceIn(0.20f, 0.96f)
        val faceEnhanceStrength = (profile.faceEnhancePercent / 100f).coerceIn(0.20f, 0.95f)

        return AdaptiveAdjustmentParams(
            profileId = profile.id,
            brightnessShift = brightnessShift.coerceIn(-12f, 14f),
            exposureHighlightCompress = highlightCompress.coerceIn(0f, 0.36f),
            contrastGain = contrastGain.coerceIn(1.0f, 1.24f),
            claheClipLimit = claheLimit.coerceIn(0.20, 1.65),
            gamma = gamma.coerceIn(0.86f, 1.16f),
            saturationGain = saturationGain.coerceIn(0.96f, 1.22f),
            vibranceGain = vibranceGain.coerceIn(0.10f, 0.65f),
            wbGainR = wbGainR,
            wbGainG = 1.0f,
            wbGainB = wbGainB,
            denoiseStrength = denoiseStrength.coerceIn(0.05f, 0.88f),
            useAiDenoise = useAiDenoise,
            deblockingStrength = deblockingStrength.coerceIn(0.05f, 0.80f),
            dehazingStrength = dehazing.coerceIn(0f, 0.22f),
            sharpnessAmount = sharpnessAmount.coerceIn(0.06f, 0.45f),
            edgeEnhancementAmount = edgeEnhancement.coerceIn(0.03f, 0.35f),
            flickerReductionAlpha = flickerAlpha,
            faceEnhanceStrength = faceEnhanceStrength,
            brillianceStrength = brillianceStrength,
            clarityStrength = clarityStrength,
            shadowsLift = shadowsLift,
            whitesGain = whitesGain,
            blacksCrush = blacksCrush,
            aiStrength = aiStrength,
            detailRecovery = detailRecovery,
            deblurStrength = deblurStrength,
            aiSharpening = aiSharpening,
            dncnnStrength = dncnnStrength.coerceIn(0.15f, 0.95f),
            skinToneProtection = true,
            enableSmoothness = profile.temporalConsistency,
            motionSmoothStrength = motionSmoothStrength,
            allowFrameInterpolation = profile.temporalConsistency,
            adjustmentSummary = buildString {
                append("${profile.displayName} • SR ${(aiStrength * 100).roundToInt()}% • DnCNN ${(dncnnStrength * 100).roundToInt()}%")
                if (notes.isNotEmpty()) {
                    append(" • ")
                    append(notes.joinToString(" • "))
                }
            }
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

            val meanMat = MatOfDouble()
            val stdMat = MatOfDouble()
            Core.meanStdDev(gray, meanMat, stdMat)
            val avgBrightness = meanMat.get(0, 0)[0].toFloat()
            val contrastStd = stdMat.get(0, 0)[0].toFloat()
            meanMat.release()
            stdMat.release()

            val darkMask = Mat()
            val overMask = Mat()
            Imgproc.threshold(gray, darkMask, 24.0, 255.0, Imgproc.THRESH_BINARY_INV)
            Imgproc.threshold(gray, overMask, 238.0, 255.0, Imgproc.THRESH_BINARY)
            val darkRatio = (Core.countNonZero(darkMask) / totalPixels).toFloat()
            val overRatio = (Core.countNonZero(overMask) / totalPixels).toFloat()
            darkMask.release()
            overMask.release()

            val dynamicRange = (contrastStd * 3.1f).coerceIn(0f, 255f)

            Imgproc.Laplacian(gray, lap, CvType.CV_64F)
            val lapMean = MatOfDouble()
            val lapStd = MatOfDouble()
            Core.meanStdDev(lap, lapMean, lapStd)
            val lapSigma = lapStd.get(0, 0)[0]
            val blurVariance = (lapSigma * lapSigma).toFloat()
            lapMean.release()
            lapStd.release()

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
            val noiseSigma = ((sumNoise * kotlin.math.sqrt(0.5 * PI)) / (6.0 * w * h)).toFloat()

            val hsvChannels = ArrayList<Mat>(3)
            Core.split(hsv, hsvChannels)
            val satMean = Core.mean(hsvChannels[1]).`val`[0].toFloat()
            hsvChannels.forEach { it.release() }

            val bgrMean = Core.mean(bgr).`val`
            val meanB = bgrMean[0].coerceAtLeast(1.0)
            val meanG = bgrMean[1].coerceAtLeast(1.0)
            val meanR = bgrMean[2].coerceAtLeast(1.0)
            val wbShiftR = (meanG / meanR).toFloat().coerceIn(0.6f, 1.5f)
            val wbShiftB = (meanG / meanB).toFloat().coerceIn(0.6f, 1.5f)

            Imgproc.Sobel(gray, sobelX, CvType.CV_32F, 1, 0, 3)
            Imgproc.Sobel(gray, sobelY, CvType.CV_32F, 0, 1, 3)
            Core.magnitude(sobelX, sobelY, sobelMag)
            val edgeMask = Mat()
            Imgproc.threshold(sobelMag, edgeMask, 45.0, 255.0, Imgproc.THRESH_BINARY)
            val edgeDensity = (Core.countNonZero(edgeMask) / totalPixels).toFloat()
            edgeMask.release()

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
