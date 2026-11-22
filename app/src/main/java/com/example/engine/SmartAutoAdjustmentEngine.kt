package com.example.engine

import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.os.Build
import com.arthenica.ffmpegkit.FFmpegKit
import com.arthenica.ffmpegkit.ReturnCode
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
     * Samples up to 5 evenly-spaced keyframes via MediaMetadataRetriever at <= 480px width,
     * falling back to targeted FFmpeg single-frame seeks if needed, with strict timeout recovery.
     */
    suspend fun analyzeVideoRepresentativeFrames(
        videoFile: File,
        metadata: VideoMetadata,
        tempDir: File,
        timeoutMs: Long = 10_000L,
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
                sampledFrames = 5,
                durationMs = elapsed
            )
        }

        val timedOutcome = withTimeoutOrNull(timeoutMs) {
            var warningMsg: String? = null
            val metricsList = mutableListOf<VideoQualityAnalysis>()

            // Primary fast sampler: MediaMetadataRetriever sync-frame decode at scaled resolution
            val retriever = MediaMetadataRetriever()
            try {
                retriever.setDataSource(videoFile.absolutePath)
                val durUs = (metadata.durationMs.coerceAtLeast(300L)) * 1000L
                val fractions = listOf(0.10, 0.30, 0.50, 0.70, 0.90)
                for (frac in fractions) {
                    val timeUs = (durUs * frac).toLong().coerceAtLeast(0L)
                    val bmp: Bitmap? = try {
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
                            retriever.getScaledFrameAtTime(
                                timeUs,
                                MediaMetadataRetriever.OPTION_CLOSEST_SYNC,
                                480,
                                480
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

            // Fallback: FFmpeg single-frame targeted seeks (still avoids extracting entire video!)
            if (metricsList.isEmpty()) {
                val sampleFiles = mutableListOf<File>()
                val sampleDir = File(tempDir, "analysis_samples").apply { mkdirs() }
                val durSec = (metadata.durationMs / 1000.0).coerceAtLeast(0.5)
                val timestamps = listOf(durSec * 0.15, durSec * 0.50, durSec * 0.85)
                for ((idx, sec) in timestamps.withIndex()) {
                    val outFile = File(sampleDir, "sample_$idx.jpg")
                    val cmd = "-y -ss ${String.format(java.util.Locale.US, "%.2f", sec)} -i \"${videoFile.absolutePath}\" -frames:v 1 -vf \"scale='min(480,iw)':-2\" -qscale:v 2 \"${outFile.absolutePath}\""
                    val session = FFmpegKit.execute(cmd)
                    if (ReturnCode.isSuccess(session.returnCode) && outFile.exists() && outFile.length() > 0L) {
                        sampleFiles.add(outFile)
                    }
                }
                if (sampleFiles.isNotEmpty()) {
                    val fallbackAnalysis = analyzeSampleFiles(sampleFiles)
                    sampleDir.deleteRecursively()
                    analysisCache[key] = fallbackAnalysis
                    val elapsed = System.currentTimeMillis() - startMs
                    logger?.info("Representative analysis via FFmpeg seek completed in ${elapsed}ms (${sampleFiles.size} frames)")
                    return@withTimeoutOrNull AnalysisOutcome(
                        analysis = fallbackAnalysis,
                        reusedFromCache = false,
                        sampledFrames = sampleFiles.size,
                        durationMs = elapsed,
                        warning = warningMsg
                    )
                }
                sampleDir.deleteRecursively()
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
        val maxAnalyticW = 480.0
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
     * Computes calibrated adaptive adjustment parameters from the fixed Clarity 1x-4x preset
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

        val qualityDampener = if (isHighQualitySource) 0.72f else 1.0f
        if (isHighQualitySource) {
            notes.add("Clean source: calibrated gentle pass")
        }

        // 1. Brightness, Highlights, Shadows, Whites, Blacks & Gamma
        var brightnessShift = profile.brightnessPercent * qualityDampener
        var gamma = profile.gamma
        var highlightCompress = (kotlin.math.abs(profile.highlightsPercent) / 100f).coerceIn(0f, 0.25f)
        var shadowsLift = (profile.shadowsPercent / 100f).coerceIn(0f, 0.18f) * qualityDampener
        val whitesGain = (profile.whitesPercent / 100f).coerceIn(0f, 0.14f) * qualityDampener
        val blacksCrush = (kotlin.math.abs(profile.blacksPercent) / 100f).coerceIn(0f, 0.12f) * qualityDampener

        when {
            analysis.avgBrightness < 78f || analysis.darkAreaRatio > 0.35f -> {
                val darkSeverity = ((82f - analysis.avgBrightness).coerceIn(0f, 50f) / 50f)
                brightnessShift += 4f + darkSeverity * 10f
                gamma = (0.95f - darkSeverity * 0.10f).coerceIn(0.84f, 0.97f)
                shadowsLift = (shadowsLift + darkSeverity * 0.06f).coerceIn(0f, 0.22f)
                notes.add("Recovered shadow detail")
            }
            analysis.avgBrightness > 188f || analysis.overexposedAreaRatio > 0.15f -> {
                val overSeverity = ((analysis.avgBrightness - 180f).coerceIn(0f, 60f) / 60f)
                brightnessShift = (-3f - overSeverity * 8f).coerceAtLeast(-14f)
                gamma = (1.05f + overSeverity * 0.10f).coerceIn(1.03f, 1.18f)
                highlightCompress = (highlightCompress + 0.08f + analysis.overexposedAreaRatio * 0.45f).coerceIn(0.05f, 0.36f)
                notes.add("Protected highlights")
            }
        }

        // 2. Contrast, Brilliance & Clarity (restrained CLAHE prevents unnatural HDR halos)
        var contrastGain = 1f + (profile.contrastPercent / 100f) * qualityDampener
        val brillianceStrength = (profile.brilliancePercent / 100f).coerceIn(0f, 0.20f) * qualityDampener
        val clarityStrength = (profile.clarityPercent / 100f).coerceIn(0f, 0.35f) * qualityDampener
        var claheLimit = (0.15 + profile.clarityPercent * 0.028 + profile.brilliancePercent * 0.018) * qualityDampener
        var dehazing = (brillianceStrength * 0.55f + (profile.dehazingPercent / 100f) * 0.35f).coerceIn(0f, 0.14f)

        if (analysis.contrastStdDev < 38f) {
            contrastGain = (contrastGain + 0.06f).coerceAtMost(1.20f)
            claheLimit = (claheLimit + 0.35).coerceAtMost(1.55)
            dehazing = (dehazing + 0.04f).coerceAtMost(0.18f)
            notes.add("Balanced flat contrast")
        } else if (analysis.contrastStdDev > 74f) {
            contrastGain = 1f + (contrastGain - 1f) * 0.55f
            claheLimit *= 0.65
        }

        // 3. Noise Reduction & Artifact Reduction
        var denoiseStrength = (profile.noiseReductionPercent / 100f) * 0.65f * qualityDampener
        if (analysis.noiseLevel > 8.5f) {
            val noiseExtra = ((analysis.noiseLevel - 8f) / 16f).coerceIn(0.08f, 0.35f)
            denoiseStrength = (denoiseStrength + noiseExtra).coerceIn(0.12f, 0.85f)
            notes.add("Applied noise reduction")
        } else if (analysis.noiseLevel < 4.0f) {
            denoiseStrength *= 0.60f
        }
        val useAiDenoise = analysis.noiseLevel >= 7.5f || (denoiseStrength >= 0.38f && analysis.noiseLevel >= 5.5f)

        val baseDeblock = (profile.artifactReductionPercent / 100f) * 0.60f * qualityDampener
        val deblockingStrength = if (analysis.compressionArtifactScore > 1.20f) {
            val extra = ((analysis.compressionArtifactScore - 1.15f) * 0.70f).coerceIn(0.08f, 0.32f)
            notes.add("Reduced compression blocks")
            (baseDeblock + extra).coerceIn(0.10f, 0.80f)
        } else {
            baseDeblock.coerceIn(0.08f, 0.65f)
        }

        // 4. Unified Clarity / Fallback Sharpness (AI modes skip separate unsharp when ONNX runs to avoid double sharpening)
        var sharpnessAmount = (profile.sharpnessPercent / 100f) * 0.85f * qualityDampener
        var edgeEnhancement = (profile.edgeEnhancementPercent / 100f) * 0.80f * qualityDampener
        if (analysis.blurVariance > 400f) {
            sharpnessAmount *= 0.55f
            edgeEnhancement *= 0.55f
        } else if (analysis.blurVariance < 100f && analysis.noiseLevel < 9f) {
            sharpnessAmount = (sharpnessAmount + 0.08f).coerceAtMost(0.55f)
            notes.add("Enhanced edge clarity")
        }
        if (analysis.noiseLevel > 11f) {
            sharpnessAmount *= 0.65f
        }

        // 5. Saturation & Neutral Temperature / Hue (Skin-tone safe)
        var saturationGain = 1f + (profile.saturationPercent / 100f) * qualityDampener
        if (analysis.saturationMean < 52f) {
            saturationGain = (saturationGain + 0.08f).coerceAtMost(1.22f)
            notes.add("Restored natural color vibrancy")
        } else if (analysis.saturationMean > 142f) {
            saturationGain = 1f + (saturationGain - 1f) * 0.45f
        }

        // Preserve Temperature = 0 & Hue = 0 defaults unless severe color cast warrants minimal correction
        val hasStrongCast = abs(analysis.whiteBalanceShiftR - 1f) > 0.16f || abs(analysis.whiteBalanceShiftB - 1f) > 0.16f
        val tempBias = (profile.temperatureShift / 100f).coerceIn(-0.15f, 0.15f)
        val wbGainR = (if (hasStrongCast) 1f + (analysis.whiteBalanceShiftR - 1f) * 0.25f else 1f) + tempBias
        val wbGainB = (if (hasStrongCast) 1f + (analysis.whiteBalanceShiftB - 1f) * 0.25f else 1f) - tempBias

        val flickerAlpha = if (profile.flickerReduction.equals("Automatic", ignoreCase = true)) {
            if (isHighQualitySource) 0.08f else 0.16f
        } else 0f

        val aiStrength = (profile.aiStrengthPercent / 100f).coerceIn(0.15f, 0.95f)
        val detailRecovery = (profile.detailRecoveryPercent / 100f).coerceIn(0.15f, 0.95f)
        val deblurStrength = (profile.deblurPercent / 100f).coerceIn(0.10f, 0.90f)
        val aiSharpening = (profile.aiSharpeningPercent / 100f).coerceIn(0.10f, 0.75f)

        return AdaptiveAdjustmentParams(
            profileId = profile.id,
            brightnessShift = brightnessShift.coerceIn(-16f, 18f),
            exposureHighlightCompress = highlightCompress.coerceIn(0f, 0.38f),
            contrastGain = contrastGain.coerceIn(1.0f, 1.24f),
            claheClipLimit = claheLimit.coerceIn(0.2, 1.55),
            gamma = gamma.coerceIn(0.84f, 1.18f),
            saturationGain = saturationGain.coerceIn(0.95f, 1.24f),
            wbGainR = wbGainR.coerceIn(0.90f, 1.12f),
            wbGainG = 1.0f,
            wbGainB = wbGainB.coerceIn(0.90f, 1.12f),
            denoiseStrength = denoiseStrength.coerceIn(0.05f, 0.85f),
            useAiDenoise = useAiDenoise,
            deblockingStrength = deblockingStrength.coerceIn(0.05f, 0.80f),
            dehazingStrength = dehazing.coerceIn(0f, 0.18f),
            sharpnessAmount = sharpnessAmount.coerceIn(0.05f, 0.58f),
            edgeEnhancementAmount = edgeEnhancement.coerceIn(0.04f, 0.42f),
            flickerReductionAlpha = flickerAlpha,
            faceEnhanceStrength = 0f,
            brillianceStrength = brillianceStrength,
            clarityStrength = clarityStrength,
            shadowsLift = shadowsLift,
            whitesGain = whitesGain,
            blacksCrush = blacksCrush,
            aiStrength = aiStrength,
            detailRecovery = detailRecovery,
            deblurStrength = deblurStrength,
            aiSharpening = aiSharpening,
            skinToneProtection = true,
            adjustmentSummary = buildString {
                append("${profile.displayName} • AI ${(aiStrength * 100).roundToInt()}%")
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
