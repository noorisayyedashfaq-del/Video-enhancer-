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
                gamma = (0.94f - darkSeverity * 0.12f).coerceIn(0.80f, 0.96f)
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

        // 3. Denoise (invoke DnCNN only when noise warrants neural denoising, avoiding redundant passes on clean 1080p)
        var denoiseStrength = (profile.denoisePercent / 100f) * 1.4f * qualityDampener
        if (analysis.noiseLevel > 8.5f) {
            val noiseExtra = ((analysis.noiseLevel - 8f) / 14f).coerceIn(0.1f, 0.55f)
            denoiseStrength = (denoiseStrength + noiseExtra).coerceIn(0.15f, 0.92f)
            notes.add("Increased AI denoising")
        } else if (analysis.noiseLevel < 4.0f) {
            denoiseStrength *= 0.55f
        }
        val useAiDenoise = analysis.noiseLevel >= 7.5f || (denoiseStrength >= 0.34f && analysis.noiseLevel >= 5.5f)

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

        // 5. Sharpness & Edge Enhancement
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
