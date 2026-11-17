package com.example.engine

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.provider.OpenableColumns
import com.arthenica.ffmpegkit.FFmpegKit
import com.arthenica.ffmpegkit.FFprobeKit
import com.arthenica.ffmpegkit.ReturnCode
import com.example.model.DeviceCapabilities
import com.example.model.EngineHealthStatus
import com.example.model.EnhancementMode
import com.example.model.EnhancementProfile
import com.example.model.EnhancementResult
import com.example.model.ProcessingProgress
import com.example.model.ProcessingStage
import com.example.model.ThermalState
import com.example.model.VideoMetadata
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import org.opencv.imgcodecs.Imgcodecs
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max
import kotlin.math.roundToInt

class VideoMediaPipeline(
    private val context: Context,
    private val capabilityDetector: DeviceCapabilityDetector,
    private val onnxEngine: OnnxAiInferenceEngine,
    private val frameProcessor: OpenCvFrameProcessor,
    private val smartAdjuster: SmartAutoAdjustmentEngine
) {
    private val cancelled = AtomicBoolean(false)

    fun cancel() {
        cancelled.set(true)
        try {
            FFmpegKit.cancel()
        } catch (_: Exception) {
        }
    }

    suspend fun initializeAndCheckHealth(capabilities: DeviceCapabilities): EngineHealthStatus =
        withContext(Dispatchers.IO) {
            val cvReady = frameProcessor.ensureInitialized()
            var ortReady = false
            var diag: String? = null
            try {
                onnxEngine.initialize(capabilities)
                ortReady = true
            } catch (t: Throwable) {
                diag = "AI Runtime error: ${t.message}"
            }
            val ffReady = try {
                val ver = com.arthenica.ffmpegkit.FFmpegKitConfig.getVersion()
                ver != null
            } catch (t: Throwable) {
                diag = (diag ?: "") + " FFmpeg error: ${t.message}"
                false
            }
            EngineHealthStatus(
                openCvReady = cvReady,
                ffmpegReady = ffReady,
                onnxRuntimeReady = ortReady,
                realEsrganModelReady = onnxEngine.isRealEsrganReady,
                dncnnModelReady = onnxEngine.isDncnnReady,
                yunetModelReady = frameProcessor.isFaceDetectorReady,
                activeExecutionProvider = onnxEngine.providerLabel,
                diagnosticMessage = diag
            )
        }

    suspend fun importAndProbeVideo(uri: Uri): Result<VideoMetadata> = withContext(Dispatchers.IO) {
        runCatching {
            val workDir = File(context.cacheDir, "luminasr_probe").apply { mkdirs() }
            val probeFile = File(workDir, "probe_input.mp4")
            copyUriToFile(uri, probeFile)

            try {
                probeLocalFile(uri, probeFile, resolveDisplayName(uri))
            } finally {
                probeFile.delete()
            }
        }
    }

    suspend fun processVideo(
        metadata: VideoMetadata,
        mode: EnhancementMode,
        profile: EnhancementProfile,
        capabilities: DeviceCapabilities,
        onProgress: (ProcessingProgress) -> Unit
    ): Result<EnhancementResult> = withContext(Dispatchers.IO) {
        cancelled.set(false)
        val startWallMs = System.currentTimeMillis()
        val sessionId = System.currentTimeMillis().toString()
        val workDir = File(context.cacheDir, "luminasr_job_$sessionId").apply { mkdirs() }
        val outputDir = File(context.filesDir, "enhanced_outputs").apply { mkdirs() }
        // Keep only the latest output preview file in filesDir to preserve storage
        outputDir.listFiles()?.forEach { it.delete() }

        val inputCopy = File(workDir, "source_input.mp4")
        val framesDir = File(workDir, "frames").apply { mkdirs() }
        val rawVideoMp4 = File(workDir, "raw_enhanced.mp4")
        val finalMp4 = File(outputDir, "LuminaSR_${mode.label}_$sessionId.mp4")

        var encoder: HardwareVideoEncoder? = null
        var reusableBitmap: Bitmap? = null

        runCatching {
        try {
            // Ensure engines are ready
            check(frameProcessor.ensureInitialized()) { "OpenCV native engine failed to initialize." }
            onnxEngine.initialize(capabilities)
            frameProcessor.resetTemporalState()

            // Storage & Memory pre-flight
            val availStorageMb = capabilityDetector.currentAvailableStorageMb()
            val inputSizeMb = (metadata.fileSizeBytes / (1024L * 1024L)).coerceAtLeast(5L)
            val neededMb = max(capabilities.minRequiredFreeStorageMb, inputSizeMb * 3L + 120L)
            if (availStorageMb < neededMb) {
                throw IllegalStateException(
                    "Insufficient storage: ${availStorageMb}MB free, at least ${neededMb}MB required."
                )
            }
            val availRamMb = capabilityDetector.currentAvailableRamMb()
            if (availRamMb < 250L) {
                throw IllegalStateException("Insufficient free memory (${availRamMb}MB). Close background apps and retry.")
            }

            // 1. Copy input video locally for deterministic FFmpeg frame extraction & audio muxing
            onProgress(
                ProcessingProgress(
                    stage = ProcessingStage.IMPORTING,
                    overallProgress = 0.03f,
                    statusDetail = "Staging source video offline...",
                    thermalState = capabilities.thermalState,
                    activeAcceleration = onnxEngine.providerLabel
                )
            )
            copyUriToFile(metadata.uri, inputCopy)
            if (cancelled.get()) throw InterruptedException("Cancelled by user")

            // 2. Compute exact aspect-ratio-preserving output dimensions
            val outDims = ResolutionCalculator.calculate(metadata, mode, profile, capabilities)

            // 3. Extract frames using FFmpeg at exact original FPS
            onProgress(
                ProcessingProgress(
                    stage = ProcessingStage.EXTRACTING,
                    overallProgress = 0.08f,
                    statusDetail = "Extracting frames via FFmpeg (${String.format("%.2f", metadata.fps)} fps)...",
                    thermalState = capabilities.thermalState,
                    activeAcceleration = onnxEngine.providerLabel
                )
            )

            // Decide PNG vs high-quality JPEG (-q:v 1) based on storage headroom & frame count
            val usePngFrames = metadata.estimatedFrameCount <= 240 && availStorageMb > 1200L
            val ext = if (usePngFrames) "png" else "jpg"
            val framePattern = File(framesDir, "frame_%06d.$ext").absolutePath
            val safeFps = metadata.fps.coerceIn(10.0, 120.0)
            val qualityFlag = if (usePngFrames) "" else "-qscale:v 1 -qmin 1 -qmax 1"
            val extractCmd = "-y -i \"${inputCopy.absolutePath}\" -vf \"fps=${safeFps}\" $qualityFlag \"$framePattern\""

            val extractSession = FFmpegKit.execute(extractCmd)
            if (cancelled.get()) throw InterruptedException("Cancelled by user")
            if (!ReturnCode.isSuccess(extractSession.returnCode)) {
                throw IllegalStateException("Unsupported or corrupted video stream (FFmpeg exit ${extractSession.returnCode}).")
            }

            val frameFiles = framesDir.listFiles { f -> f.name.startsWith("frame_") && f.name.endsWith(".$ext") }
                ?.sortedBy { it.name }
                .orEmpty()
            if (frameFiles.isEmpty()) {
                throw IllegalStateException("No decodable video frames found in source file.")
            }

            // 4. Smart Auto Adjustment Engine: analyze evenly spaced representative frames
            onProgress(
                ProcessingProgress(
                    stage = ProcessingStage.ANALYZING,
                    totalFrames = frameFiles.size,
                    overallProgress = 0.16f,
                    statusDetail = "Analyzing brightness, noise, blur, color & compression...",
                    thermalState = capabilities.thermalState,
                    activeAcceleration = onnxEngine.providerLabel
                )
            )
            val sampleIndices = selectSampleIndices(frameFiles.size, maxSamples = 6)
            val sampleFiles = sampleIndices.map { frameFiles[it] }
            val qualityAnalysis = smartAdjuster.analyzeSampleFiles(sampleFiles)
            val adaptiveParams = smartAdjuster.computeAdaptiveAdjustments(profile, qualityAnalysis)
            if (cancelled.get()) throw InterruptedException("Cancelled by user")

            // 5. Initialize streaming MediaCodec encoder + reusable output Bitmap
            encoder = HardwareVideoEncoder(
                outputFile = rawVideoMp4,
                width = outDims.width,
                height = outDims.height,
                fps = safeFps,
                bitrateBps = outDims.targetBitrateBps,
                preferH265 = outDims.useH265
            ).apply { start() }

            reusableBitmap = Bitmap.createBitmap(outDims.width, outDims.height, Bitmap.Config.ARGB_8888)

            // 6. Frame-by-frame enhancement & streaming encode with thermal + memory guard
            val totalFrames = frameFiles.size
            val frameLoopStartMs = System.currentTimeMillis()
            var totalFacesEnhanced = 0
            var corruptedFramesRecovered = 0

            for ((idx, frameFile) in frameFiles.withIndex()) {
                if (cancelled.get()) throw InterruptedException("Cancelled by user")

                val (thermalState, tempC) = capabilityDetector.readThermalInfo()
                if (thermalState == ThermalState.WARM) {
                    delay(capabilities.throttleDelayMsOnWarm)
                } else if (thermalState == ThermalState.THROTTLING || thermalState == ThermalState.CRITICAL) {
                    delay(capabilities.throttleDelayMsOnHot)
                }

                // Dynamic memory safety check
                if (idx % 8 == 0 && capabilityDetector.currentAvailableRamMb() < 180L) {
                    System.gc()
                    delay(40)
                }

                val bgr = Imgcodecs.imread(frameFile.absolutePath)
                if (bgr.empty()) {
                    bgr.release()
                    corruptedFramesRecovered++
                    // Reuse previous valid reusableBitmap if available rather than crashing
                    if (idx > 0) {
                        encoder.encodeFrame(reusableBitmap, idx)
                    }
                    frameFile.delete()
                    continue
                }

                try {
                    val faces = frameProcessor.processFrame(
                        srcBgr = bgr,
                        outDims = outDims,
                        params = adaptiveParams,
                        capabilities = capabilities,
                        reusableOutBitmap = reusableBitmap,
                        isCancelled = { cancelled.get() }
                    )
                    totalFacesEnhanced += faces
                    encoder.encodeFrame(reusableBitmap, idx)
                } catch (e: InterruptedException) {
                    throw e
                } catch (oom: OutOfMemoryError) {
                    throw IllegalStateException("Device memory exhausted during ${outDims.label} processing.", oom)
                } catch (_: Throwable) {
                    corruptedFramesRecovered++
                    if (idx > 0) {
                        encoder.encodeFrame(reusableBitmap, idx)
                    }
                } finally {
                    bgr.release()
                    // Delete extracted frame immediately to keep disk footprint tiny
                    frameFile.delete()
                }

                val done = idx + 1
                val elapsedMs = (System.currentTimeMillis() - frameLoopStartMs).coerceAtLeast(1L)
                val msPerFrame = elapsedMs.toDouble() / done.toDouble()
                val remainingSec = (((totalFrames - done) * msPerFrame) / 1000.0).roundToInt().toLong()
                val overall = 0.18f + (done.toFloat() / totalFrames.toFloat()) * 0.74f

                onProgress(
                    ProcessingProgress(
                        stage = ProcessingStage.ENHANCING,
                        processedFrames = done,
                        totalFrames = totalFrames,
                        overallProgress = overall.coerceIn(0.18f, 0.92f),
                        etaSeconds = remainingSec,
                        statusDetail = buildString {
                            append("Frame $done / $totalFrames • ${outDims.label}")
                            if (thermalState >= ThermalState.THROTTLING) {
                                append(" • Thermal throttle (${String.format("%.1f", tempC)}°C)")
                            }
                            if (corruptedFramesRecovered > 0) {
                                append(" • $corruptedFramesRecovered recovered")
                            }
                        },
                        thermalState = thermalState,
                        activeAcceleration = onnxEngine.providerLabel
                    )
                )
            }

            if (cancelled.get()) throw InterruptedException("Cancelled by user")

            // Finalize video stream
            onProgress(
                ProcessingProgress(
                    stage = ProcessingStage.ENCODING,
                    processedFrames = totalFrames,
                    totalFrames = totalFrames,
                    overallProgress = 0.94f,
                    statusDetail = "Finalizing ${encoder.activeCodecLabel} stream...",
                    thermalState = capabilities.thermalState,
                    activeAcceleration = onnxEngine.providerLabel
                )
            )
            encoder.finish()
            val usedCodecLabel = encoder.activeCodecLabel
            encoder = null

            // 7. Remux & synchronize original audio using FFmpeg
            onProgress(
                ProcessingProgress(
                    stage = ProcessingStage.MUXING_AUDIO,
                    processedFrames = totalFrames,
                    totalFrames = totalFrames,
                    overallProgress = 0.97f,
                    statusDetail = if (metadata.hasAudio) "Synchronizing original audio track..." else "Finalizing MP4 container...",
                    thermalState = capabilities.thermalState,
                    activeAcceleration = onnxEngine.providerLabel
                )
            )

            muxWithOriginalAudio(rawVideoMp4, inputCopy, finalMp4, metadata.hasAudio)

            if (!finalMp4.exists() || finalMp4.length() == 0L) {
                throw IllegalStateException("Export failed: output MP4 could not be written.")
            }

            val totalElapsed = System.currentTimeMillis() - startWallMs
            onProgress(
                ProcessingProgress(
                    stage = ProcessingStage.COMPLETED,
                    processedFrames = totalFrames,
                    totalFrames = totalFrames,
                    overallProgress = 1f,
                    etaSeconds = 0L,
                    statusDetail = "Completed in ${formatElapsed(totalElapsed)}",
                    thermalState = capabilities.thermalState,
                    activeAcceleration = onnxEngine.providerLabel
                )
            )

            EnhancementResult(
                outputFile = finalMp4,
                mode = mode,
                originalMetadata = metadata,
                outputDimensions = outDims,
                codecUsed = usedCodecLabel,
                processingDurationMs = totalElapsed,
                qualityAnalysis = qualityAnalysis,
                appliedParams = adaptiveParams,
                facesEnhancedCount = totalFacesEnhanced,
                usedAiSuperResolution = outDims.needsAiSuperResolution && onnxEngine.isRealEsrganReady,
                usedAiDenoise = adaptiveParams.useAiDenoise && onnxEngine.isDncnnReady
            )
        } catch (interrupted: InterruptedException) {
            encoder?.abort()
            finalMp4.delete()
            throw interrupted
        } catch (t: Throwable) {
            encoder?.abort()
            finalMp4.delete()
            throw t
        } finally {
            reusableBitmap?.recycle()
            frameProcessor.resetTemporalState()
            workDir.deleteRecursively()
        }
        }
    }

    private fun muxWithOriginalAudio(
        rawVideoMp4: File,
        originalInput: File,
        finalMp4: File,
        hasAudio: Boolean
    ) {
        if (!hasAudio) {
            val cmd = "-y -i \"${rawVideoMp4.absolutePath}\" -c:v copy -movflags +faststart \"${finalMp4.absolutePath}\""
            val session = FFmpegKit.execute(cmd)
            if (!ReturnCode.isSuccess(session.returnCode) || !finalMp4.exists() || finalMp4.length() == 0L) {
                rawVideoMp4.copyTo(finalMp4, overwrite = true)
            }
            return
        }

        // Attempt lossless audio stream copy first
        val copyCmd = "-y -i \"${rawVideoMp4.absolutePath}\" -i \"${originalInput.absolutePath}\" " +
            "-map 0:v:0 -map 1:a:0 -c:v copy -c:a copy -shortest -movflags +faststart \"${finalMp4.absolutePath}\""
        val copySession = FFmpegKit.execute(copyCmd)
        if (ReturnCode.isSuccess(copySession.returnCode) && finalMp4.exists() && finalMp4.length() > 0L) {
            return
        }

        // Fallback: transcode non-MP4 audio (e.g., Opus/Vorbis/PCM) to high-quality AAC
        if (finalMp4.exists()) finalMp4.delete()
        val aacCmd = "-y -i \"${rawVideoMp4.absolutePath}\" -i \"${originalInput.absolutePath}\" " +
            "-map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -b:a 192k -shortest -movflags +faststart \"${finalMp4.absolutePath}\""
        val aacSession = FFmpegKit.execute(aacCmd)
        if (!ReturnCode.isSuccess(aacSession.returnCode) || !finalMp4.exists() || finalMp4.length() == 0L) {
            rawVideoMp4.copyTo(finalMp4, overwrite = true)
        }
    }

    private fun probeLocalFile(uri: Uri, file: File, displayName: String): VideoMetadata {
        val probeInfo = FFprobeKit.getMediaInformation(file.absolutePath).mediaInformation
        val retriever = MediaMetadataRetriever()
        var mmrWidth = 0
        var mmrHeight = 0
        var mmrRot = 0
        var mmrDurMs = 0L
        try {
            retriever.setDataSource(file.absolutePath)
            mmrWidth = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toIntOrNull() ?: 0
            mmrHeight = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toIntOrNull() ?: 0
            mmrRot = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)?.toIntOrNull() ?: 0
            mmrDurMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
        } catch (_: Exception) {
        } finally {
            try {
                retriever.release()
            } catch (_: Exception) {
            }
        }

        val streams = probeInfo?.streams.orEmpty()
        val videoStream = streams.firstOrNull { it.type == "video" }
        val audioStream = streams.firstOrNull { it.type == "audio" }

        val rawW = (videoStream?.width?.toInt() ?: mmrWidth).coerceAtLeast(0)
        val rawH = (videoStream?.height?.toInt() ?: mmrHeight).coerceAtLeast(0)
        if (rawW <= 0 || rawH <= 0) {
            throw IllegalArgumentException("Unsupported or unreadable video file.")
        }

        val rotation = (mmrRot % 360 + 360) % 360
        val swapDims = rotation == 90 || rotation == 270
        val displayW = if (swapDims) rawH else rawW
        val displayH = if (swapDims) rawW else rawH

        val fps = parseFps(videoStream?.averageFrameRate)
            ?: parseFps(videoStream?.realFrameRate)
            ?: 30.0
        val durationMs = ((probeInfo?.duration?.toDoubleOrNull()?.times(1000.0))?.toLong() ?: mmrDurMs)
            .coerceAtLeast(100L)
        val estFrames = ((durationMs / 1000.0) * fps).roundToInt().coerceAtLeast(1)

        val bitrate = probeInfo?.bitrate?.toLongOrNull()
            ?: videoStream?.bitrate?.toLongOrNull()
            ?: ((file.length() * 8_000L) / durationMs)

        return VideoMetadata(
            uri = uri,
            displayName = displayName,
            width = rawW,
            height = rawH,
            rotationDegrees = rotation,
            displayWidth = displayW,
            displayHeight = displayH,
            fps = fps,
            durationMs = durationMs,
            estimatedFrameCount = estFrames,
            bitrateBps = bitrate,
            videoCodec = videoStream?.codec ?: "unknown",
            hasAudio = audioStream != null,
            audioCodec = audioStream?.codec,
            audioSampleRate = audioStream?.sampleRate?.toIntOrNull(),
            audioChannels = null,
            fileSizeBytes = file.length()
        )
    }

    private fun parseFps(rateStr: String?): Double? {
        if (rateStr.isNullOrBlank()) return null
        val parts = rateStr.split("/")
        val fps = if (parts.size == 2) {
            val num = parts[0].toDoubleOrNull() ?: return null
            val den = parts[1].toDoubleOrNull() ?: return null
            if (den <= 0.0) return null else num / den
        } else {
            rateStr.toDoubleOrNull() ?: return null
        }
        return if (fps in 5.0..120.0) fps else null
    }

    private fun selectSampleIndices(total: Int, maxSamples: Int): List<Int> {
        if (total <= maxSamples) return (0 until total).toList()
        return (0 until maxSamples).map { i ->
            ((i.toDouble() / (maxSamples - 1)) * (total - 1)).roundToInt().coerceIn(0, total - 1)
        }.distinct()
    }

    private fun copyUriToFile(uri: Uri, dest: File) {
        dest.parentFile?.mkdirs()
        context.contentResolver.openInputStream(uri)?.use { input ->
            dest.outputStream().use { output ->
                input.copyTo(output)
            }
        } ?: throw IllegalStateException("Unable to read selected video URI.")
        if (!dest.exists() || dest.length() == 0L) {
            throw IllegalStateException("Selected video file is empty.")
        }
    }

    private fun resolveDisplayName(uri: Uri): String {
        if (uri.scheme == "content") {
            context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor ->
                if (cursor.moveToFirst()) {
                    val idx = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                    if (idx >= 0) return cursor.getString(idx) ?: "video.mp4"
                }
            }
        }
        return uri.lastPathSegment?.substringAfterLast('/') ?: "video.mp4"
    }

    private fun formatElapsed(ms: Long): String {
        val sec = (ms / 1000L).coerceAtLeast(1L)
        return if (sec < 60) "${sec}s" else "${sec / 60}m ${sec % 60}s"
    }
}
