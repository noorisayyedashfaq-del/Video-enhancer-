package com.example.engine

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.provider.OpenableColumns
import com.arthenica.ffmpegkit.FFmpegKit
import com.arthenica.ffmpegkit.FFmpegSession
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
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.core.Mat
import org.opencv.imgcodecs.Imgcodecs
import java.io.File
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.max
import kotlin.math.roundToInt

class VideoMediaPipeline(
    private val context: Context,
    private val capabilityDetector: DeviceCapabilityDetector,
    private val onnxEngine: OnnxAiInferenceEngine,
    private val frameProcessor: OpenCvFrameProcessor,
    private val smartAdjuster: SmartAutoAdjustmentEngine
) {
    private data class DecodedFrame(
        val index: Int,
        val bgrMat: Mat? = null,
        val diskFile: File? = null
    )

    private val cancelled = AtomicBoolean(false)

    fun cancel() {
        cancelled.set(true)
        try {
            FFmpegKit.cancel()
        } catch (_: Throwable) {
        }
    }

    suspend fun initializeAndCheckHealth(capabilities: DeviceCapabilities): EngineHealthStatus =
        withContext(Dispatchers.IO) {
            val logger = EngineDiagnosticsLogger()
            val cvReady = frameProcessor.ensureInitialized()
            var ortReady = false
            var diag: String? = null
            try {
                onnxEngine.initialize(capabilities, logger)
                ortReady = true
            } catch (t: Throwable) {
                logger.error("AI Runtime initialization failed", t)
                diag = "AI Runtime error: ${t.message}"
            }
            val ffReady = try {
                com.arthenica.ffmpegkit.FFmpegKitConfig.getVersion() != null
            } catch (t: Throwable) {
                logger.error("FFmpeg initialization failed", t)
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
        val logger = EngineDiagnosticsLogger()
        val startWallMs = System.currentTimeMillis()
        val sessionId = System.currentTimeMillis().toString()
        val workDir = File(context.cacheDir, "luminasr_job_$sessionId").apply { mkdirs() }
        val outputDir = File(context.filesDir, "enhanced_outputs").apply { mkdirs() }
        outputDir.listFiles()?.forEach { it.delete() }

        val inputCopy = File(workDir, "source_input.mp4")
        val framesDir = File(workDir, "frames").apply { mkdirs() }
        val rawVideoMp4 = File(workDir, "raw_enhanced.mp4")
        val finalMp4 = File(outputDir, "ClearKit_${mode.label}_$sessionId.mp4")

        var encoder: HardwareVideoEncoder? = null
        var reusableBitmap: Bitmap? = null
        var activeExtractSession: FFmpegSession? = null

        runCatching {
            try {
                check(frameProcessor.ensureInitialized()) { "OpenCV native engine failed to initialize." }
                frameProcessor.resetTemporalState()

                // 1. Metadata & Resource Pre-flight
                val availStorageMb = capabilityDetector.currentAvailableStorageMb()
                val inputSizeMb = (metadata.fileSizeBytes / (1024L * 1024L)).coerceAtLeast(5L)
                val neededMb = max(capabilities.minRequiredFreeStorageMb, inputSizeMb + 100L)
                if (availStorageMb < neededMb) {
                    throw IllegalStateException(
                        "Insufficient storage: ${availStorageMb}MB free, at least ${neededMb}MB required."
                    )
                }
                val availRamMb = capabilityDetector.currentAvailableRamMb()
                if (availRamMb < 220L) {
                    throw IllegalStateException("Insufficient free memory (${availRamMb}MB). Close background apps and retry.")
                }

                onProgress(
                    ProcessingProgress(
                        stage = ProcessingStage.IMPORTING,
                        overallProgress = 0.03f,
                        statusDetail = "Staging source video (${metadata.resolutionLabel})...",
                        thermalState = capabilities.thermalState,
                        activeAcceleration = onnxEngine.providerLabel,
                        memoryUsageMb = logger.sampleHeapUsedMb()
                    )
                )
                copyUriToFile(metadata.uri, inputCopy)
                if (cancelled.get()) throw InterruptedException("Cancelled by user")

                val outDims = ResolutionCalculator.calculate(metadata, mode, profile, capabilities)
                logger.info("Job started: mode=${mode.label}, input=${metadata.resolutionLabel}@${metadata.fps}fps -> output=${outDims.label}")

                // 2. Fast Representative Video Analysis (cached + 10s timeout guard)
                onProgress(
                    ProcessingProgress(
                        stage = ProcessingStage.ANALYZING,
                        totalFrames = metadata.estimatedFrameCount,
                        overallProgress = 0.07f,
                        statusDetail = "Sampling representative frames...",
                        thermalState = capabilities.thermalState,
                        activeAcceleration = onnxEngine.providerLabel,
                        memoryUsageMb = logger.sampleHeapUsedMb()
                    )
                )
                val analysisOutcome = smartAdjuster.analyzeVideoRepresentativeFrames(
                    videoFile = inputCopy,
                    metadata = metadata,
                    tempDir = workDir,
                    timeoutMs = 10_000L,
                    logger = logger
                )
                logger.analysisMs = analysisOutcome.durationMs
                val qualityAnalysis = analysisOutcome.analysis
                if (cancelled.get()) throw InterruptedException("Cancelled by user")

                // 3. Adaptive Adjustments
                val adaptiveParams = smartAdjuster.computeAdaptiveAdjustments(profile, qualityAnalysis)

                // 4. Initialize & Benchmark GPU vs XNNPACK 4/6/8T Once
                if (outDims.needsAiSuperResolution || onnxEngine.isGlesGpuReady.not()) {
                    onProgress(
                        ProcessingProgress(
                            stage = ProcessingStage.LOADING_MODEL,
                            totalFrames = metadata.estimatedFrameCount,
                            overallProgress = 0.12f,
                            statusDetail = if (outDims.needsAiSuperResolution) {
                                "Benchmarking GPU & multi-thread AI (${outDims.aiTileSize}px × ${outDims.tilesPerFrame} tiles)..."
                            } else {
                                "Initializing hardware GPU/OpenCV acceleration..."
                            },
                            thermalState = capabilities.thermalState,
                            activeAcceleration = onnxEngine.providerLabel,
                            memoryUsageMb = logger.sampleHeapUsedMb()
                        )
                    )
                    onnxEngine.initialize(capabilities, logger)
                    logger.modelInitMs = onnxEngine.initDurationMs
                } else {
                    logger.modelInitMs = 0L
                }

                val safeFps = metadata.fps.coerceIn(10.0, 120.0)
                val activeEncoder = HardwareVideoEncoder(
                    outputFile = rawVideoMp4,
                    width = outDims.width,
                    height = outDims.height,
                    fps = safeFps,
                    bitrateBps = outDims.targetBitrateBps,
                    preferH265 = outDims.useH265
                ).apply { start() }
                encoder = activeEncoder

                val outputBitmap = Bitmap.createBitmap(outDims.width, outDims.height, Bitmap.Config.ARGB_8888)
                reusableBitmap = outputBitmap

                onProgress(
                    ProcessingProgress(
                        stage = ProcessingStage.EXTRACTING,
                        totalFrames = metadata.estimatedFrameCount,
                        overallProgress = 0.16f,
                        statusDetail = "Decoding video stream via zero-copy MediaCodec...",
                        thermalState = capabilities.thermalState,
                        activeAcceleration = onnxEngine.providerLabel,
                        memoryUsageMb = logger.sampleHeapUsedMb()
                    )
                )

                val presetTileSize = outDims.aiTileSize
                val channelCap = if (capabilities.availableRamMb > 1800L) 3 else 2
                val frameChannel = Channel<DecodedFrame>(capacity = channelCap)

                var totalFacesEnhanced = 0
                var recoveredFrames = 0
                var processedCount = 0
                var lightweightFallbackActive = false
                var activeAccelLabel = onnxEngine.providerLabel
                val expectedTotalFrames = metadata.estimatedFrameCount.coerceAtLeast(1)
                val decodedCountAtomic = AtomicInteger(0)

                coroutineScope {
                    // Producer: Primary zero-disk hardware MediaCodec decoder, automatic FFmpegKit fallback
                    val producerJob = launch(Dispatchers.IO) {
                        val decodeStartMs = System.currentTimeMillis()
                        var mediaCodecSucceeded = false
                        var sentAny = false

                        try {
                            val fastDecoder = FastVideoFrameDecoder(
                                videoFile = inputCopy,
                                rotationDegrees = metadata.rotationDegrees
                            )
                            val count = fastDecoder.decodeAllFrames(
                                isCancelled = { cancelled.get() || !isActive },
                                onFrame = { idx, bgrMat, _ ->
                                    sentAny = true
                                    decodedCountAtomic.set(idx + 1)
                                    val sendOk = frameChannel.trySend(DecodedFrame(index = idx, bgrMat = bgrMat)).isSuccess
                                    if (!sendOk) {
                                        // Backpressure via blocking send inside coroutine isn't inline here; switch cleanly:
                                        kotlinx.coroutines.runBlocking {
                                            frameChannel.send(DecodedFrame(index = idx, bgrMat = bgrMat))
                                        }
                                    }
                                }
                            )
                            if (count > 0) {
                                mediaCodecSucceeded = true
                                logger.decoderBackend = "MediaCodec Zero-Copy YUV->BGR"
                                logger.extractionMs = System.currentTimeMillis() - decodeStartMs
                                logger.info("Fast MediaCodec decoder yielded $count frames in ${logger.extractionMs}ms")
                            }
                        } catch (t: Throwable) {
                            if (cancelled.get()) {
                                frameChannel.close(InterruptedException("Cancelled by user"))
                                return@launch
                            }
                            if (!sentAny) {
                                logger.warn("MediaCodec zero-copy decoder failed (${t.message}); using FFmpeg fallback")
                            } else {
                                frameChannel.close(t)
                                return@launch
                            }
                        }

                        if (!mediaCodecSucceeded && !cancelled.get()) {
                            logger.decoderBackend = "FFmpegKit Fallback Extractor"
                            runFfmpegFallbackExtractor(
                                inputCopy = inputCopy,
                                framesDir = framesDir,
                                safeFps = safeFps,
                                availStorageMb = availStorageMb,
                                expectedTotalFrames = expectedTotalFrames,
                                capabilities = capabilities,
                                logger = logger,
                                frameChannel = frameChannel,
                                decodedCountAtomic = decodedCountAtomic,
                                onSessionCreated = { activeExtractSession = it },
                                onProgress = onProgress
                            )
                        } else {
                            frameChannel.close()
                        }
                    }

                    // Consumer: Frame enhancement -> AI tiles / GPU fallback -> Hardware encoder
                    val enhanceLoopStartMs = System.currentTimeMillis()
                    var lastThermalCheckMs = 0L
                    var currentThermalState = capabilities.thermalState

                    try {
                        for (item in frameChannel) {
                            if (cancelled.get()) {
                                item.bgrMat?.release()
                                item.diskFile?.delete()
                                throw InterruptedException("Cancelled by user")
                            }

                            val frameIndex = item.index
                            val dynamicTotal = max(expectedTotalFrames, max(decodedCountAtomic.get(), frameIndex + 1))

                            // Periodic thermal & RAM check
                            val now = System.currentTimeMillis()
                            if (now - lastThermalCheckMs >= 800L || frameIndex == 0) {
                                val thermalPair = capabilityDetector.readThermalInfo()
                                currentThermalState = thermalPair.first
                                lastThermalCheckMs = now

                                val freeRamMb = capabilityDetector.currentAvailableRamMb()
                                if (freeRamMb < 220L) {
                                    System.gc()
                                    delay(30L)
                                    val postGcRam = capabilityDetector.currentAvailableRamMb()
                                    if (postGcRam < 180L) {
                                        item.bgrMat?.release()
                                        item.diskFile?.delete()
                                        throw IllegalStateException("Low device memory (${postGcRam}MB free) during ${mode.label} enhancement.")
                                    }
                                }
                            }

                            if (currentThermalState == ThermalState.WARM) {
                                delay(capabilities.throttleDelayMsOnWarm)
                            } else if (currentThermalState >= ThermalState.THROTTLING) {
                                delay(capabilities.throttleDelayMsOnHot)
                            }

                            val bgr = item.bgrMat ?: item.diskFile?.let { f -> Imgcodecs.imread(f.absolutePath) } ?: Mat()
                            if (bgr.empty()) {
                                bgr.release()
                                item.diskFile?.delete()
                                recoveredFrames++
                                logger.warn("Recovered empty/corrupted frame #$frameIndex")
                                if (frameIndex > 0) {
                                    activeEncoder.encodeFrame(outputBitmap, frameIndex)
                                }
                                continue
                            }

                            val baseFrameProgress = 0.20f + (frameIndex.toFloat() / dynamicTotal.toFloat()) * 0.70f
                            val frameSlice = 0.70f / dynamicTotal.toFloat()

                            if (frameIndex == 0 || frameIndex % 3 == 0) {
                                onProgress(
                                    ProcessingProgress(
                                        stage = ProcessingStage.ENHANCING,
                                        processedFrames = frameIndex,
                                        totalFrames = dynamicTotal,
                                        overallProgress = baseFrameProgress.coerceIn(0.20f, 0.89f),
                                        statusDetail = "Enhancing frame ${frameIndex + 1}/$dynamicTotal",
                                        thermalState = currentThermalState,
                                        activeAcceleration = activeAccelLabel,
                                        memoryUsageMb = logger.sampleHeapUsedMb(),
                                        speedFallbackActive = lightweightFallbackActive
                                    )
                                )
                            }

                            val frameProcStartMs = System.currentTimeMillis()
                            try {
                                val frameResult = frameProcessor.processFrame(
                                    srcBgr = bgr,
                                    outDims = outDims,
                                    params = adaptiveParams,
                                    capabilities = capabilities,
                                    reusableOutBitmap = outputBitmap,
                                    estimatedTotalFrames = dynamicTotal,
                                    forceLightweightFallback = lightweightFallbackActive,
                                    tileSizeOverride = if (presetTileSize > 0) presetTileSize else null,
                                    logger = logger,
                                    isCancelled = { cancelled.get() },
                                    onUpscaleTileProgress = { tileDone, totalTiles, usedTile ->
                                        val frac = tileDone.toFloat() / totalTiles.coerceAtLeast(1).toFloat()
                                        val tileOverall = baseFrameProgress + frac * frameSlice
                                        onProgress(
                                            ProcessingProgress(
                                                stage = ProcessingStage.ENHANCING,
                                                processedFrames = frameIndex,
                                                totalFrames = dynamicTotal,
                                                overallProgress = tileOverall.coerceIn(0.20f, 0.90f),
                                                statusDetail = "${mode.label} • Frame ${frameIndex + 1}/$dynamicTotal (${outDims.label})",
                                                thermalState = currentThermalState,
                                                activeAcceleration = activeAccelLabel,
                                                memoryUsageMb = logger.sampleHeapUsedMb(),
                                                tileProgressLabel = "Tile $tileDone/$totalTiles (${usedTile}px)",
                                                speedFallbackActive = lightweightFallbackActive
                                            )
                                        )
                                    }
                                )
                                totalFacesEnhanced += frameResult.facesEnhanced
                                if (frameResult.lightweightFallbackTriggered && !lightweightFallbackActive) {
                                    lightweightFallbackActive = true
                                    logger.speedFallbackEngaged = true
                                    logger.speedFallbackReason = frameResult.fallbackReason
                                }
                                if (frameResult.activeAccelerationLabel.isNotBlank()) {
                                    activeAccelLabel = frameResult.activeAccelerationLabel
                                }
                                activeEncoder.encodeFrame(outputBitmap, frameIndex)
                            } catch (e: InterruptedException) {
                                throw e
                            } catch (oom: OutOfMemoryError) {
                                logger.error("OutOfMemoryError at frame $frameIndex (${outDims.label})", oom)
                                throw IllegalStateException("Device memory exhausted during ${outDims.label} processing.", oom)
                            } catch (t: Throwable) {
                                logger.error("Frame $frameIndex processing error", t)
                                recoveredFrames++
                                if (frameIndex > 0) {
                                    activeEncoder.encodeFrame(outputBitmap, frameIndex)
                                } else {
                                    throw t
                                }
                            } finally {
                                bgr.release()
                                item.diskFile?.delete()
                                logger.enhancementMs.addAndGet(System.currentTimeMillis() - frameProcStartMs)
                            }

                            processedCount++
                            val elapsedLoopMs = (System.currentTimeMillis() - enhanceLoopStartMs).coerceAtLeast(1L)
                            val msPerFrame = elapsedLoopMs.toDouble() / processedCount.toDouble()
                            val fpsSpeed = 1000.0 / msPerFrame.coerceAtLeast(1.0)
                            val remainingSec = (((dynamicTotal - processedCount).coerceAtLeast(0) * msPerFrame) / 1000.0)
                                .roundToInt()
                                .toLong()
                            val doneProgress = 0.20f + (processedCount.toFloat() / dynamicTotal.toFloat()) * 0.70f

                            onProgress(
                                ProcessingProgress(
                                    stage = ProcessingStage.ENHANCING,
                                    processedFrames = processedCount,
                                    totalFrames = dynamicTotal,
                                    overallProgress = doneProgress.coerceIn(0.20f, 0.90f),
                                    etaSeconds = remainingSec,
                                    statusDetail = buildString {
                                        append("Frame $processedCount / $dynamicTotal • ${outDims.label}")
                                        if (recoveredFrames > 0) {
                                            append(" • $recoveredFrames recovered")
                                        }
                                    },
                                    thermalState = currentThermalState,
                                    activeAcceleration = activeAccelLabel,
                                    processingSpeedLabel = if (fpsSpeed >= 1.0) {
                                        String.format(Locale.US, "%.1f fps", fpsSpeed)
                                    } else {
                                        "${msPerFrame.roundToInt()} ms/f"
                                    },
                                    memoryUsageMb = logger.sampleHeapUsedMb(),
                                    speedFallbackActive = lightweightFallbackActive
                                )
                            )
                        }
                    } finally {
                        // Drain any remaining queued native Mats to prevent native leaks on cancel/error
                        while (true) {
                            val leftover = frameChannel.tryReceive().getOrNull() ?: break
                            leftover.bgrMat?.release()
                            leftover.diskFile?.delete()
                        }
                    }

                    producerJob.join()
                }

                if (cancelled.get()) throw InterruptedException("Cancelled by user")
                if (processedCount == 0) {
                    throw IllegalStateException("Zero video frames were processed.")
                }

                // 6. Finalize Video Encoding
                onProgress(
                    ProcessingProgress(
                        stage = ProcessingStage.ENCODING,
                        processedFrames = processedCount,
                        totalFrames = processedCount,
                        overallProgress = 0.92f,
                        statusDetail = "Finalizing ${activeEncoder.activeCodecLabel} stream...",
                        thermalState = capabilities.thermalState,
                        activeAcceleration = activeAccelLabel,
                        memoryUsageMb = logger.sampleHeapUsedMb(),
                        speedFallbackActive = lightweightFallbackActive
                    )
                )
                activeEncoder.finish()
                val usedCodecLabel = activeEncoder.activeCodecLabel
                logger.encodingMs.set(activeEncoder.totalEncodeTimeMs)
                encoder = null

                // 7. Preserve Original Audio & Finalize Container
                onProgress(
                    ProcessingProgress(
                        stage = ProcessingStage.FINALIZING,
                        processedFrames = processedCount,
                        totalFrames = processedCount,
                        overallProgress = 0.96f,
                        statusDetail = if (metadata.hasAudio) "Preserving original audio track..." else "Finalizing MP4 output...",
                        thermalState = capabilities.thermalState,
                        activeAcceleration = activeAccelLabel,
                        memoryUsageMb = logger.sampleHeapUsedMb(),
                        speedFallbackActive = lightweightFallbackActive
                    )
                )

                val tMux0 = System.currentTimeMillis()
                muxWithOriginalAudio(rawVideoMp4, inputCopy, finalMp4, metadata.hasAudio, logger)
                logger.muxingMs = System.currentTimeMillis() - tMux0

                if (!finalMp4.exists() || finalMp4.length() == 0L) {
                    throw IllegalStateException("Export failed: output MP4 could not be written.")
                }

                val totalElapsed = System.currentTimeMillis() - startWallMs
                val diagnostics = logger.buildReport(
                    totalMs = totalElapsed,
                    executionProvider = activeAccelLabel,
                    benchmarkSummary = onnxEngine.benchmarkSummary,
                    codecUsed = usedCodecLabel,
                    initialTileSize = presetTileSize,
                    finalTileSize = if (lightweightFallbackActive) 0 else presetTileSize,
                    framesProcessed = processedCount,
                    recoveredFrames = recoveredFrames,
                    analysisReusedFromCache = analysisOutcome.reusedFromCache
                )

                logger.info(
                    "Job completed in ${totalElapsed}ms • frames=$processedCount • " +
                        "decoder=${logger.decoderBackend} • init=${diagnostics.modelInitMs}ms • " +
                        "ai=${diagnostics.aiInferenceMs}ms • encode=${diagnostics.encodingMs}ms • " +
                        "peakMem=${diagnostics.peakMemoryMb}MB • provider=${diagnostics.executionProvider}"
                )

                onProgress(
                    ProcessingProgress(
                        stage = ProcessingStage.COMPLETED,
                        processedFrames = processedCount,
                        totalFrames = processedCount,
                        overallProgress = 1f,
                        etaSeconds = 0L,
                        statusDetail = "Completed in ${formatElapsed(totalElapsed)} (${outDims.label})",
                        thermalState = capabilities.thermalState,
                        activeAcceleration = activeAccelLabel,
                        memoryUsageMb = diagnostics.peakMemoryMb,
                        speedFallbackActive = lightweightFallbackActive
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
                    usedAiSuperResolution = outDims.needsAiSuperResolution && !lightweightFallbackActive && onnxEngine.isRealEsrganReady,
                    usedAiDenoise = adaptiveParams.useAiDenoise && onnxEngine.isDncnnReady,
                    diagnostics = diagnostics
                )
            } catch (interrupted: InterruptedException) {
                try {
                    activeExtractSession?.let { FFmpegKit.cancel(it.sessionId) }
                } catch (_: Exception) {
                }
                encoder?.abort()
                finalMp4.delete()
                throw interrupted
            } catch (t: Throwable) {
                logger.error("Pipeline failed", t)
                try {
                    activeExtractSession?.let { FFmpegKit.cancel(it.sessionId) }
                } catch (_: Exception) {
                }
                encoder?.abort()
                finalMp4.delete()
                throw t
            } finally {
                reusableBitmap?.recycle()
                frameProcessor.resetTemporalState()
                onnxEngine.releaseWorkBuffers()
                workDir.deleteRecursively()
            }
        }
    }

    private suspend fun runFfmpegFallbackExtractor(
        inputCopy: File,
        framesDir: File,
        safeFps: Double,
        availStorageMb: Long,
        expectedTotalFrames: Int,
        capabilities: DeviceCapabilities,
        logger: EngineDiagnosticsLogger,
        frameChannel: Channel<DecodedFrame>,
        decodedCountAtomic: AtomicInteger,
        onSessionCreated: (FFmpegSession) -> Unit,
        onProgress: (ProcessingProgress) -> Unit
    ) {
        val ext = "jpg"
        val framePattern = File(framesDir, "frame_%06d.$ext").absolutePath
        val extractCmd = "-y -i \"${inputCopy.absolutePath}\" -vf \"fps=${safeFps}\" -qscale:v 2 -qmin 1 -qmax 3 \"$framePattern\""

        val extractionStartMs = System.currentTimeMillis()
        val extractionDone = AtomicBoolean(false)
        val extractionFailure = AtomicReference<Throwable?>(null)
        val lastActivityMs = AtomicLong(System.currentTimeMillis())

        val session = FFmpegKit.executeAsync(
            extractCmd,
            { finished ->
                lastActivityMs.set(System.currentTimeMillis())
                if (!ReturnCode.isSuccess(finished.returnCode) && !cancelled.get()) {
                    val errLog = finished.logsAsString?.takeLast(600).orEmpty()
                    val msg = "FFmpeg frame extraction failed (code ${finished.returnCode}): $errLog"
                    logger.error(msg)
                    extractionFailure.compareAndSet(null, IllegalStateException(msg))
                }
                logger.extractionMs = System.currentTimeMillis() - extractionStartMs
                extractionDone.set(true)
            },
            { _ -> lastActivityMs.set(System.currentTimeMillis()) },
            { stats ->
                lastActivityMs.set(System.currentTimeMillis())
                if (stats.videoFrameNumber > 0) {
                    decodedCountAtomic.set(stats.videoFrameNumber)
                }
            }
        )
        onSessionCreated(session)

        var nextFileNum = 1
        var stalledEmitted = false
        try {
            while (!cancelled.get()) {
                extractionFailure.get()?.let { throw it }
                val currentFile = File(framesDir, String.format(Locale.US, "frame_%06d.%s", nextFileNum, ext))
                val nextFile = File(framesDir, String.format(Locale.US, "frame_%06d.%s", nextFileNum + 1, ext))

                val fileReady = currentFile.exists() && currentFile.length() > 0L &&
                    (nextFile.exists() || extractionDone.get())

                if (fileReady) {
                    lastActivityMs.set(System.currentTimeMillis())
                    stalledEmitted = false
                    frameChannel.send(DecodedFrame(index = nextFileNum - 1, diskFile = currentFile))
                    nextFileNum++
                } else if (extractionDone.get()) {
                    if (currentFile.exists() && currentFile.length() > 0L) {
                        frameChannel.send(DecodedFrame(index = nextFileNum - 1, diskFile = currentFile))
                        nextFileNum++
                    } else {
                        break
                    }
                } else {
                    val idleMs = System.currentTimeMillis() - lastActivityMs.get()
                    if (idleMs > 25_000L) {
                        val stallErr = IllegalStateException("FFmpeg extraction stalled for ${idleMs / 1000}s without new frames.")
                        logger.error(stallErr.message ?: "FFmpeg stalled", stallErr)
                        try {
                            FFmpegKit.cancel(session.sessionId)
                        } catch (_: Exception) {
                        }
                        throw stallErr
                    }
                    if (!stalledEmitted && nextFileNum == 1 && idleMs > 600L) {
                        val extFrames = decodedCountAtomic.get()
                        onProgress(
                            ProcessingProgress(
                                stage = ProcessingStage.EXTRACTING,
                                processedFrames = extFrames,
                                totalFrames = expectedTotalFrames,
                                overallProgress = 0.17f,
                                statusDetail = "Extracting frame stream ($extFrames / ~$expectedTotalFrames)...",
                                thermalState = capabilities.thermalState,
                                activeAcceleration = onnxEngine.providerLabel,
                                memoryUsageMb = logger.sampleHeapUsedMb()
                            )
                        )
                        stalledEmitted = true
                    }
                    delay(15L)
                }
            }
            if (nextFileNum == 1 && !cancelled.get()) {
                extractionFailure.get()?.let { throw it }
                throw IllegalStateException("No decodable video frames found in source file.")
            }
            frameChannel.close()
        } catch (t: Throwable) {
            frameChannel.close(t)
        }
    }

    private fun muxWithOriginalAudio(
        rawVideoMp4: File,
        originalInput: File,
        finalMp4: File,
        hasAudio: Boolean,
        logger: EngineDiagnosticsLogger
    ) {
        if (!hasAudio) {
            val cmd = "-y -i \"${rawVideoMp4.absolutePath}\" -c:v copy -movflags +faststart \"${finalMp4.absolutePath}\""
            val session = FFmpegKit.execute(cmd)
            if (!ReturnCode.isSuccess(session.returnCode) || !finalMp4.exists() || finalMp4.length() == 0L) {
                logger.warn("FFmpeg faststart copy failed (${session.returnCode}); falling back to direct MP4 copy")
                rawVideoMp4.copyTo(finalMp4, overwrite = true)
            }
            return
        }

        val copyCmd = "-y -i \"${rawVideoMp4.absolutePath}\" -i \"${originalInput.absolutePath}\" " +
            "-map 0:v:0 -map 1:a:0 -c:v copy -c:a copy -shortest -movflags +faststart \"${finalMp4.absolutePath}\""
        val copySession = FFmpegKit.execute(copyCmd)
        if (ReturnCode.isSuccess(copySession.returnCode) && finalMp4.exists() && finalMp4.length() > 0L) {
            return
        }

        if (finalMp4.exists()) finalMp4.delete()
        val aacCmd = "-y -i \"${rawVideoMp4.absolutePath}\" -i \"${originalInput.absolutePath}\" " +
            "-map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -b:a 192k -shortest -movflags +faststart \"${finalMp4.absolutePath}\""
        val aacSession = FFmpegKit.execute(aacCmd)
        if (!ReturnCode.isSuccess(aacSession.returnCode) || !finalMp4.exists() || finalMp4.length() == 0L) {
            logger.warn("Audio muxing failed (${aacSession.returnCode}); saving video stream without audio track")
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
