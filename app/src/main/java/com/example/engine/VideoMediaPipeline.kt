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
import kotlin.math.roundToLong

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
        val diskFile: File? = null,
        val ptsUs: Long = -1L,
        val fromDecoderPool: Boolean = false
    )

    private val cancelled = AtomicBoolean(false)

    fun cancel() {
        cancelled.set(true)
        try {
            FFmpegKit.cancel()
        } catch (_: Throwable) {
        }
    }

    suspend fun initializeAndCheckHealth(
        capabilities: DeviceCapabilities,
        forceRebenchmark: Boolean = false
    ): EngineHealthStatus =
        withContext(Dispatchers.IO) {
            val logger = EngineDiagnosticsLogger()
            val cvReady = frameProcessor.ensureInitialized()
            var ortReady = false
            var diag: String? = null
            try {
                onnxEngine.initialize(capabilities, logger, forceRebenchmark = forceRebenchmark)
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
                hardwareProfile = onnxEngine.adaptiveProfile,
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
        var outputBitmapA: Bitmap? = null
        var outputBitmapB: Bitmap? = null
        var activeExtractSession: FFmpegSession? = null
        var activeFastDecoder: FastVideoFrameDecoder? = null

        runCatching {
            try {
                check(frameProcessor.ensureInitialized()) { "OpenCV native engine failed to initialize." }
                frameProcessor.resetTemporalState()
                onnxEngine.resetTemporalCaches()

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
                    timeoutMs = 6_000L,
                    logger = logger
                )
                logger.analysisMs = analysisOutcome.durationMs
                val qualityAnalysis = analysisOutcome.analysis
                if (cancelled.get()) throw InterruptedException("Cancelled by user")

                // 3. Adaptive Adjustments
                val adaptiveParams = smartAdjuster.computeAdaptiveAdjustments(profile, qualityAnalysis)

                // 4. Adaptive Hardware Acceleration Benchmark & Model Initialization
                if (outDims.needsAiSuperResolution || !onnxEngine.isGlesGpuReady || !onnxEngine.isRealEsrganReady) {
                    onProgress(
                        ProcessingProgress(
                            stage = ProcessingStage.LOADING_MODEL,
                            totalFrames = metadata.estimatedFrameCount,
                            overallProgress = 0.12f,
                            statusDetail = if (outDims.needsAiSuperResolution) {
                                "Optimizing GPU / NPU / CPU per model (${outDims.aiTileSize}px × ${outDims.tilesPerFrame} tiles)..."
                            } else {
                                "Initializing adaptive GPU / NPU / OpenCV acceleration..."
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
                val medianStepUs = (1_000_000.0 / safeFps).roundToLong().coerceAtLeast(8_333L)

                val srLatencyMs = onnxEngine.adaptiveProfile
                    ?.plans
                    ?.get(com.example.model.ModelKind.REAL_ESRGAN)
                    ?.measuredLatencyMs
                    ?: 90L
                val dnLatencyMs = onnxEngine.adaptiveProfile
                    ?.plans
                    ?.get(com.example.model.ModelKind.DNCNN)
                    ?.measuredLatencyMs
                    ?: 10L
                val presetTileSize = outDims.aiTileSize
                val budgetGovernor = SpeedBudgetGovernor(
                    mode = mode,
                    durationMs = metadata.durationMs,
                    estimatedFrames = metadata.estimatedFrameCount,
                    presetTileSize = presetTileSize,
                    presetTilesCap = outDims.tilesPerFrame,
                    measuredSrTileMs = srLatencyMs,
                    deviceCapacityScore = capabilities.processingCapacityScore,
                    measuredDnCnnMs = dnLatencyMs
                )
                val modeBudgetFrameMs = budgetGovernor.targetMsPerFrame.roundToLong().coerceAtLeast(15L)

                val cadenceInterpolate = MotionSmoothMath.shouldInterpolateCadence(
                    sourceFps = safeFps,
                    allowInterpolation = adaptiveParams.allowFrameInterpolation && budgetGovernor.allowInterpolationByBaseline()
                )
                val encoderFps = MotionSmoothMath.targetOutputFps(safeFps, cadenceInterpolate)

                val activeEncoder = HardwareVideoEncoder(
                    outputFile = rawVideoMp4,
                    width = outDims.width,
                    height = outDims.height,
                    fps = encoderFps,
                    bitrateBps = outDims.targetBitrateBps,
                    preferH265 = outDims.useH265
                ).apply { start() }
                encoder = activeEncoder

                // Ping-pong double-buffered Bitmaps overlap AI Enhance(N+1) with Hardware Encode(N)
                val bmpA = Bitmap.createBitmap(outDims.width, outDims.height, Bitmap.Config.ARGB_8888)
                val bmpB = Bitmap.createBitmap(outDims.width, outDims.height, Bitmap.Config.ARGB_8888)
                outputBitmapA = bmpA
                outputBitmapB = bmpB
                var lastCompletedBitmap = bmpA

                onProgress(
                    ProcessingProgress(
                        stage = ProcessingStage.EXTRACTING,
                        totalFrames = metadata.estimatedFrameCount,
                        overallProgress = 0.16f,
                        statusDetail = "Decoding video stream via zero-copy NV12/I420 MediaCodec...",
                        thermalState = capabilities.thermalState,
                        activeAcceleration = onnxEngine.providerLabel,
                        memoryUsageMb = logger.sampleHeapUsedMb()
                    )
                )

                logger.info(
                    "SpeedBudgetGovernor: mode=${mode.label} • max=${budgetGovernor.maxBudgetMs}ms • " +
                        "target=${String.format(Locale.US, "%.1f", budgetGovernor.targetMsPerFrame)}ms/f • " +
                        "tier=${budgetGovernor.activeTier.label} (k=${budgetGovernor.currentKeyframeInterval}) • " +
                        "smooth=${if (cadenceInterpolate) "Flow 2x (${encoderFps.roundToInt()}fps)" else "VFR Stabilized (${safeFps.roundToInt()}fps)"}"
                )

                val channelCap = if (capabilities.availableRamMb > 1400L) 4 else 3
                val frameChannel = Channel<DecodedFrame>(capacity = channelCap)

                var totalFacesEnhanced = 0
                var usedAiDenoiseAny = false
                var dncnnFramesProcessed = 0
                var dncnnAsyncPrefetchHits = 0
                var recoveredFrames = 0
                var processedCount = 0
                var duplicateFramesReused = 0
                var interpolatedFramesCount = 0
                var stutterRepairsCount = 0
                var colorEnhancedFramesCount = 0
                var lightweightFallbackActive = false
                var activeAccelLabel = onnxEngine.providerLabel
                val expectedTotalFrames = metadata.estimatedFrameCount.coerceAtLeast(1)
                val decodedCountAtomic = AtomicInteger(0)

                coroutineScope {
                    // Stage 1 Producer: Pooled Zero-Disk NV12/I420 MediaCodec decoder, automatic FFmpeg fallback
                    val producerJob = launch(Dispatchers.IO) {
                        val decodeStartMs = System.currentTimeMillis()
                        var mediaCodecSucceeded = false
                        var sentAny = false

                        try {
                            val fastDecoder = FastVideoFrameDecoder(
                                videoFile = inputCopy,
                                rotationDegrees = metadata.rotationDegrees
                            ).also { activeFastDecoder = it }

                            val count = fastDecoder.decodeAllFrames(
                                isCancelled = { cancelled.get() || !isActive },
                                onFrame = { idx, bgrMat, ptsUs ->
                                    sentAny = true
                                    decodedCountAtomic.set(idx + 1)
                                    val frameMsg = DecodedFrame(
                                        index = idx,
                                        bgrMat = bgrMat,
                                        ptsUs = ptsUs,
                                        fromDecoderPool = true
                                    )
                                    val sendOk = frameChannel.trySend(frameMsg).isSuccess
                                    if (!sendOk) {
                                        kotlinx.coroutines.runBlocking {
                                            frameChannel.send(frameMsg)
                                        }
                                    }
                                }
                            )
                            if (count > 0) {
                                mediaCodecSucceeded = true
                                logger.decoderBackend = "MediaCodec NV12/I420 Pooled"
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

                    // Stage 2 & 3 Overlapped Consumer: Motion Smoothness + AI Enhance(N) -> Async HW Encode
                    val enhanceLoopStartMs = System.currentTimeMillis()
                    var lastThermalCheckMs = 0L
                    var lastProgressEmitMs = 0L
                    var currentThermalState = capabilities.thermalState
                    var encodeSlotIndex = 0
                    var prevRawBgr: Mat? = null
                    var prevFramePtsUs: Long = -1L
                    val interpScratchBgr = if (adaptiveParams.enableSmoothness) Mat() else null
                    var pendingDnCnnPrefetch: OnnxAiInferenceEngine.DnCnnPrefetchToken? = null
                    var queuedNextItem: DecodedFrame? = frameChannel.receiveCatching().getOrNull()

                    try {
                        while (queuedNextItem != null) {
                            val item = queuedNextItem
                            queuedNextItem = frameChannel.tryReceive().getOrNull()
                            if (cancelled.get()) {
                                if (item.fromDecoderPool) activeFastDecoder?.recycleMat(item.bgrMat) else item.bgrMat?.release()
                                item.diskFile?.delete()
                                throw InterruptedException("Cancelled by user")
                            }

                            val frameIndex = item.index
                            val dynamicTotal = max(expectedTotalFrames, max(decodedCountAtomic.get(), frameIndex + 1))
                            val rawPts = if (item.ptsUs >= 0L) item.ptsUs else frameIndex.toLong() * medianStepUs
                            val currPtsUs = if (prevFramePtsUs < 0L) rawPts else max(rawPts, prevFramePtsUs + 2_000L)

                            // Periodic thermal watchdog + adaptive backend step-down only under severe heat
                            val now = System.currentTimeMillis()
                            if (now - lastThermalCheckMs >= 1_200L || frameIndex == 0) {
                                val thermalPair = capabilityDetector.readThermalInfo()
                                currentThermalState = thermalPair.first
                                lastThermalCheckMs = now
                                onnxEngine.handleThermalPressure(currentThermalState, logger)

                                val freeRamMb = capabilityDetector.currentAvailableRamMb()
                                if (freeRamMb < 200L) {
                                    System.gc()
                                    delay(20L)
                                    val postGcRam = capabilityDetector.currentAvailableRamMb()
                                    if (postGcRam < 160L) {
                                        if (item.fromDecoderPool) activeFastDecoder?.recycleMat(item.bgrMat) else item.bgrMat?.release()
                                        item.diskFile?.delete()
                                        throw IllegalStateException("Low device memory (${postGcRam}MB free) during ${mode.label} enhancement.")
                                    }
                                }
                            }

                            if (currentThermalState >= ThermalState.THROTTLING && frameIndex > 0 && frameIndex % 8 == 0) {
                                delay(if (currentThermalState == ThermalState.CRITICAL) 24L else 12L)
                            }

                            val bgr = item.bgrMat ?: item.diskFile?.let { f -> Imgcodecs.imread(f.absolutePath) } ?: Mat()
                            if (bgr.empty()) {
                                bgr.release()
                                item.diskFile?.delete()
                                recoveredFrames++
                                logger.warn("Recovered empty/corrupted frame #$frameIndex")
                                if (frameIndex > 0) {
                                    activeEncoder.encodeFrameWithPts(lastCompletedBitmap, currPtsUs)
                                    prevFramePtsUs = currPtsUs
                                }
                                continue
                            }

                            val baseFrameProgress = 0.20f + (frameIndex.toFloat() / dynamicTotal.toFloat()) * 0.70f
                            val frameSlice = 0.70f / dynamicTotal.toFloat()

                            val speedDecision = budgetGovernor.decisionForFrame(
                                frameIndex = frameIndex,
                                isSceneCut = false,
                                forceLightweight = lightweightFallbackActive
                            )

                            val frameProcStartMs = System.currentTimeMillis()
                            try {
                                // Smoothness Enhancement: VFR dropped-frame repair & optional 2x optical-flow interpolation
                                val prevMat = prevRawBgr
                                if (adaptiveParams.enableSmoothness && frameIndex > 0 && prevMat != null && interpScratchBgr != null) {
                                    val droppedGap = MotionSmoothMath.detectDroppedFrameGap(prevFramePtsUs, currPtsUs, medianStepUs)
                                    val needMidpoint = (droppedGap > 0) || (cadenceInterpolate && budgetGovernor.allowInterpolationByBaseline())
                                    if (needMidpoint) {
                                        val flowOk = frameProcessor.motionEngine.interpolateMidpointBgr(
                                            prevBgr = prevMat,
                                            nextBgr = bgr,
                                            outBgr = interpScratchBgr,
                                            alpha = 0.5
                                        )
                                        if (flowOk || droppedGap > 0) {
                                            val midPtsUs = prevFramePtsUs + ((currPtsUs - prevFramePtsUs) / 2L).coerceAtLeast(1_000L)
                                            val interpBitmap = if ((encodeSlotIndex and 1) == 0) bmpA else bmpB
                                            encodeSlotIndex++
                                            val interpDecision = speedDecision.copy(allowFaceDetect = false, allowDnCnn = true)
                                            val interpResult = frameProcessor.processFrame(
                                                srcBgr = interpScratchBgr,
                                                outDims = outDims,
                                                params = adaptiveParams,
                                                capabilities = capabilities,
                                                reusableOutBitmap = interpBitmap,
                                                estimatedTotalFrames = dynamicTotal,
                                                forceLightweightFallback = lightweightFallbackActive,
                                                tileSizeOverride = if (presetTileSize > 0) presetTileSize else null,
                                                frameIndex = frameIndex,
                                                logger = logger,
                                                isCancelled = { cancelled.get() },
                                                speedDecision = interpDecision,
                                                modeBudgetFrameMs = modeBudgetFrameMs
                                            )
                                            if (interpResult.usedAiDenoise) {
                                                usedAiDenoiseAny = true
                                                dncnnFramesProcessed++
                                            }
                                            if (interpResult.dncnnInferenceMs > 0L) {
                                                logger.dncnnInferenceMs.addAndGet(interpResult.dncnnInferenceMs)
                                            }
                                            lastCompletedBitmap = interpBitmap
                                            activeEncoder.enqueueFrameAsyncWithPts(interpBitmap, midPtsUs)
                                            interpolatedFramesCount++
                                            colorEnhancedFramesCount++
                                            if (droppedGap > 0) stutterRepairsCount++
                                        }
                                    }
                                }

                                val targetBitmap = if ((encodeSlotIndex and 1) == 0) bmpA else bmpB
                                encodeSlotIndex++

                                val currentPrefetchToken = pendingDnCnnPrefetch
                                pendingDnCnnPrefetch = null

                                val frameResult = frameProcessor.processFrame(
                                    srcBgr = bgr,
                                    outDims = outDims,
                                    params = adaptiveParams,
                                    capabilities = capabilities,
                                    reusableOutBitmap = targetBitmap,
                                    estimatedTotalFrames = dynamicTotal,
                                    forceLightweightFallback = lightweightFallbackActive,
                                    tileSizeOverride = if (presetTileSize > 0) presetTileSize else null,
                                    frameIndex = frameIndex,
                                    logger = logger,
                                    isCancelled = { cancelled.get() },
                                    onUpscaleTileProgress = { tileDone, totalTiles, usedTile ->
                                        val nowTileMs = System.currentTimeMillis()
                                        if (nowTileMs - lastProgressEmitMs >= 250L) {
                                            lastProgressEmitMs = nowTileMs
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
                                    },
                                    speedDecision = speedDecision,
                                    modeBudgetFrameMs = modeBudgetFrameMs,
                                    prefetchedDnCnnToken = currentPrefetchToken,
                                    onDnCnnStageComplete = {
                                        if (queuedNextItem == null) {
                                            queuedNextItem = frameChannel.tryReceive().getOrNull()
                                        }
                                        val nextBgr = queuedNextItem?.bgrMat
                                        if (adaptiveParams.useAiDenoise &&
                                            nextBgr != null &&
                                            !nextBgr.empty() &&
                                            !cancelled.get()
                                        ) {
                                            pendingDnCnnPrefetch = onnxEngine.prefetchDnCnnAsync(
                                                frameIndex = queuedNextItem!!.index,
                                                srcBgr = nextBgr,
                                                strength = adaptiveParams.dncnnStrength,
                                                logger = logger
                                            )
                                        }
                                    }
                                )
                                totalFacesEnhanced += frameResult.facesEnhanced
                                if (frameResult.usedAiDenoise) {
                                    usedAiDenoiseAny = true
                                    dncnnFramesProcessed++
                                }
                                if (frameResult.usedAsyncDnCnnPrefetch) dncnnAsyncPrefetchHits++
                                if (frameResult.usedDuplicateFrameReuse) duplicateFramesReused++
                                if (frameResult.colorEnhanced) colorEnhancedFramesCount++
                                if (frameResult.dncnnInferenceMs > 0L) logger.dncnnInferenceMs.addAndGet(frameResult.dncnnInferenceMs)
                                if (frameResult.faceAiMs > 0L) logger.faceAiMs.addAndGet(frameResult.faceAiMs)
                                if (frameResult.lightweightFallbackTriggered && !lightweightFallbackActive && speedDecision.runNeuralSrNow) {
                                    lightweightFallbackActive = true
                                    logger.speedFallbackEngaged = true
                                    logger.speedFallbackReason = frameResult.fallbackReason
                                }
                                if (frameResult.activeAccelerationLabel.isNotBlank()) {
                                    activeAccelLabel = frameResult.activeAccelerationLabel
                                }
                                lastCompletedBitmap = targetBitmap
                                activeEncoder.enqueueFrameAsyncWithPts(targetBitmap, currPtsUs)
                                prevFramePtsUs = currPtsUs

                                if (adaptiveParams.enableSmoothness) {
                                    val cachePrev = prevRawBgr
                                    if (cachePrev == null || cachePrev.cols() != bgr.cols() || cachePrev.rows() != bgr.rows()) {
                                        prevRawBgr?.release()
                                        prevRawBgr = bgr.clone()
                                    } else {
                                        bgr.copyTo(cachePrev)
                                    }
                                }
                            } catch (e: InterruptedException) {
                                throw e
                            } catch (oom: OutOfMemoryError) {
                                logger.error("OutOfMemoryError at frame $frameIndex (${outDims.label})", oom)
                                throw IllegalStateException("Device memory exhausted during ${outDims.label} processing.", oom)
                            } catch (t: Throwable) {
                                logger.error("Frame $frameIndex processing error", t)
                                recoveredFrames++
                                if (frameIndex > 0) {
                                    activeEncoder.encodeFrameWithPts(lastCompletedBitmap, currPtsUs)
                                    prevFramePtsUs = currPtsUs
                                } else {
                                    throw t
                                }
                            } finally {
                                if (item.fromDecoderPool) {
                                    activeFastDecoder?.recycleMat(bgr)
                                } else {
                                    bgr.release()
                                }
                                item.diskFile?.delete()
                                val frameElapsedMs = (System.currentTimeMillis() - frameProcStartMs).coerceAtLeast(1L)
                                logger.enhancementMs.addAndGet(frameElapsedMs)
                                budgetGovernor.onFrameCompleted(
                                    elapsedJobMs = System.currentTimeMillis() - startWallMs,
                                    processedFrames = processedCount + 1,
                                    totalFrames = dynamicTotal,
                                    lastFrameMs = frameElapsedMs
                                )
                            }

                            processedCount++
                            val nowEmitMs = System.currentTimeMillis()
                            if (nowEmitMs - lastProgressEmitMs >= 120L || processedCount == 1 || processedCount >= dynamicTotal) {
                                lastProgressEmitMs = nowEmitMs
                                val elapsedLoopMs = (nowEmitMs - enhanceLoopStartMs).coerceAtLeast(1L)
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
                                            append("Frame $processedCount / $dynamicTotal • ${outDims.label} • ${budgetGovernor.activeTier.label}")
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
                                        speedFallbackActive = lightweightFallbackActive || budgetGovernor.activeTier == SpeedExecutionTier.GPU_SHADER_FAST_PASS
                                    )
                                )
                            }

                            if (queuedNextItem == null) {
                                queuedNextItem = frameChannel.receiveCatching().getOrNull()
                            }
                        }
                    } finally {
                        pendingDnCnnPrefetch?.future?.let { runCatching { it.cancel(false) } }
                        pendingDnCnnPrefetch = null
                        queuedNextItem?.let { leftover ->
                            if (leftover.fromDecoderPool) {
                                activeFastDecoder?.recycleMat(leftover.bgrMat)
                            } else {
                                leftover.bgrMat?.release()
                            }
                            leftover.diskFile?.delete()
                        }
                        queuedNextItem = null
                        prevRawBgr?.release()
                        interpScratchBgr?.release()
                        while (true) {
                            val leftover = frameChannel.tryReceive().getOrNull() ?: break
                            if (leftover.fromDecoderPool) {
                                activeFastDecoder?.recycleMat(leftover.bgrMat)
                            } else {
                                leftover.bgrMat?.release()
                            }
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
                val scaleLabel = String.format(Locale.US, "%.1fx", outDims.targetScaleFactor)
                val smoothLabel = when {
                    cadenceInterpolate -> "DIS Optical Flow 2x (${encoderFps.roundToInt()}fps)"
                    stutterRepairsCount > 0 -> "DIS Optical Flow VFR Repair ($stutterRepairsCount)"
                    adaptiveParams.enableSmoothness -> "Flow + Micro-Jitter Stabilized"
                    else -> "Original Cadence"
                }
                val diagnostics = logger.buildReport(
                    totalMs = totalElapsed,
                    executionProvider = activeAccelLabel,
                    benchmarkSummary = onnxEngine.benchmarkSummary,
                    codecUsed = usedCodecLabel,
                    initialTileSize = presetTileSize,
                    finalTileSize = if (lightweightFallbackActive) 0 else presetTileSize,
                    framesProcessed = processedCount + interpolatedFramesCount,
                    recoveredFrames = recoveredFrames,
                    analysisReusedFromCache = analysisOutcome.reusedFromCache,
                    upscaleScaleLabel = scaleLabel,
                    perModelAccelerationSummary = onnxEngine.currentPerModelSummary(),
                    staticTilesReused = onnxEngine.totalStaticTilesReused,
                    framesReusedDuplicate = duplicateFramesReused,
                    runtimeFallbacksCount = onnxEngine.runtimeFallbacksCount,
                    interpolatedFramesCount = interpolatedFramesCount,
                    stutterRepairsCount = stutterRepairsCount,
                    colorEnhancedFrames = colorEnhancedFramesCount,
                    smoothnessModeLabel = smoothLabel,
                    dncnnFramesProcessed = dncnnFramesProcessed,
                    dncnnAsyncPrefetchHits = dncnnAsyncPrefetchHits
                )

                logger.info(
                    "Job completed in ${totalElapsed}ms (tier=${budgetGovernor.activeTier.label}, budget=${budgetGovernor.maxBudgetMs}ms) • " +
                        "frames=$processedCount (+$interpolatedFramesCount interp, $stutterRepairsCount VFR repairs) • " +
                        "dncnnFrames=$dncnnFramesProcessed (asyncHits=$dncnnAsyncPrefetchHits) • colorFrames=$colorEnhancedFramesCount • " +
                        "dupReuse=$duplicateFramesReused • staticTileReuse=${diagnostics.staticTilesReused} • " +
                        "fallbacks=${diagnostics.runtimeFallbacksCount} • scale=$scaleLabel (${outDims.label}) • " +
                        "sr=${diagnostics.aiInferenceMs}ms • dncnn=${diagnostics.dncnnInferenceMs}ms • encode=${diagnostics.encodingMs}ms • provider=${diagnostics.executionProvider}"
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
                    usedAiDenoise = usedAiDenoiseAny,
                    diagnostics = diagnostics
                )
            } catch (interrupted: InterruptedException) {
                try {
                    activeExtractSession?.let { FFmpegKit.cancel(it.sessionId) }
                } catch (_: Exception) {
                }
                encoder?.abort()
                encoder = null
                finalMp4.delete()
                throw interrupted
            } catch (t: Throwable) {
                logger.error("Pipeline failed", t)
                try {
                    activeExtractSession?.let { FFmpegKit.cancel(it.sessionId) }
                } catch (_: Exception) {
                }
                encoder?.abort()
                encoder = null
                finalMp4.delete()
                throw t
            } finally {
                encoder?.abort()
                outputBitmapA?.recycle()
                outputBitmapB?.recycle()
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
                    val idx = nextFileNum - 1
                    val estPts = ((idx.toLong() * 1_000_000.0) / safeFps).roundToLong()
                    frameChannel.send(DecodedFrame(index = idx, diskFile = currentFile, ptsUs = estPts))
                    nextFileNum++
                } else if (extractionDone.get()) {
                    if (currentFile.exists() && currentFile.length() > 0L) {
                        val idx = nextFileNum - 1
                        val estPts = ((idx.toLong() * 1_000_000.0) / safeFps).roundToLong()
                        frameChannel.send(DecodedFrame(index = idx, diskFile = currentFile, ptsUs = estPts))
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
