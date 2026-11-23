package com.example.engine

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import com.example.model.AdaptiveHardwareProfile
import com.example.model.BackendCandidateId
import com.example.model.DeviceCapabilities
import com.example.model.ModelKind
import com.example.model.ThermalState
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.concurrent.Callable
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.Future
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

data class TileProcessOutcome(
    val outputBgr: Mat,
    val usedLightweightFallback: Boolean,
    val fallbackReason: String? = null,
    val activeAccelerationLabel: String,
    val faceAiMs: Long = 0L,
    val staticTilesReused: Int = 0
)

class OnnxAiInferenceEngine(private val context: Context) {

    private val env: OrtEnvironment by lazy { OrtEnvironment.getEnvironment() }
    val gpuProcessor: GlesGpuImageProcessor = GlesGpuImageProcessor()
    val yuNetRunner: YuNetOnnxRunner by lazy { YuNetOnnxRunner(env) }
    private val hwAccelerator: AdaptiveHardwareAccelerator by lazy {
        AdaptiveHardwareAccelerator(context, env)
    }

    // Independent ranked fallback chains per AI model
    private var srRankedSessions: List<Pair<BackendCandidateId, OrtSession>> = emptyList()
    private var srActiveIndex: Int = 0
    private var dncnnRankedSessions: List<Pair<BackendCandidateId, OrtSession>> = emptyList()
    private var dncnnActiveIndex: Int = 0

    var adaptiveProfile: AdaptiveHardwareProfile? = null
        private set

    var runtimeFallbacksCount: Int = 0
        private set

    private var activeProviderDescription: String = "Initializing..."
    private var benchmarkDetails: String = "Not benchmarked"
    private var lastInitDurationMs: Long = 0L
    private var glesGpuAvailable = false

    // Double-buffered direct FloatBuffers for pipelined zero-GC Real-ESRGAN tile packing
    private var allocTileSize: Int = 0
    private var tileInBufA: FloatBuffer? = null
    private var tileInBufB: FloatBuffer? = null
    private var tileInPlaneTmpA: FloatArray = FloatArray(0)
    private var tileInPlaneTmpB: FloatArray = FloatArray(0)
    private var tileOutPlaneTmp: FloatArray = FloatArray(0)
    private var tilePrepExecutor: ExecutorService? = null

    // Direct off-heap FloatBuffer for DnCNN [1, 1, 256, 256]
    private val dncnnTileSize = 256
    private val dncnnTilePixels = dncnnTileSize * dncnnTileSize
    private var dncnnDirectInBuf: FloatBuffer? = null
    private var dncnnInTmpArr: FloatArray? = null
    private var dncnnOutArr: FloatArray? = null
    private var dncnnPadMat: Mat? = null
    private var dncnnFloatMat: Mat? = null
    private var dncnnOutTileMat: Mat? = null
    private var dncnnAccumMat: Mat? = null
    private var dncnnWeightMat: Mat? = null
    private var dncnnWeightedMat: Mat? = null
    private var dncnnU8Mat: Mat? = null

    // Static tile temporal reuse caches (100% quality preserving)
    private val srTileCache = StaticTileReuseCache()
    private val dncnnTileCache = StaticTileReuseCache()

    // Reusable OpenCV native C++ scratch Mats
    private var srcRgbMat: Mat? = null
    private var baseRgb8u: Mat? = null
    private var tilePaddedMatA: Mat? = null
    private var tilePaddedMatB: Mat? = null
    private var tileFloatHwcA: Mat? = null
    private var tileFloatHwcB: Mat? = null
    private var tileChannelsA: List<Mat>? = null
    private var tileChannelsB: List<Mat>? = null
    private var outFullPlaneMat: Mat? = null
    private var resizedOutChannels: List<Mat>? = null
    private var outResizedFloat: Mat? = null
    private var weightedTileFloat: Mat? = null
    private var accumRgbMat: Mat? = null
    private var accumWeightMat: Mat? = null
    private var invWeightSingleMat: Mat? = null
    private var invWeightThreeMat: Mat? = null
    private var weightClampFloorMat: Mat? = null
    private var outRgb8u: Mat? = null

    // Face ROI super-resolution buffers & canonical mask template (zero per-frame heap churn)
    private var faceSrFloatMat: Mat? = null
    private var faceMaskTemplate128: Mat? = null
    private var faceMaskScaledSingle: Mat? = null
    private var faceMask3C: Mat? = null
    private var faceDstSubF: Mat? = null
    private var faceSrWeightedF: Mat? = null
    private var faceBlendF: Mat? = null

    // Lightweight OpenCV fallback scratch Mats
    private var lwBlurMat: Mat? = null
    private var lwDetailMat: Mat? = null

    // Feather weight Mat cache keyed by tile dimensions & overlap
    private val featherMaskCache = HashMap<Long, Mat>()

    val isRealEsrganReady: Boolean get() = srActiveIndex < srRankedSessions.size
    val isDncnnReady: Boolean get() = dncnnActiveIndex < dncnnRankedSessions.size
    val isGlesGpuReady: Boolean get() = glesGpuAvailable
    val providerLabel: String get() = activeProviderDescription
    val benchmarkSummary: String get() = benchmarkDetails
    val initDurationMs: Long get() = lastInitDurationMs
    val totalStaticTilesReused: Int get() = srTileCache.totalTilesReused + dncnnTileCache.totalTilesReused

    @Synchronized
    fun initialize(
        capabilities: DeviceCapabilities,
        logger: EngineDiagnosticsLogger? = null,
        forceRebenchmark: Boolean = false
    ) {
        if (!forceRebenchmark && isRealEsrganReady && isDncnnReady && adaptiveProfile != null) {
            return
        }
        val t0 = System.currentTimeMillis()
        val modelDir = File(context.filesDir, "ai_models").apply { mkdirs() }
        val srFile = ensureAssetCopied(
            "ai_models/realesrgan/realesr_general_x4v3.onnx",
            File(modelDir, "realesrgan/realesr_general_x4v3.onnx")
        )
        val dncnnFile = ensureAssetCopied(
            "ai_models/denoise/dncnn.onnx",
            File(modelDir, "denoise/dncnn.onnx")
        )
        ensureAssetCopied(
            "ai_models/denoise/dncnn.data",
            File(modelDir, "denoise/dncnn.data")
        )
        val yunetFile = ensureAssetCopied(
            "ai_models/face_enhancement/face_detection_yunet_2023mar.onnx",
            File(modelDir, "face_enhancement/face_detection_yunet_2023mar.onnx")
        )

        glesGpuAvailable = gpuProcessor.ensureInitialized(logger)

        if (forceRebenchmark) {
            closeModelSessionsOnly()
        }

        val bundle = hwAccelerator.buildOrRestoreExecutionBundle(
            srFile = srFile,
            dncnnFile = dncnnFile,
            yunetFile = yunetFile,
            yunetRunner = yuNetRunner,
            capabilities = capabilities,
            forceRebenchmark = forceRebenchmark,
            logger = logger
        )

        adaptiveProfile = bundle.profile
        srRankedSessions = bundle.srRankedSessions
        srActiveIndex = 0
        dncnnRankedSessions = bundle.dncnnRankedSessions
        dncnnActiveIndex = 0

        refreshActiveLabelsAndSummary()
        lastInitDurationMs = System.currentTimeMillis() - t0
        logger?.info("Adaptive AI init finished in ${lastInitDurationMs}ms: $activeProviderDescription • [$benchmarkDetails]")
    }

    @Synchronized
    fun rebenchmarkHardware(
        capabilities: DeviceCapabilities,
        logger: EngineDiagnosticsLogger? = null
    ): AdaptiveHardwareProfile? {
        initialize(capabilities, logger, forceRebenchmark = true)
        return adaptiveProfile
    }

    fun resetTemporalCaches() {
        srTileCache.reset()
        dncnnTileCache.reset()
        runtimeFallbacksCount = 0
    }

    /**
     * Checks live thermal pressure: if CRITICAL overheating hits while running heavy GPU/NPU
     * sessions, gracefully demotes to a cooler verified backend without ever reducing AI strength.
     */
    @Synchronized
    fun handleThermalPressure(thermalState: ThermalState, logger: EngineDiagnosticsLogger?) {
        if (thermalState == ThermalState.CRITICAL) {
            val srCurr = srRankedSessions.getOrNull(srActiveIndex)
            if (srCurr != null && srCurr.first.kind != com.example.model.AcceleratorKind.CPU && srActiveIndex + 1 < srRankedSessions.size) {
                srActiveIndex++
                runtimeFallbacksCount++
                refreshActiveLabelsAndSummary()
                logger?.warn("Thermal critical demotion for Real-ESRGAN -> ${srRankedSessions[srActiveIndex].first.displayLabel}")
            }
        }
    }

    private fun refreshActiveLabelsAndSummary() {
        val srCurr = srRankedSessions.getOrNull(srActiveIndex)?.first
        val dnCurr = dncnnRankedSessions.getOrNull(dncnnActiveIndex)?.first
        val faceCurr = yuNetRunner.currentAcceleratorKind

        val srKindLabel = srCurr?.kind?.label ?: "CPU"
        val dnKindLabel = dnCurr?.kind?.label ?: "CPU"
        activeProviderDescription = "SR:$srKindLabel • DN:$dnKindLabel • Face:${faceCurr.label}"

        val prof = adaptiveProfile
        benchmarkDetails = if (prof != null) {
            val srPlan = prof.plans[ModelKind.REAL_ESRGAN]
            val dnPlan = prof.plans[ModelKind.DNCNN]
            val fcPlan = prof.plans[ModelKind.FACE_YUNET]
            buildString {
                srPlan?.let { append("SR=${it.selectedLabel}(${it.measuredLatencyMs}ms)") }
                dnPlan?.let { append(" | DN=${it.selectedLabel}(${it.measuredLatencyMs}ms)") }
                fcPlan?.let { append(" | Face=${it.selectedLabel}(${it.measuredLatencyMs}ms)") }
                if (prof.fromCache) append(" [Cached]")
            }
        } else {
            activeProviderDescription
        }
    }

    fun currentPerModelSummary(): String = activeProviderDescription

    /**
     * Runs genuine DnCNN neural denoising ([1, 1, 256, 256] luminance Y tensor) over [yChan]
     * using DnCNN's independently selected hardware processor + static-tile temporal cache
     * and live fallback along DnCNN's ranked backend chain.
     */
    fun denoiseLuminanceYWithDnCnn(
        yChan: Mat,
        strength: Float,
        isSceneCut: Boolean = false,
        logger: EngineDiagnosticsLogger? = null
    ): Boolean {
        if (!isDncnnReady) return false
        val blend = strength.toDouble().coerceIn(0.05, 0.95)
        val w = yChan.cols()
        val h = yChan.rows()
        if (w < 8 || h < 8) return false

        ensureDnCnnBuffers(w, h)
        val inBuf = dncnnDirectInBuf!!
        val inTmp = dncnnInTmpArr!!
        val outArr = dncnnOutArr!!
        val padMat = dncnnPadMat!!
        val floatMat = dncnnFloatMat!!
        val outTileMat = dncnnOutTileMat!!
        val accum = dncnnAccumMat!!
        val weightAccum = dncnnWeightMat!!
        val weighted = dncnnWeightedMat!!
        val u8Out = dncnnU8Mat!!

        val shape = longArrayOf(1L, 1L, dncnnTileSize.toLong(), dncnnTileSize.toLong())

        return try {
            val cols = if (w <= dncnnTileSize) 1 else ceil((w - 24).toDouble() / (dncnnTileSize - 24).toDouble()).toInt()
            val rows = if (h <= dncnnTileSize) 1 else ceil((h - 24).toDouble() / (dncnnTileSize - 24).toDouble()).toInt()
            val xSegments = buildFixedWindowSegments(w, dncnnTileSize, cols)
            val ySegments = buildFixedWindowSegments(h, dncnnTileSize, rows)

            accum.setTo(Scalar(0.0))
            weightAccum.setTo(Scalar(0.0))

            for ((r, yInfo) in ySegments.withIndex()) {
                val (y0, cropH) = yInfo
                for ((c, xInfo) in xSegments.withIndex()) {
                    val (x0, cropW) = xInfo
                    val cropRect = Rect(x0, y0, cropW, cropH)
                    val tileIdx = r * cols + c
                    val sub = Mat(yChan, cropRect)

                    val tileSig = dncnnTileCache.computeTileSignature(sub)
                    val reused = dncnnTileCache.tryReuseTileOutput(tileIdx, tileSig, isSceneCut, outTileMat)

                    if (!reused) {
                        if (cropW == dncnnTileSize && cropH == dncnnTileSize) {
                            sub.convertTo(floatMat, CvType.CV_32FC1, 1.0 / 255.0)
                        } else {
                            Core.copyMakeBorder(
                                sub,
                                padMat,
                                0,
                                dncnnTileSize - cropH,
                                0,
                                dncnnTileSize - cropW,
                                Core.BORDER_REFLECT_101
                            )
                            padMat.convertTo(floatMat, CvType.CV_32FC1, 1.0 / 255.0)
                        }

                        floatMat.get(0, 0, inTmp)
                        inBuf.clear()
                        inBuf.put(inTmp, 0, dncnnTilePixels)
                        inBuf.rewind()

                        runDnCnnWithFallback(shape, inBuf, outArr, logger)
                        outTileMat.put(0, 0, outArr)
                        dncnnTileCache.storeTileOutput(tileIdx, tileSig, outTileMat)
                    }
                    sub.release()

                    val validOut = if (cropW == dncnnTileSize && cropH == dncnnTileSize) {
                        outTileMat
                    } else {
                        Mat(outTileMat, Rect(0, 0, cropW, cropH))
                    }

                    val featherTile = AiTilingMath.TileRegion(
                        index = tileIdx,
                        row = r,
                        col = c,
                        x = x0,
                        y = y0,
                        width = cropW,
                        height = cropH,
                        outX = x0,
                        outY = y0,
                        outWidth = cropW,
                        outHeight = cropH,
                        isFirstX = (c == 0),
                        isLastX = (c == cols - 1),
                        isFirstY = (r == 0),
                        isLastY = (r == rows - 1)
                    )
                    val ovX = if (cols > 1) 24 else 0
                    val ovY = if (rows > 1) 24 else 0
                    val featherSingle = getOrCreateFeatherSingle(featherTile, ovX, ovY)

                    Core.multiply(validOut, featherSingle, weighted)
                    val accSub = Mat(accum, cropRect)
                    Core.add(accSub, weighted, accSub)
                    accSub.release()

                    val wSub = Mat(weightAccum, cropRect)
                    Core.add(wSub, featherSingle, wSub)
                    wSub.release()

                    if (validOut !== outTileMat) {
                        validOut.release()
                    }
                }
            }

            Core.max(weightAccum, Scalar(1e-5), weightAccum)
            Core.divide(accum, weightAccum, accum)
            accum.convertTo(u8Out, CvType.CV_8UC1, 255.0)
            Core.addWeighted(u8Out, blend, yChan, 1.0 - blend, 0.0, yChan)
            true
        } catch (t: Throwable) {
            logger?.warn("DnCNN inference fallback to temporal cleanup (${t.message})")
            false
        }
    }

    private fun runDnCnnWithFallback(
        shape: LongArray,
        inBuf: FloatBuffer,
        outArr: FloatArray,
        logger: EngineDiagnosticsLogger?
    ) {
        while (dncnnActiveIndex < dncnnRankedSessions.size) {
            val (backendId, session) = dncnnRankedSessions[dncnnActiveIndex]
            try {
                val inputName = session.inputNames.first()
                inBuf.rewind()
                OnnxTensor.createTensor(env, inBuf, shape).use { tensor ->
                    session.run(mapOf(inputName to tensor)).use { results ->
                        val outTensor = results[0] as OnnxTensor
                        val outBuf = outTensor.floatBuffer
                        outBuf.position(0)
                        outBuf.get(outArr, 0, dncnnTilePixels)
                        val probe = outArr[dncnnTilePixels / 2]
                        if (probe.isNaN() || probe.isInfinite()) {
                            throw IllegalStateException("NaN output from ${backendId.displayLabel}")
                        }
                    }
                }
                return
            } catch (t: Throwable) {
                logger?.warn("DnCNN runtime fallback from ${backendId.displayLabel} (${t.message})")
                dncnnActiveIndex++
                runtimeFallbacksCount++
                refreshActiveLabelsAndSummary()
            }
        }
        throw IllegalStateException("All DnCNN backends failed")
    }

    private fun buildFixedWindowSegments(totalLen: Int, windowSize: Int, count: Int): List<Pair<Int, Int>> {
        if (totalLen <= windowSize || count <= 1) {
            return listOf(0 to min(totalLen, windowSize))
        }
        val step = (totalLen - windowSize).toDouble() / (count - 1).toDouble()
        return (0 until count).map { i ->
            val start = if (i == count - 1) {
                (totalLen - windowSize).coerceAtLeast(0)
            } else {
                (i * step).roundToInt().coerceIn(0, (totalLen - windowSize).coerceAtLeast(0))
            }
            start to windowSize
        }
    }

    /**
     * High-speed Real-ESRGAN Super-Resolution + Face ROI SR Refinement:
     * - Double-buffered direct FloatBuffer tile packing (tile i+1 packed concurrently while tile i runs)
     * - Motion-gated Static Tile Temporal Reuse (preserves 100% Clarity neural sharpness)
     * - Single-channel reciprocal 4K accumulation normalization
     * - Live fallback along Real-ESRGAN's ranked GPU -> NPU -> CPU session chain
     */
    fun enhanceBgrWithPresetTiles(
        srcBgr: Mat,
        outW: Int,
        outH: Int,
        tileSize: Int,
        targetTilesPerFrame: Int,
        estimatedTotalFrames: Int,
        forceLightweightFallback: Boolean,
        aiStrength: Float = 0.65f,
        fallbackSharpnessBoost: Float = 0.28f,
        faces: List<Rect> = emptyList(),
        faceEnhanceStrength: Float = 0.65f,
        isSceneCut: Boolean = false,
        isCancelled: () -> Boolean,
        logger: EngineDiagnosticsLogger? = null,
        onTileProgress: (done: Int, total: Int, tileSize: Int) -> Unit = { _, _, _ -> }
    ): TileProcessOutcome {
        val inW = srcBgr.cols()
        val inH = srcBgr.rows()
        val safeOutW = outW.coerceAtLeast(inW)
        val safeOutH = outH.coerceAtLeast(inH)

        if (forceLightweightFallback || !isRealEsrganReady) {
            val resizedBase = if (inW != safeOutW || inH != safeOutH) {
                Mat().also {
                    Imgproc.resize(srcBgr, it, Size(safeOutW.toDouble(), safeOutH.toDouble()), 0.0, 0.0, Imgproc.INTER_CUBIC)
                }
            } else {
                srcBgr.clone()
            }
            val lw = enhanceFastLightweightCopy(
                resizedBase,
                sharpnessBoost = fallbackSharpnessBoost.coerceIn(0.08f, 0.45f),
                contrastBoost = 1.02f,
                satBoost = 1.01f
            )
            resizedBase.release()
            val label = if (glesGpuAvailable) "GLES 3.1 GPU Fast" else "OpenCV SIMD Fast"
            return TileProcessOutcome(
                outputBgr = lw,
                usedLightweightFallback = true,
                activeAccelerationLabel = label,
                faceAiMs = 0L
            )
        }

        val safeTileSize = tileSize.coerceIn(192, 512)
        val grid = AiTilingMath.planUpscaleTileGrid(
            inW = inW,
            inH = inH,
            outW = safeOutW,
            outH = safeOutH,
            tileSize = safeTileSize,
            maxTilesCap = targetTilesPerFrame.coerceAtLeast(1)
        )

        ensureNativeWorkMats(inW, inH, safeOutW, safeOutH)
        ensureTileBuffers(safeTileSize)

        val localRgb = srcRgbMat!!
        Imgproc.cvtColor(srcBgr, localRgb, Imgproc.COLOR_BGR2RGB)

        if (inW == safeOutW && inH == safeOutH) {
            localRgb.copyTo(baseRgb8u!!)
        } else {
            Imgproc.resize(localRgb, baseRgb8u!!, Size(safeOutW.toDouble(), safeOutH.toDouble()), 0.0, 0.0, Imgproc.INTER_CUBIC)
        }

        accumRgbMat!!.setTo(Scalar(0.0, 0.0, 0.0))
        accumWeightMat!!.setTo(Scalar(0.0))

        val fixedShape = longArrayOf(1L, 3L, safeTileSize.toLong(), safeTileSize.toLong())
        val outResized = outResizedFloat!!
        val weightedTile = weightedTileFloat!!

        data class PreparedTile(
            val tile: AiTilingMath.TileRegion,
            val reusedFromCache: Boolean,
            val signature: ByteArray,
            val validSrW: Int,
            val validSrH: Int,
            val bufIndex: Int
        )

        fun prepareSingleTile(tile: AiTilingMath.TileRegion, bufIdx: Int): PreparedTile {
            val inRect = Rect(tile.x, tile.y, tile.width, tile.height)
            val roi = Mat(localRgb, inRect)
            val sig = srTileCache.computeTileSignature(roi)
            val useSrCache = grid.totalTiles > 1
            val padMat = if (bufIdx == 0) tilePaddedMatA!! else tilePaddedMatB!!
            val floatHwc = if (bufIdx == 0) tileFloatHwcA!! else tileFloatHwcB!!
            val chIn = if (bufIdx == 0) tileChannelsA!! else tileChannelsB!!
            val planeTmp = if (bufIdx == 0) tileInPlaneTmpA else tileInPlaneTmpB
            val directBuf = if (bufIdx == 0) tileInBufA!! else tileInBufB!!

            val validSrW: Int
            val validSrH: Int
            if (tile.width <= safeTileSize && tile.height <= safeTileSize) {
                if (tile.width == safeTileSize && tile.height == safeTileSize) {
                    roi.copyTo(padMat)
                } else {
                    Core.copyMakeBorder(
                        roi, padMat,
                        0, safeTileSize - tile.height,
                        0, safeTileSize - tile.width,
                        Core.BORDER_REFLECT_101
                    )
                }
                validSrW = tile.width * 4
                validSrH = tile.height * 4
            } else {
                Imgproc.resize(roi, padMat, Size(safeTileSize.toDouble(), safeTileSize.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
                validSrW = safeTileSize * 4
                validSrH = safeTileSize * 4
            }
            roi.release()

            padMat.convertTo(floatHwc, CvType.CV_32FC3, 1.0 / 255.0)
            Core.split(floatHwc, chIn)
            directBuf.clear()
            for (c in 0 until 3) {
                chIn[c].get(0, 0, planeTmp)
                directBuf.put(planeTmp)
            }
            directBuf.rewind()
            return PreparedTile(
                tile = tile,
                reusedFromCache = useSrCache && false,
                signature = sig,
                validSrW = validSrW,
                validSrH = validSrH,
                bufIndex = bufIdx
            )
        }

        val prepExec = tilePrepExecutor ?: Executors.newSingleThreadExecutor().also { tilePrepExecutor = it }
        var tileTimingSumMs = 0L
        var timedTiles = 0
        var reusedInFrame = 0

        val tiles = grid.tiles
        var nextPrepFuture: Future<PreparedTile>? = if (tiles.isNotEmpty()) {
            prepExec.submit(Callable { prepareSingleTile(tiles[0], 0) })
        } else {
            null
        }

        for (i in tiles.indices) {
            if (isCancelled()) throw InterruptedException("Cancelled during AI tile enhancement")
            val tTile0 = System.currentTimeMillis()
            val prepared = nextPrepFuture!!.get()
            val tile = prepared.tile

            // Pipeline next tile buffer preparation concurrently on background thread while this tile runs
            nextPrepFuture = if (i + 1 < tiles.size) {
                val nextIdx = (prepared.bufIndex + 1) and 1
                val nextTile = tiles[i + 1]
                prepExec.submit(Callable { prepareSingleTile(nextTile, nextIdx) })
            } else {
                null
            }

            val reused = grid.totalTiles > 1 && srTileCache.tryReuseTileOutput(
                tileKey = tile.index,
                currentSig = prepared.signature,
                isSceneCut = isSceneCut,
                dstOutFloat = outResized
            )

            if (reused) {
                reusedInFrame++
            } else {
                val dstSize = Size(tile.outWidth.toDouble(), tile.outHeight.toDouble())
                val activeBuf = if (prepared.bufIndex == 0) tileInBufA!! else tileInBufB!!
                runSrTileAndResizeDirect(
                    inputBuf = activeBuf,
                    shape = fixedShape,
                    tileSize = safeTileSize,
                    validSrW = prepared.validSrW,
                    validSrH = prepared.validSrH,
                    dstSize = dstSize,
                    outResizedHwc = outResized,
                    logger = logger
                )
                if (grid.totalTiles > 1) {
                    srTileCache.storeTileOutput(tile.index, prepared.signature, outResized)
                }

                val tileElapsed = (System.currentTimeMillis() - tTile0).coerceAtLeast(1L)
                tileTimingSumMs += tileElapsed
                timedTiles++
            }

            val featherMat = getOrCreateOutputFeatherMask(tile, grid.outOverlapX, grid.outOverlapY)
            val featherSingle = getOrCreateOutputFeatherSingle(tile, grid.outOverlapX, grid.outOverlapY)

            Core.multiply(outResized, featherMat, weightedTile)
            val outRect = Rect(tile.outX, tile.outY, tile.outWidth, tile.outHeight)
            val accumSub = Mat(accumRgbMat!!, outRect)
            Core.add(accumSub, weightedTile, accumSub)
            accumSub.release()

            val weightSub = Mat(accumWeightMat!!, outRect)
            Core.add(weightSub, featherSingle, weightSub)
            weightSub.release()

            onTileProgress(tile.index + 1, grid.totalTiles, safeTileSize)

            if (!reused && timedTiles == 2 && AiSpeedPolicy.shouldEngageLightweightFallback(
                    measuredTileMs = tileTimingSumMs / timedTiles,
                    tilesPerFrame = grid.totalTiles,
                    estimatedFrameCount = estimatedTotalFrames
                )
            ) {
                val avgTileMs = tileTimingSumMs / timedTiles
                val reason = "Emergency Fallback: ${safeTileSize}px neural tile stalled at ${avgTileMs}ms/tile"
                logger?.warn(reason)
                nextPrepFuture?.cancel(true)
                val lw = enhanceFastLightweightCopy(
                    baseRgb8u!!,
                    sharpnessBoost = fallbackSharpnessBoost.coerceIn(0.08f, 0.45f),
                    contrastBoost = 1.02f,
                    satBoost = 1.01f
                )
                val bgrOut = Mat(safeOutH, safeOutW, CvType.CV_8UC3)
                Imgproc.cvtColor(lw, bgrOut, Imgproc.COLOR_RGB2BGR)
                lw.release()
                val label = if (glesGpuAvailable) "GLES 3.1 GPU Fast" else "OpenCV SIMD Fast"
                return TileProcessOutcome(
                    outputBgr = bgrOut,
                    usedLightweightFallback = true,
                    fallbackReason = reason,
                    activeAccelerationLabel = label,
                    faceAiMs = 0L,
                    staticTilesReused = reusedInFrame
                )
            }
        }

        // Fast Single-Channel Reciprocal 4K Normalization (eliminates 7 full-frame 4K FP32 passes!)
        val wMat = accumWeightMat!!
        val invSingle = invWeightSingleMat!!
        val invThree = invWeightThreeMat!!
        Core.max(wMat, weightClampFloorMat!!, wMat)
        Core.divide(1.0, wMat, invSingle)
        Core.merge(listOf(invSingle, invSingle, invSingle), invThree)
        Core.multiply(accumRgbMat!!, invThree, accumRgbMat!!)
        accumRgbMat!!.convertTo(outRgb8u!!, CvType.CV_8UC3, 255.0) // NEON saturates [0..255]

        val aiBlend = aiStrength.toDouble().coerceIn(0.20, 0.96)
        Core.addWeighted(outRgb8u!!, aiBlend, baseRgb8u!!, 1.0 - aiBlend, 0.0, outRgb8u!!)

        // Dedicated Face ROI Neural Super-Resolution Refinement
        var faceMs = 0L
        if (faces.isNotEmpty() && faceEnhanceStrength >= 0.15f) {
            val tf0 = System.currentTimeMillis()
            enhanceFaceRegionsWithSr(
                srcRgb = localRgb,
                targetRgb8u = outRgb8u!!,
                inW = inW,
                inH = inH,
                outW = safeOutW,
                outH = safeOutH,
                faces = faces,
                safeTileSize = safeTileSize,
                fixedShape = fixedShape,
                strength = faceEnhanceStrength,
                isCancelled = isCancelled,
                logger = logger
            )
            faceMs = System.currentTimeMillis() - tf0
        }

        val finalBgr = Mat(safeOutH, safeOutW, CvType.CV_8UC3)
        Imgproc.cvtColor(outRgb8u!!, finalBgr, Imgproc.COLOR_RGB2BGR)

        return TileProcessOutcome(
            outputBgr = finalBgr,
            usedLightweightFallback = false,
            activeAccelerationLabel = activeProviderDescription,
            faceAiMs = faceMs,
            staticTilesReused = reusedInFrame
        )
    }

    private fun enhanceFaceRegionsWithSr(
        srcRgb: Mat,
        targetRgb8u: Mat,
        inW: Int,
        inH: Int,
        outW: Int,
        outH: Int,
        faces: List<Rect>,
        safeTileSize: Int,
        fixedShape: LongArray,
        strength: Float,
        isCancelled: () -> Boolean,
        logger: EngineDiagnosticsLogger?
    ) {
        val chIn = tileChannelsA!!
        val paddedTile = tilePaddedMatA!!
        val floatTileHwc = tileFloatHwcA!!
        val directBuf = tileInBufA!!
        val planeTmp = tileInPlaneTmpA
        val scaleX = outW.toDouble() / inW.toDouble()
        val scaleY = outH.toDouble() / inH.toDouble()

        val faceSrF = faceSrFloatMat ?: Mat().also { faceSrFloatMat = it }
        val templateMask = getOrCreateFaceMaskTemplate128()
        val maskScaled = faceMaskScaledSingle ?: Mat().also { faceMaskScaledSingle = it }
        val mask3 = faceMask3C ?: Mat().also { faceMask3C = it }
        val dstSubF = faceDstSubF ?: Mat().also { faceDstSubF = it }
        val srWeightedF = faceSrWeightedF ?: Mat().also { faceSrWeightedF = it }
        val blendOutF = faceBlendF ?: Mat().also { faceBlendF = it }
        val maxWeight = strength.toDouble().coerceIn(0.25, 0.92)

        for (face in faces.take(3)) {
            if (isCancelled()) break

            val padX = (face.width * 0.35).roundToInt()
            val padTop = (face.height * 0.45).roundToInt()
            val padBot = (face.height * 0.28).roundToInt()

            val fx0 = (face.x - padX).coerceIn(0, inW - 2)
            val fy0 = (face.y - padTop).coerceIn(0, inH - 2)
            val fx1 = (face.x + face.width + padX).coerceIn(fx0 + 2, inW)
            val fy1 = (face.y + face.height + padBot).coerceIn(fy0 + 2, inH)
            val fw = fx1 - fx0
            val fh = fy1 - fy0
            if (fw < 16 || fh < 16) continue

            val roi = Mat(srcRgb, Rect(fx0, fy0, fw, fh))
            val validSrW: Int
            val validSrH: Int
            if (fw <= safeTileSize && fh <= safeTileSize) {
                Core.copyMakeBorder(
                    roi, paddedTile,
                    0, safeTileSize - fh,
                    0, safeTileSize - fw,
                    Core.BORDER_REFLECT_101
                )
                validSrW = fw * 4
                validSrH = fh * 4
            } else {
                Imgproc.resize(roi, paddedTile, Size(safeTileSize.toDouble(), safeTileSize.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
                validSrW = safeTileSize * 4
                validSrH = safeTileSize * 4
            }
            roi.release()

            paddedTile.convertTo(floatTileHwc, CvType.CV_32FC3, 1.0 / 255.0)
            Core.split(floatTileHwc, chIn)
            directBuf.clear()
            for (c in 0 until 3) {
                chIn[c].get(0, 0, planeTmp)
                directBuf.put(planeTmp)
            }
            directBuf.rewind()

            val outX0 = floor(fx0 * scaleX).toInt().coerceIn(0, outW - 2)
            val outY0 = floor(fy0 * scaleY).toInt().coerceIn(0, outH - 2)
            val outX1 = ceil(fx1 * scaleX).toInt().coerceIn(outX0 + 2, outW)
            val outY1 = ceil(fy1 * scaleY).toInt().coerceIn(outY0 + 2, outH)
            val outFw = outX1 - outX0
            val outFh = outY1 - outY0

            runSrTileAndResizeDirect(
                inputBuf = directBuf,
                shape = fixedShape,
                tileSize = safeTileSize,
                validSrW = validSrW,
                validSrH = validSrH,
                dstSize = Size(outFw.toDouble(), outFh.toDouble()),
                outResizedHwc = faceSrF,
                logger = logger
            )

            // Resize cached canonical 128x128 elliptical mask instead of reallocating FloatArrays per face
            Imgproc.resize(templateMask, maskScaled, Size(outFw.toDouble(), outFh.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
            if (abs(maxWeight - 1.0) > 0.01) {
                maskScaled.convertTo(maskScaled, CvType.CV_32FC1, maxWeight, 0.0)
            }
            Core.merge(listOf(maskScaled, maskScaled, maskScaled), mask3)

            val dstSub = Mat(targetRgb8u, Rect(outX0, outY0, outFw, outFh))
            try {
                dstSub.convertTo(dstSubF, CvType.CV_32FC3)
                faceSrF.convertTo(srWeightedF, CvType.CV_32FC3, 255.0)
                // Alpha blend: dst + mask * (sr - dst) -> avoids Mat.ones & invMask3 allocation completely!
                Core.subtract(srWeightedF, dstSubF, blendOutF)
                Core.multiply(blendOutF, mask3, blendOutF)
                Core.add(dstSubF, blendOutF, dstSubF)
                dstSubF.convertTo(dstSub, CvType.CV_8UC3)
            } finally {
                dstSub.release()
            }
        }
    }

    private fun getOrCreateFaceMaskTemplate128(): Mat {
        faceMaskTemplate128?.let { return it }
        val size = 128
        val data = FloatArray(size * size)
        val half = size * 0.5f
        for (y in 0 until size) {
            val dy = (y - half) / half
            val rowOff = y * size
            for (x in 0 until size) {
                val dx = (x - half) / half
                val radial = sqrt(dx * dx + dy * dy).coerceIn(0f, 1f)
                data[rowOff + x] = if (radial < 0.45f) {
                    1f
                } else {
                    val t = (radial - 0.45f) / 0.55f
                    0.5f * (1f + cos(PI.toFloat() * t))
                }
            }
        }
        return Mat(size, size, CvType.CV_32FC1).also {
            it.put(0, 0, data)
            faceMaskTemplate128 = it
        }
    }

    fun enhanceFastLightweightCopy(
        srcBgr: Mat,
        sharpnessBoost: Float = 0.28f,
        contrastBoost: Float = 1.03f,
        satBoost: Float = 1.02f
    ): Mat {
        val out = srcBgr.clone()
        if (glesGpuAvailable) {
            val gpuOk = gpuProcessor.enhanceBgrInPlace(
                bgr = out,
                sharpness = sharpnessBoost,
                contrast = contrastBoost,
                saturation = satBoost
            )
            if (gpuOk) return out
        }
        val blur = lwBlurMat ?: Mat().also { lwBlurMat = it }
        val detail = lwDetailMat ?: Mat().also { lwDetailMat = it }
        Imgproc.GaussianBlur(out, blur, Size(0.0, 0.0), 1.0)
        Core.addWeighted(out, 1.0 + sharpnessBoost, blur, -sharpnessBoost.toDouble(), 0.0, detail)
        detail.copyTo(out)
        return out
    }

    private fun runSrTileAndResizeDirect(
        inputBuf: FloatBuffer,
        shape: LongArray,
        tileSize: Int,
        validSrW: Int,
        validSrH: Int,
        dstSize: Size,
        outResizedHwc: Mat,
        logger: EngineDiagnosticsLogger?
    ) {
        while (srActiveIndex < srRankedSessions.size) {
            val (backendId, current) = srRankedSessions[srActiveIndex]
            try {
                val inputName = current.inputNames.first()
                inputBuf.rewind()
                OnnxTensor.createTensor(env, inputBuf, shape).use { tensor ->
                    current.run(mapOf(inputName to tensor)).use { results ->
                        val outTensor = results[0] as OnnxTensor
                        val outBuf = outTensor.floatBuffer
                        val probe = outBuf.get(0)
                        if (probe.isNaN() || probe.isInfinite()) {
                            throw IllegalStateException("NaN output from ${backendId.displayLabel}")
                        }
                        extractValidPlanesAndResize(outBuf, tileSize, validSrW, validSrH, dstSize, outResizedHwc)
                    }
                }
                return
            } catch (t: Throwable) {
                logger?.warn("Real-ESRGAN runtime fallback from ${backendId.displayLabel} (${t.message})")
                srActiveIndex++
                runtimeFallbacksCount++
                refreshActiveLabelsAndSummary()
            }
        }
        throw IllegalStateException("All Real-ESRGAN ONNX execution backends failed")
    }

    private fun extractValidPlanesAndResize(
        outBuf: FloatBuffer,
        tileSize: Int,
        validSrW: Int,
        validSrH: Int,
        dstSize: Size,
        outResizedHwc: Mat
    ) {
        val outDim = tileSize * 4
        val outPixels = outDim * outDim
        val fullPlane = outFullPlaneMat!!
        val smallCh = resizedOutChannels!!
        val safeValidW = validSrW.coerceIn(1, outDim)
        val safeValidH = validSrH.coerceIn(1, outDim)
        val needCrop = safeValidW < outDim || safeValidH < outDim
        val interp = when {
            dstSize.width < safeValidW - 0.5 -> Imgproc.INTER_AREA
            dstSize.width > safeValidW + 0.5 -> Imgproc.INTER_CUBIC
            else -> Imgproc.INTER_LINEAR
        }

        for (c in 0 until 3) {
            outBuf.position(c * outPixels)
            outBuf.get(tileOutPlaneTmp, 0, outPixels)
            fullPlane.put(0, 0, tileOutPlaneTmp)
            val srcRegion = if (needCrop) Mat(fullPlane, Rect(0, 0, safeValidW, safeValidH)) else fullPlane
            Imgproc.resize(srcRegion, smallCh[c], dstSize, 0.0, 0.0, interp)
            if (needCrop) {
                srcRegion.release()
            }
        }
        Core.merge(smallCh, outResizedHwc)
    }

    private fun ensureDnCnnBuffers(w: Int, h: Int) {
        if (dncnnDirectInBuf == null) {
            dncnnDirectInBuf = ByteBuffer.allocateDirect(dncnnTilePixels * 4)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer()
            dncnnInTmpArr = FloatArray(dncnnTilePixels)
            dncnnOutArr = FloatArray(dncnnTilePixels)
            dncnnPadMat = Mat(dncnnTileSize, dncnnTileSize, CvType.CV_8UC1)
            dncnnFloatMat = Mat(dncnnTileSize, dncnnTileSize, CvType.CV_32FC1)
            dncnnOutTileMat = Mat(dncnnTileSize, dncnnTileSize, CvType.CV_32FC1)
            dncnnWeightedMat = Mat()
        }
        val curW = dncnnAccumMat?.cols() ?: -1
        val curH = dncnnAccumMat?.rows() ?: -1
        if (curW != w || curH != h) {
            dncnnAccumMat?.release()
            dncnnWeightMat?.release()
            dncnnU8Mat?.release()
            dncnnAccumMat = Mat(h, w, CvType.CV_32FC1)
            dncnnWeightMat = Mat(h, w, CvType.CV_32FC1)
            dncnnU8Mat = Mat(h, w, CvType.CV_8UC1)
        }
    }

    private fun ensureTileBuffers(tileSize: Int) {
        if (allocTileSize != tileSize) {
            allocTileSize = tileSize
            val px = tileSize * tileSize
            val bytes = 3 * px * 4
            tileInBufA = ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder()).asFloatBuffer()
            tileInBufB = ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder()).asFloatBuffer()
            tileInPlaneTmpA = FloatArray(px)
            tileInPlaneTmpB = FloatArray(px)

            val outDim = tileSize * 4
            val outPx = outDim * outDim
            tileOutPlaneTmp = FloatArray(outPx)

            tileChannelsA?.forEach { it.release() }
            tileChannelsB?.forEach { it.release() }
            resizedOutChannels?.forEach { it.release() }
            tileChannelsA = List(3) { Mat(tileSize, tileSize, CvType.CV_32FC1) }
            tileChannelsB = List(3) { Mat(tileSize, tileSize, CvType.CV_32FC1) }
            resizedOutChannels = List(3) { Mat() }

            outFullPlaneMat?.release()
            outFullPlaneMat = Mat(outDim, outDim, CvType.CV_32FC1)
            tilePaddedMatA?.release(); tilePaddedMatA = Mat(tileSize, tileSize, CvType.CV_8UC3)
            tilePaddedMatB?.release(); tilePaddedMatB = Mat(tileSize, tileSize, CvType.CV_8UC3)
            tileFloatHwcA?.release(); tileFloatHwcA = Mat(tileSize, tileSize, CvType.CV_32FC3)
            tileFloatHwcB?.release(); tileFloatHwcB = Mat(tileSize, tileSize, CvType.CV_32FC3)
        }
    }

    private fun ensureNativeWorkMats(inW: Int, inH: Int, outW: Int, outH: Int) {
        val curInW = srcRgbMat?.cols() ?: -1
        val curInH = srcRgbMat?.rows() ?: -1
        if (curInW != inW || curInH != inH) {
            srcRgbMat?.release()
            srcRgbMat = Mat(inH, inW, CvType.CV_8UC3)
        }

        val curOutW = outRgb8u?.cols() ?: -1
        val curOutH = outRgb8u?.rows() ?: -1
        if (curOutW != outW || curOutH != outH) {
            baseRgb8u?.release()
            accumRgbMat?.release()
            accumWeightMat?.release()
            invWeightSingleMat?.release()
            invWeightThreeMat?.release()
            weightClampFloorMat?.release()
            outRgb8u?.release()
            clearFeatherCache()

            baseRgb8u = Mat(outH, outW, CvType.CV_8UC3)
            accumRgbMat = Mat(outH, outW, CvType.CV_32FC3)
            accumWeightMat = Mat(outH, outW, CvType.CV_32FC1)
            invWeightSingleMat = Mat(outH, outW, CvType.CV_32FC1)
            invWeightThreeMat = Mat(outH, outW, CvType.CV_32FC3)
            weightClampFloorMat = Mat(outH, outW, CvType.CV_32FC1, Scalar(1e-5))
            outRgb8u = Mat(outH, outW, CvType.CV_8UC3)
            outResizedFloat = outResizedFloat ?: Mat()
            weightedTileFloat = weightedTileFloat ?: Mat()
        }
    }

    private fun featherCacheKey(
        w: Int,
        h: Int,
        ovX: Int,
        ovY: Int,
        isFirstX: Boolean,
        isLastX: Boolean,
        isFirstY: Boolean,
        isLastY: Boolean,
        singleChannel: Boolean
    ): Long {
        var key = w.toLong()
        key = key * 31L + h.toLong()
        key = key * 31L + ovX.toLong()
        key = key * 31L + ovY.toLong()
        val flags = (if (isFirstX) 1 else 0) or
            (if (isLastX) 2 else 0) or
            (if (isFirstY) 4 else 0) or
            (if (isLastY) 8 else 0) or
            (if (singleChannel) 16 else 0)
        return (key shl 5) or flags.toLong()
    }

    private fun getOrCreateFeatherSingle(tile: AiTilingMath.TileRegion, ovX: Int, ovY: Int): Mat {
        val key = featherCacheKey(
            tile.width, tile.height, ovX, ovY,
            tile.isFirstX, tile.isLastX, tile.isFirstY, tile.isLastY, true
        )
        featherMaskCache[key]?.let { return it }

        val w = tile.width
        val h = tile.height
        val rowW = FloatArray(w) { dx -> AiTilingMath.featherWeight(dx, w, tile.isFirstX, tile.isLastX, ovX) }
        val colW = FloatArray(h) { dy -> AiTilingMath.featherWeight(dy, h, tile.isFirstY, tile.isLastY, ovY) }
        val flat = FloatArray(w * h)
        for (dy in 0 until h) {
            val wy = colW[dy]
            val off = dy * w
            for (dx in 0 until w) {
                flat[off + dx] = rowW[dx] * wy
            }
        }
        val mat = Mat(h, w, CvType.CV_32FC1)
        mat.put(0, 0, flat)
        featherMaskCache[key] = mat
        return mat
    }

    private fun getOrCreateOutputFeatherSingle(tile: AiTilingMath.TileRegion, ovX: Int, ovY: Int): Mat {
        val key = featherCacheKey(
            tile.outWidth, tile.outHeight, ovX, ovY,
            tile.isFirstX, tile.isLastX, tile.isFirstY, tile.isLastY, true
        )
        featherMaskCache[key]?.let { return it }

        val w = tile.outWidth
        val h = tile.outHeight
        val rowW = FloatArray(w) { dx -> AiTilingMath.featherWeight(dx, w, tile.isFirstX, tile.isLastX, ovX) }
        val colW = FloatArray(h) { dy -> AiTilingMath.featherWeight(dy, h, tile.isFirstY, tile.isLastY, ovY) }
        val flat = FloatArray(w * h)
        for (dy in 0 until h) {
            val wy = colW[dy]
            val off = dy * w
            for (dx in 0 until w) {
                flat[off + dx] = rowW[dx] * wy
            }
        }
        val mat = Mat(h, w, CvType.CV_32FC1)
        mat.put(0, 0, flat)
        featherMaskCache[key] = mat
        return mat
    }

    private fun getOrCreateOutputFeatherMask(tile: AiTilingMath.TileRegion, ovX: Int, ovY: Int): Mat {
        val key = featherCacheKey(
            tile.outWidth, tile.outHeight, ovX, ovY,
            tile.isFirstX, tile.isLastX, tile.isFirstY, tile.isLastY, false
        )
        featherMaskCache[key]?.let { return it }

        val single = getOrCreateOutputFeatherSingle(tile, ovX, ovY)
        val mask3C = Mat(tile.outHeight, tile.outWidth, CvType.CV_32FC3)
        Core.merge(listOf(single, single, single), mask3C)
        featherMaskCache[key] = mask3C
        return mask3C
    }

    private fun clearFeatherCache() {
        featherMaskCache.values.forEach { it.release() }
        featherMaskCache.clear()
    }

    fun releaseWorkBuffers() {
        srTileCache.reset()
        dncnnTileCache.reset()
        yuNetRunner.releaseWorkBuffers()

        tilePrepExecutor?.shutdownNow()
        tilePrepExecutor = null

        allocTileSize = 0
        tileInBufA = null
        tileInBufB = null
        tileInPlaneTmpA = FloatArray(0)
        tileInPlaneTmpB = FloatArray(0)
        tileOutPlaneTmp = FloatArray(0)

        dncnnDirectInBuf = null
        dncnnInTmpArr = null
        dncnnOutArr = null
        dncnnPadMat?.release(); dncnnPadMat = null
        dncnnFloatMat?.release(); dncnnFloatMat = null
        dncnnOutTileMat?.release(); dncnnOutTileMat = null
        dncnnAccumMat?.release(); dncnnAccumMat = null
        dncnnWeightMat?.release(); dncnnWeightMat = null
        dncnnWeightedMat?.release(); dncnnWeightedMat = null
        dncnnU8Mat?.release(); dncnnU8Mat = null

        tileChannelsA?.forEach { it.release() }; tileChannelsA = null
        tileChannelsB?.forEach { it.release() }; tileChannelsB = null
        resizedOutChannels?.forEach { it.release() }; resizedOutChannels = null
        outFullPlaneMat?.release(); outFullPlaneMat = null
        tilePaddedMatA?.release(); tilePaddedMatA = null
        tilePaddedMatB?.release(); tilePaddedMatB = null
        tileFloatHwcA?.release(); tileFloatHwcA = null
        tileFloatHwcB?.release(); tileFloatHwcB = null
        outResizedFloat?.release(); outResizedFloat = null
        weightedTileFloat?.release(); weightedTileFloat = null

        srcRgbMat?.release(); srcRgbMat = null
        baseRgb8u?.release(); baseRgb8u = null
        accumRgbMat?.release(); accumRgbMat = null
        accumWeightMat?.release(); accumWeightMat = null
        invWeightSingleMat?.release(); invWeightSingleMat = null
        invWeightThreeMat?.release(); invWeightThreeMat = null
        weightClampFloorMat?.release(); weightClampFloorMat = null
        outRgb8u?.release(); outRgb8u = null

        faceSrFloatMat?.release(); faceSrFloatMat = null
        faceMaskTemplate128?.release(); faceMaskTemplate128 = null
        faceMaskScaledSingle?.release(); faceMaskScaledSingle = null
        faceMask3C?.release(); faceMask3C = null
        faceDstSubF?.release(); faceDstSubF = null
        faceSrWeightedF?.release(); faceSrWeightedF = null
        faceBlendF?.release(); faceBlendF = null

        lwBlurMat?.release(); lwBlurMat = null
        lwDetailMat?.release(); lwDetailMat = null
        clearFeatherCache()
    }

    private fun ensureAssetCopied(assetPath: String, destFile: File): File {
        destFile.parentFile?.mkdirs()
        if (!destFile.exists() || destFile.length() == 0L) {
            context.assets.open(assetPath).use { input ->
                destFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        }
        return destFile
    }

    private fun closeModelSessionsOnly() {
        srRankedSessions.forEach { runCatching { it.second.close() } }
        dncnnRankedSessions.forEach { runCatching { it.second.close() } }
        srRankedSessions = emptyList()
        dncnnRankedSessions = emptyList()
        srActiveIndex = 0
        dncnnActiveIndex = 0
        yuNetRunner.close()
    }

    fun close() {
        releaseWorkBuffers()
        gpuProcessor.release()
        closeModelSessionsOnly()
    }
}
