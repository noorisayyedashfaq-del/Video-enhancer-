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

    private val envLazy = lazy { OrtEnvironment.getEnvironment() }
    private val env: OrtEnvironment get() = envLazy.value
    val gpuProcessor: GlesGpuImageProcessor = GlesGpuImageProcessor()
    private val yuNetRunnerLazy = lazy { YuNetOnnxRunner(env) }
    val yuNetRunner: YuNetOnnxRunner get() = yuNetRunnerLazy.value
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
    private var dncnnResidualF: Mat? = null
    private var dncnnResidualUpF: Mat? = null
    private var dncnnYFloatMat: Mat? = null

    // Static tile temporal reuse caches
    private val srTileCache = StaticTileReuseCache()
    private val dncnnTileCache = StaticTileReuseCache()

    // Reusable OpenCV native C++ scratch Mats
    private var srcRgbMat: Mat? = null
    private var srWorkRgbMat: Mat? = null
    private var baseRgb8u: Mat? = null
    private var baseLowBlur8u: Mat? = null
    private var baseHighDetailS16: Mat? = null
    private var outRgbS16: Mat? = null
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
    private var lwBlurFineMat: Mat? = null
    private var lwMinNMat: Mat? = null
    private var lwMaxNMat: Mat? = null

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
        fun copySafe(assetPath: String, dest: File): File = runCatching {
            ensureAssetCopied(assetPath, dest)
        }.onFailure { err ->
            logger?.warn("Asset copy failed for $assetPath: ${err.message}")
        }.getOrDefault(dest)

        val srFile = copySafe(
            "ai_models/realesrgan/realesr_general_x4v3.onnx",
            File(modelDir, "realesrgan/realesr_general_x4v3.onnx")
        )
        val dncnnFile = copySafe(
            "ai_models/denoise/dncnn.onnx",
            File(modelDir, "denoise/dncnn.onnx")
        )
        copySafe(
            "ai_models/denoise/dncnn.data",
            File(modelDir, "denoise/dncnn.data")
        )
        val yunetFile = copySafe(
            "ai_models/face_enhancement/face_detection_yunet_2023mar.onnx",
            File(modelDir, "face_enhancement/face_detection_yunet_2023mar.onnx")
        )

        glesGpuAvailable = runCatching { gpuProcessor.ensureInitialized(logger) }.getOrDefault(false)

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
     * Edge-preserving DnCNN neural luminance denoising ([1, 1, 256, 256] ONNX).
     * When input luminance exceeds 256x256, extracts ONLY the smooth DnCNN noise-removal
     * residual delta at 256x256 and upsamples the correction + applies light edge-preserving
     * smoothing on flat blocks — keeping full-resolution sharp textures intact!
     */
    fun denoiseLuminanceYWithDnCnn(
        yChan: Mat,
        strength: Float,
        isSceneCut: Boolean = false,
        logger: EngineDiagnosticsLogger? = null
    ): Boolean {
        if (!isDncnnReady) return false
        val blend = strength.toDouble().coerceIn(0.10, 0.90)
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
        val u8Out = dncnnU8Mat!!
        val shape = longArrayOf(1L, 1L, dncnnTileSize.toLong(), dncnnTileSize.toLong())

        return try {
            if (w <= dncnnTileSize && h <= dncnnTileSize) {
                if (w == dncnnTileSize && h == dncnnTileSize) {
                    yChan.convertTo(floatMat, CvType.CV_32FC1, 1.0 / 255.0)
                } else {
                    Core.copyMakeBorder(
                        yChan,
                        padMat,
                        0,
                        dncnnTileSize - h,
                        0,
                        dncnnTileSize - w,
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

                val valid = if (w == dncnnTileSize && h == dncnnTileSize) outTileMat else Mat(outTileMat, Rect(0, 0, w, h))
                valid.convertTo(u8Out, CvType.CV_8UC1, 255.0)
                if (valid !== outTileMat) valid.release()
                Core.addWeighted(u8Out, blend, yChan, 1.0 - blend, 0.0, yChan)
                return true
            }

            // High-res frame: extract low-freq DnCNN noise residual (denoised256 - orig256) so fine detail never blurs!
            Imgproc.resize(yChan, padMat, Size(dncnnTileSize.toDouble(), dncnnTileSize.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
            padMat.convertTo(floatMat, CvType.CV_32FC1, 1.0 / 255.0)
            floatMat.get(0, 0, inTmp)
            inBuf.clear()
            inBuf.put(inTmp, 0, dncnnTilePixels)
            inBuf.rewind()

            runDnCnnWithFallback(shape, inBuf, outArr, logger)
            outTileMat.put(0, 0, outArr)

            val resF = dncnnResidualF ?: Mat(dncnnTileSize, dncnnTileSize, CvType.CV_32FC1).also { dncnnResidualF = it }
            val resUpF = dncnnResidualUpF ?: Mat().also { dncnnResidualUpF = it }
            val yFloat = dncnnYFloatMat ?: Mat().also { dncnnYFloatMat = it }

            Core.subtract(outTileMat, floatMat, resF)
            Imgproc.resize(resF, resUpF, Size(w.toDouble(), h.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
            yChan.convertTo(yFloat, CvType.CV_32FC1, 1.0 / 255.0)
            Core.scaleAdd(resUpF, blend, yFloat, yFloat)

            // Suppress fine high-frequency grain gently via fast edge-gated 3x3 smooth on flat areas only
            yFloat.convertTo(u8Out, CvType.CV_8UC1, 255.0)
            val grainSigmaBlend = (blend * 0.28).coerceIn(0.04, 0.24)
            Imgproc.GaussianBlur(u8Out, dncnnAccumMat!!, Size(3.0, 3.0), 0.75)
            // Keep crisp gradient edges untouched using absdiff edge mask
            Core.absdiff(u8Out, dncnnAccumMat!!, u8Out)
            val flatMask = dncnnWeightMat!!
            Imgproc.threshold(u8Out, flatMask, 9.0, 255.0, Imgproc.THRESH_BINARY_INV)
            flatMask.convertTo(u8Out, CvType.CV_8UC1)
            yFloat.convertTo(yChan, CvType.CV_8UC1, 255.0)
            Core.addWeighted(dncnnAccumMat!!, grainSigmaBlend, yChan, 1.0 - grainSigmaBlend, 0.0, dncnnAccumMat!!)
            dncnnAccumMat!!.copyTo(yChan, u8Out)
            true
        } catch (t: Throwable) {
            logger?.warn("DnCNN fast skip (${t.message})")
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
     * High-Speed Real-ESRGAN Super-Resolution:
     * - Scales neural working size dynamically using AiTilingMath.computeNeuralWorkingSize (prevents 4x bloat!)
     * - Zero-accumulation Single-Tile Fast Path when grid.totalTiles == 1 (eliminates 4K FP32 accum buffers!)
     * - Motion-gated Static Tile Temporal Reuse + double-buffered direct FloatBuffer prep for multi-tile
     * - Live fallback along Real-ESRGAN's ranked GPU -> NPU -> CPU chain and speed budget enforcement
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
        detailRecovery: Float = 0.65f,
        deblurStrength: Float = 0.50f,
        aiSharpening: Float = 0.32f,
        fallbackSharpnessBoost: Float = 0.28f,
        faces: List<Rect> = emptyList(),
        faceEnhanceStrength: Float = 0.65f,
        isSceneCut: Boolean = false,
        modeBudgetFrameMs: Long = AiSpeedPolicy.MAX_ACCEPTABLE_AI_FRAME_MS,
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
                sharpnessBoost = (fallbackSharpnessBoost + deblurStrength * 0.25f).coerceIn(0.12f, 0.58f),
                contrastBoost = 1.03f,
                satBoost = 1.02f
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

        val safeTileSize = tileSize.coerceIn(128, 384)
        val cappedTiles = targetTilesPerFrame.coerceIn(1, 4)
        val (srInW, srInH) = AiTilingMath.computeNeuralWorkingSize(
            inW = inW,
            inH = inH,
            outW = safeOutW,
            outH = safeOutH,
            tileSize = safeTileSize,
            maxTilesCap = cappedTiles
        )

        val grid = AiTilingMath.planUpscaleTileGrid(
            inW = srInW,
            inH = srInH,
            outW = safeOutW,
            outH = safeOutH,
            tileSize = safeTileSize,
            maxTilesCap = cappedTiles
        )

        ensureNativeWorkMats(inW, inH, safeOutW, safeOutH, allocateAccum = grid.totalTiles > 1)
        ensureTileBuffers(safeTileSize)

        val localRgb = srcRgbMat!!
        Imgproc.cvtColor(srcBgr, localRgb, Imgproc.COLOR_BGR2RGB)

        val srSourceRgb = if (srInW == inW && srInH == inH) {
            localRgb
        } else {
            val work = srWorkRgbMat ?: Mat().also { srWorkRgbMat = it }
            Imgproc.resize(localRgb, work, Size(srInW.toDouble(), srInH.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
            work
        }

        val fixedShape = longArrayOf(1L, 3L, safeTileSize.toLong(), safeTileSize.toLong())
        val outResized = outResizedFloat!!

        // Fast Single-Tile Zero-Accumulation Path: bypasses 4K FP32 accum, feathering & worker thread overhead
        if (grid.totalTiles == 1) {
            val tile = grid.tiles[0]
            val padMat = tilePaddedMatA!!
            val floatHwc = tileFloatHwcA!!
            val chIn = tileChannelsA!!
            val directBuf = tileInBufA!!
            val planeTmp = tileInPlaneTmpA

            val inRect = Rect(tile.x, tile.y, tile.width, tile.height)
            val roi = Mat(srSourceRgb, inRect)
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
                Imgproc.resize(roi, padMat, Size(safeTileSize.toDouble(), safeTileSize.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
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

            val tTile0 = System.currentTimeMillis()
            val dstSize = Size(tile.outWidth.toDouble(), tile.outHeight.toDouble())
            runSrTileAndResizeDirect(
                inputBuf = directBuf,
                shape = fixedShape,
                tileSize = safeTileSize,
                validSrW = validSrW,
                validSrH = validSrH,
                dstSize = dstSize,
                outResizedHwc = outResized,
                logger = logger
            )
            val tileMs = (System.currentTimeMillis() - tTile0).coerceAtLeast(1L)
            onTileProgress(1, 1, safeTileSize)

            outResized.convertTo(outRgb8u!!, CvType.CV_8UC3, 255.0)
            fuseHighResDetailAndDeblur(
                localRgb = localRgb,
                targetRgb8u = outRgb8u!!,
                inW = inW,
                inH = inH,
                srInW = srInW,
                srInH = srInH,
                outW = safeOutW,
                outH = safeOutH,
                aiStrength = aiStrength,
                detailRecovery = detailRecovery,
                deblurStrength = deblurStrength,
                aiSharpening = aiSharpening
            )

            val hitLimit = AiSpeedPolicy.shouldEngageLightweightFallback(
                measuredTileMs = tileMs,
                tilesPerFrame = 1,
                estimatedFrameCount = estimatedTotalFrames,
                modeBudgetFrameMs = modeBudgetFrameMs
            )

            val finalBgr = Mat(safeOutH, safeOutW, CvType.CV_8UC3)
            Imgproc.cvtColor(outRgb8u!!, finalBgr, Imgproc.COLOR_RGB2BGR)
            return TileProcessOutcome(
                outputBgr = finalBgr,
                usedLightweightFallback = hitLimit,
                fallbackReason = if (hitLimit) "Speed budget guard: ${tileMs}ms/tile > ${modeBudgetFrameMs}ms budget" else null,
                activeAccelerationLabel = activeProviderDescription,
                faceAiMs = 0L,
                staticTilesReused = 0
            )
        }

        // Multi-tile path (2-4 tiles max)
        val aiBlend = aiStrength.toDouble().coerceIn(0.25, 0.96)
        if (aiBlend < 0.88) {
            if (inW == safeOutW && inH == safeOutH) {
                localRgb.copyTo(baseRgb8u!!)
            } else {
                Imgproc.resize(localRgb, baseRgb8u!!, Size(safeOutW.toDouble(), safeOutH.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
            }
        }

        accumRgbMat!!.setTo(Scalar(0.0, 0.0, 0.0))
        accumWeightMat!!.setTo(Scalar(0.0))

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
            val roi = Mat(srSourceRgb, inRect)
            val sig = srTileCache.computeTileSignature(roi)
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
                Imgproc.resize(roi, padMat, Size(safeTileSize.toDouble(), safeTileSize.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
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
                reusedFromCache = false,
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

            nextPrepFuture = if (i + 1 < tiles.size) {
                val nextIdx = (prepared.bufIndex + 1) and 1
                val nextTile = tiles[i + 1]
                prepExec.submit(Callable { prepareSingleTile(nextTile, nextIdx) })
            } else {
                null
            }

            val reused = srTileCache.tryReuseTileOutput(
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
                srTileCache.storeTileOutput(tile.index, prepared.signature, outResized)

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

            if (!reused && timedTiles >= 1 && AiSpeedPolicy.shouldEngageLightweightFallback(
                    measuredTileMs = tileTimingSumMs / timedTiles,
                    tilesPerFrame = grid.totalTiles,
                    estimatedFrameCount = estimatedTotalFrames,
                    modeBudgetFrameMs = modeBudgetFrameMs
                )
            ) {
                val avgTileMs = tileTimingSumMs / timedTiles
                val reason = "Speed budget fallback: ${safeTileSize}px tile ${avgTileMs}ms x ${grid.totalTiles} exceeds ${modeBudgetFrameMs}ms"
                logger?.warn(reason)
                nextPrepFuture?.cancel(true)
                val fallbackSrc = baseRgb8u?.takeIf { aiBlend < 0.88 } ?: localRgb.let {
                    Imgproc.resize(it, baseRgb8u!!, Size(safeOutW.toDouble(), safeOutH.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
                    baseRgb8u!!
                }
                val lw = enhanceFastLightweightCopy(
                    fallbackSrc,
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

        val wMat = accumWeightMat!!
        val invSingle = invWeightSingleMat!!
        val invThree = invWeightThreeMat!!
        Core.max(wMat, weightClampFloorMat!!, wMat)
        Core.divide(1.0, wMat, invSingle)
        Core.merge(listOf(invSingle, invSingle, invSingle), invThree)
        Core.multiply(accumRgbMat!!, invThree, accumRgbMat!!)
        accumRgbMat!!.convertTo(outRgb8u!!, CvType.CV_8UC3, 255.0)

        fuseHighResDetailAndDeblur(
            localRgb = localRgb,
            targetRgb8u = outRgb8u!!,
            inW = inW,
            inH = inH,
            srInW = srInW,
            srInH = srInH,
            outW = safeOutW,
            outH = safeOutH,
            aiStrength = aiStrength,
            detailRecovery = detailRecovery,
            deblurStrength = deblurStrength,
            aiSharpening = aiSharpening
        )

        var faceMs = 0L
        if (faces.isNotEmpty() && faceEnhanceStrength >= 0.25f) {
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

    /**
     * Frequency-Aware Detail-Preserving Neural Fusion + Deblurring:
     * 1. When Real-ESRGAN ran on a downscaled working canvas (srInW < inW) on 720p/1080p inputs,
     *    injects the native high-frequency micro-texture from the original high-res frame so
     *    neural super-resolution never smudges crisp textures down to working-canvas resolution!
     * 2. Applies halo-clamped adaptive unsharp/deblurring driven by [detailRecovery],
     *    [deblurStrength], and [aiSharpening].
     */
    private fun fuseHighResDetailAndDeblur(
        localRgb: Mat,
        targetRgb8u: Mat,
        inW: Int,
        inH: Int,
        srInW: Int,
        srInH: Int,
        outW: Int,
        outH: Int,
        aiStrength: Float,
        detailRecovery: Float,
        deblurStrength: Float,
        aiSharpening: Float
    ) {
        val aiBlend = aiStrength.toDouble().coerceIn(0.28, 0.96)
        val base = baseRgb8u!!
        if (inW == outW && inH == outH) {
            localRgb.copyTo(base)
        } else {
            Imgproc.resize(localRgb, base, Size(outW.toDouble(), outH.toDouble()), 0.0, 0.0, Imgproc.INTER_CUBIC)
        }

        if (aiBlend < 0.88) {
            Core.addWeighted(targetRgb8u, aiBlend, base, 1.0 - aiBlend, 0.0, targetRgb8u)
        }

        // Inject native high-frequency details whenever SR input was downscaled relative to input
        if (srInW < (inW * 0.85).toInt()) {
            val baseBlur = baseLowBlur8u ?: Mat().also { baseLowBlur8u = it }
            val highS16 = baseHighDetailS16 ?: Mat().also { baseHighDetailS16 = it }
            val targetS16 = outRgbS16 ?: Mat().also { outRgbS16 = it }
            val baseS16Tmp = Mat()
            val blurS16Tmp = Mat()
            try {
                Imgproc.GaussianBlur(base, baseBlur, Size(0.0, 0.0), 1.6)
                base.convertTo(baseS16Tmp, CvType.CV_16SC3)
                baseBlur.convertTo(blurS16Tmp, CvType.CV_16SC3)
                Core.subtract(baseS16Tmp, blurS16Tmp, highS16)

                targetRgb8u.convertTo(targetS16, CvType.CV_16SC3)
                val hfGain = (0.45 + detailRecovery * 0.55).coerceIn(0.45, 1.05)
                Core.scaleAdd(highS16, hfGain, targetS16, targetS16)
                targetS16.convertTo(targetRgb8u, CvType.CV_8UC3)
            } finally {
                baseS16Tmp.release()
                blurS16Tmp.release()
            }
        }

        // Apply Halo-Clamped Multi-Scale Deblur + AI Sharpening (activates previously dead params!)
        val deblurAmt = (deblurStrength * 0.28 + aiSharpening * 0.34).coerceIn(0.06, 0.52)
        if (deblurAmt > 0.05) {
            applyHaloClampedSharpen(targetRgb8u, amount = deblurAmt, sigma = if (outW > 1920) 1.35 else 1.05)
        }
    }

    private fun applyHaloClampedSharpen(mat8u: Mat, amount: Double, sigma: Double) {
        val blur = lwBlurMat ?: Mat().also { lwBlurMat = it }
        val detail = lwDetailMat ?: Mat().also { lwDetailMat = it }
        val minN = lwMinNMat ?: Mat().also { lwMinNMat = it }
        val maxN = lwMaxNMat ?: Mat().also { lwMaxNMat = it }
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
        try {
            Imgproc.erode(mat8u, minN, kernel)
            Imgproc.dilate(mat8u, maxN, kernel)
            Imgproc.GaussianBlur(mat8u, blur, Size(0.0, 0.0), sigma)
            Core.addWeighted(mat8u, 1.0 + amount, blur, -amount, 0.0, detail)
            // Clamp overshoot to 3x3 neighborhood bounds to avoid white/black halo ringing
            Core.max(detail, minN, detail)
            Core.min(detail, maxN, mat8u)
        } finally {
            kernel.release()
        }
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

        for (face in faces.take(1)) {
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
        applyHaloClampedSharpen(out, amount = sharpnessBoost.toDouble().coerceIn(0.08, 0.52), sigma = 1.05)
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
        val dstW = dstSize.width.roundToInt()
        val dstH = dstSize.height.roundToInt()
        val exactMatch = (dstW == safeValidW && dstH == safeValidH)

        for (c in 0 until 3) {
            outBuf.position(c * outPixels)
            outBuf.get(tileOutPlaneTmp, 0, outPixels)
            fullPlane.put(0, 0, tileOutPlaneTmp)
            val srcRegion = if (needCrop) Mat(fullPlane, Rect(0, 0, safeValidW, safeValidH)) else fullPlane
            if (exactMatch) {
                srcRegion.copyTo(smallCh[c])
            } else {
                Imgproc.resize(srcRegion, smallCh[c], dstSize, 0.0, 0.0, Imgproc.INTER_LINEAR)
            }
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

    private fun ensureNativeWorkMats(inW: Int, inH: Int, outW: Int, outH: Int, allocateAccum: Boolean = true) {
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
            accumRgbMat?.release(); accumRgbMat = null
            accumWeightMat?.release(); accumWeightMat = null
            invWeightSingleMat?.release(); invWeightSingleMat = null
            invWeightThreeMat?.release(); invWeightThreeMat = null
            weightClampFloorMat?.release(); weightClampFloorMat = null
            outRgb8u?.release()
            clearFeatherCache()

            baseRgb8u = Mat(outH, outW, CvType.CV_8UC3)
            outRgb8u = Mat(outH, outW, CvType.CV_8UC3)
            outResizedFloat = outResizedFloat ?: Mat()
            weightedTileFloat = weightedTileFloat ?: Mat()
        }
        if (allocateAccum && (accumRgbMat == null || accumRgbMat?.cols() != outW || accumRgbMat?.rows() != outH)) {
            accumRgbMat?.release()
            accumWeightMat?.release()
            invWeightSingleMat?.release()
            invWeightThreeMat?.release()
            weightClampFloorMat?.release()
            accumRgbMat = Mat(outH, outW, CvType.CV_32FC3)
            accumWeightMat = Mat(outH, outW, CvType.CV_32FC1)
            invWeightSingleMat = Mat(outH, outW, CvType.CV_32FC1)
            invWeightThreeMat = Mat(outH, outW, CvType.CV_32FC3)
            weightClampFloorMat = Mat(outH, outW, CvType.CV_32FC1, Scalar(1e-5))
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
        if (yuNetRunnerLazy.isInitialized()) {
            yuNetRunner.releaseWorkBuffers()
        }

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
        dncnnResidualF?.release(); dncnnResidualF = null
        dncnnResidualUpF?.release(); dncnnResidualUpF = null
        dncnnYFloatMat?.release(); dncnnYFloatMat = null

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
        srWorkRgbMat?.release(); srWorkRgbMat = null
        baseRgb8u?.release(); baseRgb8u = null
        baseLowBlur8u?.release(); baseLowBlur8u = null
        baseHighDetailS16?.release(); baseHighDetailS16 = null
        outRgbS16?.release(); outRgbS16 = null
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
        lwBlurFineMat?.release(); lwBlurFineMat = null
        lwMinNMat?.release(); lwMinNMat = null
        lwMaxNMat?.release(); lwMaxNMat = null
        clearFeatherCache()
    }

    private fun ensureAssetCopied(assetPath: String, destFile: File): File {
        destFile.parentFile?.mkdirs()
        val expectedLength = runCatching {
            context.assets.openFd(assetPath).use { it.length }
        }.getOrDefault(-1L)
        if (destFile.exists() && destFile.length() > 0L && (expectedLength <= 0L || destFile.length() == expectedLength)) {
            return destFile
        }
        val tmpFile = File(destFile.parentFile, "${destFile.name}.tmp")
        if (tmpFile.exists()) tmpFile.delete()
        context.assets.open(assetPath).use { input ->
            tmpFile.outputStream().use { output ->
                input.copyTo(output)
                output.flush()
            }
        }
        if (tmpFile.exists() && tmpFile.length() > 0L) {
            if (destFile.exists()) destFile.delete()
            if (!tmpFile.renameTo(destFile)) {
                tmpFile.copyTo(destFile, overwrite = true)
                tmpFile.delete()
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
        if (yuNetRunnerLazy.isInitialized()) {
            yuNetRunner.close()
        }
    }

    fun close() {
        releaseWorkBuffers()
        gpuProcessor.release()
        closeModelSessionsOnly()
    }
}
