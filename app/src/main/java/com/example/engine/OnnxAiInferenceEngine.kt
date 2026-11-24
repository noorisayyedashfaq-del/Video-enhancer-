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
import java.util.concurrent.TimeUnit
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.log10
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

/**
 * Pure-Kotlin Q4 fixed-point DnCNN residual math & edge-gated flat-block grain suppressor.
 * Matches the native OpenCV CV_16SC1 Q4 pipeline numerically and allows deterministic
 * host-JVM unit verification and microbenchmarking of denoising quality + edge retention.
 */
object DnCnnResidualMath {
    const val Q4_SHIFT = 4
    const val Q4_SCALE = 16.0
    const val EDGE_DIFF_THRESHOLD = 9

    fun clampStrength(strength: Float): Double = strength.toDouble().coerceIn(0.10, 0.95)

    fun quantizeResidualToQ4(
        origNorm: FloatArray,
        denoisedNorm: FloatArray,
        strength: Float,
        dstQ4: ShortArray
    ) {
        val blendScale = clampStrength(strength) * 255.0 * Q4_SCALE
        val n = minOf(origNorm.size, denoisedNorm.size, dstQ4.size)
        for (i in 0 until n) {
            val q = ((denoisedNorm[i] - origNorm[i]) * blendScale).roundToInt().coerceIn(-2048, 2048)
            dstQ4[i] = q.toShort()
        }
    }

    fun applyQ4ResidualAndEdgeGatedSmooth(
        srcY: ByteArray,
        width: Int,
        height: Int,
        resQ4Full: ShortArray,
        strength: Float,
        dstY: ByteArray
    ) {
        val blend = clampStrength(strength)
        val grainSigmaBlend = (blend * 0.28).coerceIn(0.04, 0.24)
        val total = width * height
        val yDen = IntArray(total)

        for (i in 0 until total) {
            val base = srcY[i].toInt() and 0xFF
            val corr = (resQ4Full[i].toInt() + 8) shr Q4_SHIFT
            yDen[i] = (base + corr).coerceIn(0, 255)
        }

        val invBlend = 1.0 - grainSigmaBlend
        for (y in 0 until height) {
            val yUp = if (y > 0) y - 1 else 0
            val yDn = if (y + 1 < height) y + 1 else height - 1
            val rowOff = y * width
            val rowUp = yUp * width
            val rowDn = yDn * width

            for (x in 0 until width) {
                val xLt = if (x > 0) x - 1 else 0
                val xRt = if (x + 1 < width) x + 1 else width - 1
                val center = yDen[rowOff + x]

                // 3x3 Gaussian weights [1 2 1; 2 4 2; 1 2 1] / 16
                val blur3x3 = (
                    (center shl 2) +
                        ((yDen[rowUp + x] + yDen[rowDn + x] + yDen[rowOff + xLt] + yDen[rowOff + xRt]) shl 1) +
                        yDen[rowUp + xLt] + yDen[rowUp + xRt] + yDen[rowDn + xLt] + yDen[rowDn + xRt] + 8
                    ) shr 4

                val out = if (abs(center - blur3x3) <= EDGE_DIFF_THRESHOLD) {
                    (blur3x3 * grainSigmaBlend + center * invBlend).roundToInt().coerceIn(0, 255)
                } else {
                    center
                }
                dstY[rowOff + x] = out.toByte()
            }
        }
    }

    fun computePsnrDb(refY: ByteArray, testY: ByteArray): Double {
        val n = minOf(refY.size, testY.size)
        if (n <= 0) return 0.0
        var mseSum = 0.0
        for (i in 0 until n) {
            val d = (refY[i].toInt() and 0xFF) - (testY[i].toInt() and 0xFF)
            mseSum += (d * d).toDouble()
        }
        val mse = mseSum / n.toDouble()
        if (mse <= 1e-6) return 99.0
        return 10.0 * log10((255.0 * 255.0) / mse)
    }

    fun computeEdgeJump(frameY: ByteArray, width: Int, row: Int, colLeft: Int, colRight: Int): Int {
        val left = frameY[row * width + colLeft].toInt() and 0xFF
        val right = frameY[row * width + colRight].toInt() and 0xFF
        return abs(right - left)
    }
}

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

    // Double-buffered direct ByteBuffers + off-heap slice Mats + cached OnnxTensors (zero FloatArray/Tensor churn)
    private var allocTileSize: Int = 0
    private var tileInByteBufA: ByteBuffer? = null
    private var tileInByteBufB: ByteBuffer? = null
    private var tileInBufA: FloatBuffer? = null
    private var tileInBufB: FloatBuffer? = null
    private var tileCachedTensorA: OnnxTensor? = null
    private var tileCachedTensorB: OnnxTensor? = null
    private var tileOutDirectByteBuf: ByteBuffer? = null
    private var tileOutDirectFloatBuf: FloatBuffer? = null
    private var tileOutPlaneWraps: List<Mat>? = null
    private var tilePrepExecutor: ExecutorService? = null

    data class DnCnnPrefetchToken(
        val frameIndex: Int,
        val strength: Float,
        val srcWidth: Int,
        val srcHeight: Int,
        val future: Future<Boolean>
    )

    // Double-buffered zero-copy off-heap DnCNN [1, 1, 256, 256] (Slot 0 inline, Slot 1 async prefetch)
    private val dncnnTileSize = 256
    private val dncnnTilePixels = dncnnTileSize * dncnnTileSize
    private val dncnnShape = longArrayOf(1L, 1L, dncnnTileSize.toLong(), dncnnTileSize.toLong())
    private val dncnnSessionLock = Any()

    private val dncnnInByteBufs = arrayOfNulls<ByteBuffer>(2)
    private val dncnnInFloatBufs = arrayOfNulls<FloatBuffer>(2)
    private val dncnnInMats = arrayOfNulls<Mat>(2)
    private val dncnnCachedInTensors = arrayOfNulls<OnnxTensor>(2)
    private val dncnnOutByteBufs = arrayOfNulls<ByteBuffer>(2)
    private val dncnnOutFloatBufs = arrayOfNulls<FloatBuffer>(2)
    private val dncnnOutMats = arrayOfNulls<Mat>(2)
    private val dncnnPadMats = arrayOfNulls<Mat>(2)
    private var dncnnAsyncBgrSmall: Mat? = null
    private var dncnnAsyncGrayTmp: Mat? = null
    private var dncnnAsyncExecutor: ExecutorService? = null

    // High-res Q4 fixed-point residual + edge-gated grain suppression scratch Mats (strict fixed CvTypes)
    private var dncnnResidualF: Mat? = null
    private var dncnnResQ4Mat256: Mat? = null
    private var dncnnResQ4Up: Mat? = null
    private var dncnnYQ4Mat: Mat? = null
    private var dncnnU8Mat: Mat? = null
    private var dncnnBlurU8Mat: Mat? = null
    private var dncnnDiffU8Mat: Mat? = null
    private var dncnnFlatMaskU8Mat: Mat? = null
    private var dncnnBlendedU8Mat: Mat? = null

    // Static tile temporal reuse cache for Real-ESRGAN tiles (DnCNN never skips frames)
    private val srTileCache = StaticTileReuseCache()

    // Reusable OpenCV native C++ scratch Mats
    private var srcRgbMat: Mat? = null
    private var srWorkRgbMat: Mat? = null
    private var baseRgb8u: Mat? = null
    private var baseLowBlur8u: Mat? = null
    private var baseHighDetailS16: Mat? = null
    private var baseS16Tmp: Mat? = null
    private var blurS16Tmp: Mat? = null
    private var outRgbS16: Mat? = null
    private var tilePaddedMatA: Mat? = null
    private var tilePaddedMatB: Mat? = null
    private var tileFloatHwcA: Mat? = null
    private var tileFloatHwcB: Mat? = null
    private var tileChannelsA: List<Mat>? = null
    private var tileChannelsB: List<Mat>? = null
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

    // Lightweight OpenCV & halo-clamped sharpen scratch Mats
    private var lwBlurMat: Mat? = null
    private var lwDetailMat: Mat? = null
    private var lwBlurFineMat: Mat? = null
    private var lwMinNMat: Mat? = null
    private var lwMaxNMat: Mat? = null
    private var morphKernel3x3: Mat? = null

    // Feather weight Mat cache keyed by tile dimensions & overlap
    private val featherMaskCache = HashMap<Long, Mat>()

    val isRealEsrganReady: Boolean get() = srActiveIndex < srRankedSessions.size
    val isDncnnReady: Boolean get() = dncnnActiveIndex < dncnnRankedSessions.size
    val isGlesGpuReady: Boolean get() = glesGpuAvailable
    val providerLabel: String get() = activeProviderDescription
    val benchmarkSummary: String get() = benchmarkDetails
    val initDurationMs: Long get() = lastInitDurationMs
    val totalStaticTilesReused: Int get() = srTileCache.totalTilesReused

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
     * Submits asynchronous DnCNN neural inference on Slot 1 for the next queued BGR frame
     * while the current frame completes Real-ESRGAN / Color / HardwareVideoEncoder stages.
     * Pre-resizes to tiny 256x256 in <0.4ms on submission so the decoder's pooled Mat stays decoupled.
     */
    fun prefetchDnCnnAsync(
        frameIndex: Int,
        srcBgr: Mat,
        strength: Float,
        logger: EngineDiagnosticsLogger? = null
    ): DnCnnPrefetchToken? {
        if (!isDncnnReady || srcBgr.empty()) return null
        val w = srcBgr.cols()
        val h = srcBgr.rows()
        if (w < 8 || h < 8) return null

        return try {
            ensureDnCnnSlotBuffers(1)
            val preBgr = dncnnAsyncBgrSmall ?: Mat().also { dncnnAsyncBgrSmall = it }
            if (w <= dncnnTileSize && h <= dncnnTileSize) {
                srcBgr.copyTo(preBgr)
            } else {
                Imgproc.resize(srcBgr, preBgr, Size(dncnnTileSize.toDouble(), dncnnTileSize.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
            }

            val exec = dncnnAsyncExecutor ?: Executors.newSingleThreadExecutor { r ->
                Thread(r, "ClearKit-DnCNN-Async").apply { isDaemon = true }
            }.also { dncnnAsyncExecutor = it }

            val fut = exec.submit(Callable {
                try {
                    val padMat1 = dncnnPadMats[1]!!
                    val inFloatMat1 = dncnnInMats[1]!!
                    if (w <= dncnnTileSize && h <= dncnnTileSize) {
                        val grayTmp = dncnnAsyncGrayTmp ?: Mat().also { dncnnAsyncGrayTmp = it }
                        Imgproc.cvtColor(preBgr, grayTmp, Imgproc.COLOR_BGR2GRAY)
                        if (w == dncnnTileSize && h == dncnnTileSize) {
                            grayTmp.convertTo(inFloatMat1, CvType.CV_32FC1, 1.0 / 255.0)
                        } else {
                            Core.copyMakeBorder(
                                grayTmp,
                                padMat1,
                                0,
                                dncnnTileSize - h,
                                0,
                                dncnnTileSize - w,
                                Core.BORDER_REFLECT_101
                            )
                            padMat1.convertTo(inFloatMat1, CvType.CV_32FC1, 1.0 / 255.0)
                        }
                    } else {
                        // Linear operators (INTER_AREA resize & BGR2GRAY) commute cleanly: 65K pixels instead of 2M!
                        Imgproc.cvtColor(preBgr, padMat1, Imgproc.COLOR_BGR2GRAY)
                        padMat1.convertTo(inFloatMat1, CvType.CV_32FC1, 1.0 / 255.0)
                    }
                    runDnCnnSlotWithFallback(slot = 1, logger = logger)
                    true
                } catch (t: Throwable) {
                    logger?.warn("DnCNN async prefetch failed (${t.message})")
                    false
                }
            })

            DnCnnPrefetchToken(
                frameIndex = frameIndex,
                strength = strength,
                srcWidth = w,
                srcHeight = h,
                future = fut
            )
        } catch (t: Throwable) {
            logger?.warn("DnCNN async schedule skipped (${t.message})")
            null
        }
    }

    /**
     * Zero-skip, zero-copy Edge-Preserving DnCNN Neural Luminance Denoising ([1, 1, 256, 256] ONNX):
     * - Consumes an already-completed async Slot 1 prediction when [prefetchedToken] matches [frameIndex],
     *   or executes zero-copy Slot 0 directly into off-heap OnnxTensor ByteBuffer.
     * - Applies Q4 fixed-point residual extraction (CV_16SC1) + edge-gated 3x3 flat-region grain suppression
     *   so high-resolution textures, skin pores, hair strands, and sharp edges remain unblurred.
     *
     * Returns Pair(appliedDnCnn, usedAsyncPrefetchHit).
     */
    fun denoiseLuminanceYWithDnCnnWithPrefetch(
        yChan: Mat,
        strength: Float,
        isSceneCut: Boolean = false,
        frameIndex: Int = -1,
        prefetchedToken: DnCnnPrefetchToken? = null,
        logger: EngineDiagnosticsLogger? = null
    ): Pair<Boolean, Boolean> {
        if (!isDncnnReady) return false to false
        val w = yChan.cols()
        val h = yChan.rows()
        if (w < 8 || h < 8) return false to false
        val blend = DnCnnResidualMath.clampStrength(strength)

        var activeSlot = 0
        var usedPrefetchHit = false
        if (prefetchedToken != null &&
            frameIndex >= 0 &&
            prefetchedToken.frameIndex == frameIndex &&
            prefetchedToken.srcWidth == w &&
            prefetchedToken.srcHeight == h
        ) {
            val prefetchOk = runCatching { prefetchedToken.future.get() }.getOrDefault(false)
            if (prefetchOk) {
                activeSlot = 1
                usedPrefetchHit = true
            }
        } else if (prefetchedToken != null && !prefetchedToken.future.isDone) {
            // Do not leak in-flight Slot 1 work on mismatch
            runCatching { prefetchedToken.future.cancel(false) }
        }

        ensureDnCnnSlotBuffers(activeSlot)
        ensureDnCnnHighResMats(w, h)

        val padMat = dncnnPadMats[activeSlot]!!
        val inFloatMat = dncnnInMats[activeSlot]!!
        val outTileMat = dncnnOutMats[activeSlot]!!
        val u8Out = dncnnU8Mat!!

        return try {
            if (!usedPrefetchHit) {
                if (w <= dncnnTileSize && h <= dncnnTileSize) {
                    if (w == dncnnTileSize && h == dncnnTileSize) {
                        yChan.convertTo(inFloatMat, CvType.CV_32FC1, 1.0 / 255.0)
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
                        padMat.convertTo(inFloatMat, CvType.CV_32FC1, 1.0 / 255.0)
                    }
                } else {
                    Imgproc.resize(yChan, padMat, Size(dncnnTileSize.toDouble(), dncnnTileSize.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
                    padMat.convertTo(inFloatMat, CvType.CV_32FC1, 1.0 / 255.0)
                }
                runDnCnnSlotWithFallback(slot = activeSlot, logger = logger)
            }

            if (w <= dncnnTileSize && h <= dncnnTileSize) {
                val valid = if (w == dncnnTileSize && h == dncnnTileSize) outTileMat else Mat(outTileMat, Rect(0, 0, w, h))
                valid.convertTo(u8Out, CvType.CV_8UC1, 255.0)
                if (valid !== outTileMat) valid.release()
                Core.addWeighted(u8Out, blend, yChan, 1.0 - blend, 0.0, yChan)
                return true to usedPrefetchHit
            }

            // High-res frame: Q4 fixed-point residual (denoised256 - orig256) in CV_16SC1 for SIMD speed & sharp edges
            val resF = dncnnResidualF!!
            val resQ4_256 = dncnnResQ4Mat256!!
            val resQ4Up = dncnnResQ4Up!!
            val yQ4 = dncnnYQ4Mat!!
            val blurU8 = dncnnBlurU8Mat!!
            val diffU8 = dncnnDiffU8Mat!!
            val flatMask = dncnnFlatMaskU8Mat!!
            val blendedU8 = dncnnBlendedU8Mat!!

            Core.subtract(outTileMat, inFloatMat, resF)
            resF.convertTo(resQ4_256, CvType.CV_16SC1, blend * 255.0 * DnCnnResidualMath.Q4_SCALE)
            Imgproc.resize(resQ4_256, resQ4Up, Size(w.toDouble(), h.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)

            yChan.convertTo(yQ4, CvType.CV_16SC1, DnCnnResidualMath.Q4_SCALE)
            Core.add(yQ4, resQ4Up, yQ4)
            yQ4.convertTo(u8Out, CvType.CV_8UC1, 1.0 / DnCnnResidualMath.Q4_SCALE, 0.0)

            // Fast edge-gated 3x3 smoothing on flat low-gradient blocks only
            val grainSigmaBlend = (blend * 0.28).coerceIn(0.04, 0.24)
            Imgproc.GaussianBlur(u8Out, blurU8, Size(3.0, 3.0), 0.75)
            Core.absdiff(u8Out, blurU8, diffU8)
            Imgproc.threshold(diffU8, flatMask, DnCnnResidualMath.EDGE_DIFF_THRESHOLD.toDouble(), 255.0, Imgproc.THRESH_BINARY_INV)
            Core.addWeighted(blurU8, grainSigmaBlend, u8Out, 1.0 - grainSigmaBlend, 0.0, blendedU8)
            u8Out.copyTo(yChan)
            blendedU8.copyTo(yChan, flatMask)
            true to usedPrefetchHit
        } catch (t: Throwable) {
            logger?.warn("DnCNN execution error (${t.message})")
            false to usedPrefetchHit
        }
    }

    fun denoiseLuminanceYWithDnCnn(
        yChan: Mat,
        strength: Float,
        isSceneCut: Boolean = false,
        logger: EngineDiagnosticsLogger? = null
    ): Boolean {
        return denoiseLuminanceYWithDnCnnWithPrefetch(
            yChan = yChan,
            strength = strength,
            isSceneCut = isSceneCut,
            frameIndex = -1,
            prefetchedToken = null,
            logger = logger
        ).first
    }

    private fun runDnCnnSlotWithFallback(
        slot: Int,
        logger: EngineDiagnosticsLogger?
    ) {
        val inFloatBuf = dncnnInFloatBufs[slot]!!
        val outFloatBuf = dncnnOutFloatBufs[slot]!!

        synchronized(dncnnSessionLock) {
            val cachedTensor = dncnnCachedInTensors[slot] ?: run {
                inFloatBuf.clear()
                OnnxTensor.createTensor(env, inFloatBuf, dncnnShape).also {
                    dncnnCachedInTensors[slot] = it
                }
            }
            while (dncnnActiveIndex < dncnnRankedSessions.size) {
                val (backendId, session) = dncnnRankedSessions[dncnnActiveIndex]
                try {
                    val inputName = session.inputNames.first()
                    inFloatBuf.rewind()
                    session.run(mapOf(inputName to cachedTensor)).use { results ->
                        val outTensor = results[0] as OnnxTensor
                        val srcOutBuf = outTensor.floatBuffer
                        srcOutBuf.position(0)
                        outFloatBuf.clear()
                        outFloatBuf.put(srcOutBuf)
                        outFloatBuf.rewind()
                        val probe = outFloatBuf.get(dncnnTilePixels / 2)
                        if (probe.isNaN() || probe.isInfinite()) {
                            throw IllegalStateException("NaN output from ${backendId.displayLabel}")
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
            val cachedTensor = tileCachedTensorA!!

            val inRect = Rect(tile.x, tile.y, tile.width, tile.height)
            val roi = Mat(srSourceRgb, inRect)
            val sig = srTileCache.computeTileSignature(roi)
            var reusedSingle = outResized.cols() == tile.outWidth &&
                outResized.rows() == tile.outHeight &&
                srTileCache.tryReuseTileOutput(
                    tileKey = 0,
                    currentSig = sig,
                    isSceneCut = isSceneCut,
                    dstOutFloat = outResized
                )

            var tileMs = 1L
            if (!reusedSingle) {
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

                padMat.convertTo(floatHwc, CvType.CV_32FC3, 1.0 / 255.0)
                // Write NCHW channels directly into off-heap cached OnnxTensor memory (zero FloatArray copy)
                for (c in 0 until 3) {
                    Core.extractChannel(floatHwc, chIn[c], c)
                }
                directBuf.rewind()

                val tTile0 = System.currentTimeMillis()
                val dstSize = Size(tile.outWidth.toDouble(), tile.outHeight.toDouble())
                runSrTileAndResizeDirect(
                    inputBuf = directBuf,
                    cachedTensor = cachedTensor,
                    shape = fixedShape,
                    tileSize = safeTileSize,
                    validSrW = validSrW,
                    validSrH = validSrH,
                    dstSize = dstSize,
                    outResizedHwc = outResized,
                    logger = logger
                )
                tileMs = (System.currentTimeMillis() - tTile0).coerceAtLeast(1L)
                srTileCache.storeTileOutput(0, sig, outResized)
            }
            roi.release()
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

            val hitLimit = !reusedSingle && AiSpeedPolicy.shouldEngageLightweightFallback(
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
                staticTilesReused = if (reusedSingle) 1 else 0
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
            for (c in 0 until 3) {
                Core.extractChannel(floatHwc, chIn[c], c)
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

        val prepExec = tilePrepExecutor ?: Executors.newSingleThreadExecutor { r ->
            Thread(r, "ClearKit-TilePrep").apply { isDaemon = true }
        }.also { tilePrepExecutor = it }
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
                val activeTensor = if (prepared.bufIndex == 0) tileCachedTensorA!! else tileCachedTensorB!!
                runSrTileAndResizeDirect(
                    inputBuf = activeBuf,
                    cachedTensor = activeTensor,
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
            val bS16 = baseS16Tmp ?: Mat().also { baseS16Tmp = it }
            val blS16 = blurS16Tmp ?: Mat().also { blurS16Tmp = it }
            Imgproc.GaussianBlur(base, baseBlur, Size(0.0, 0.0), 1.6)
            base.convertTo(bS16, CvType.CV_16SC3)
            baseBlur.convertTo(blS16, CvType.CV_16SC3)
            Core.subtract(bS16, blS16, highS16)

            targetRgb8u.convertTo(targetS16, CvType.CV_16SC3)
            val hfGain = (0.45 + detailRecovery * 0.55).coerceIn(0.45, 1.05)
            Core.scaleAdd(highS16, hfGain, targetS16, targetS16)
            targetS16.convertTo(targetRgb8u, CvType.CV_8UC3)
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
        val kernel = morphKernel3x3 ?: Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0)).also {
            morphKernel3x3 = it
        }
        Imgproc.erode(mat8u, minN, kernel)
        Imgproc.dilate(mat8u, maxN, kernel)
        Imgproc.GaussianBlur(mat8u, blur, Size(0.0, 0.0), sigma)
        Core.addWeighted(mat8u, 1.0 + amount, blur, -amount, 0.0, detail)
        // Clamp overshoot to 3x3 neighborhood bounds to avoid white/black halo ringing
        Core.max(detail, minN, detail)
        Core.min(detail, maxN, mat8u)
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
        val cachedTensor = tileCachedTensorA!!
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
            for (c in 0 until 3) {
                Core.extractChannel(floatTileHwc, chIn[c], c)
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
                cachedTensor = cachedTensor,
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
        cachedTensor: OnnxTensor,
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
                current.run(mapOf(inputName to cachedTensor)).use { results ->
                    val outTensor = results[0] as OnnxTensor
                    val outBuf = outTensor.floatBuffer
                    val probe = outBuf.get(0)
                    if (probe.isNaN() || probe.isInfinite()) {
                        throw IllegalStateException("NaN output from ${backendId.displayLabel}")
                    }
                    extractValidPlanesAndResize(outBuf, tileSize, validSrW, validSrH, dstSize, outResizedHwc)
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
        val dstOutFloats = tileOutDirectFloatBuf!!
        val outWraps = tileOutPlaneWraps!!
        val smallCh = resizedOutChannels!!
        val safeValidW = validSrW.coerceIn(1, outDim)
        val safeValidH = validSrH.coerceIn(1, outDim)
        val needCrop = safeValidW < outDim || safeValidH < outDim
        val dstW = dstSize.width.roundToInt()
        val dstH = dstSize.height.roundToInt()
        val exactMatch = (dstW == safeValidW && dstH == safeValidH)

        // Bulk direct-to-direct native FloatBuffer transfer -> OpenCV Mat slices wrap memory directly with zero FloatArray copy!
        outBuf.position(0)
        val needed = 3 * outPixels
        if (outBuf.remaining() > needed) {
            val slicedSrc = outBuf.slice().apply { limit(needed) }
            dstOutFloats.clear()
            dstOutFloats.put(slicedSrc)
        } else {
            dstOutFloats.clear()
            dstOutFloats.put(outBuf)
        }
        dstOutFloats.rewind()

        for (c in 0 until 3) {
            val fullPlane = outWraps[c]
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

    private fun ensureDnCnnSlotBuffers(slot: Int) {
        if (dncnnInByteBufs[slot] == null) {
            val bytes = dncnnTilePixels * 4
            val inByteBuf = ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder())
            val outByteBuf = ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder())
            dncnnInByteBufs[slot] = inByteBuf
            dncnnInFloatBufs[slot] = inByteBuf.asFloatBuffer()
            dncnnInMats[slot] = Mat(dncnnTileSize, dncnnTileSize, CvType.CV_32FC1, inByteBuf)
            dncnnOutByteBufs[slot] = outByteBuf
            dncnnOutFloatBufs[slot] = outByteBuf.asFloatBuffer()
            dncnnOutMats[slot] = Mat(dncnnTileSize, dncnnTileSize, CvType.CV_32FC1, outByteBuf)
            dncnnPadMats[slot] = Mat(dncnnTileSize, dncnnTileSize, CvType.CV_8UC1)
        }
        if (dncnnResidualF == null) {
            dncnnResidualF = Mat(dncnnTileSize, dncnnTileSize, CvType.CV_32FC1)
            dncnnResQ4Mat256 = Mat(dncnnTileSize, dncnnTileSize, CvType.CV_16SC1)
        }
    }

    private fun ensureDnCnnHighResMats(w: Int, h: Int) {
        val curW = dncnnU8Mat?.cols() ?: -1
        val curH = dncnnU8Mat?.rows() ?: -1
        if (curW != w || curH != h) {
            dncnnResQ4Up?.release()
            dncnnYQ4Mat?.release()
            dncnnU8Mat?.release()
            dncnnBlurU8Mat?.release()
            dncnnDiffU8Mat?.release()
            dncnnFlatMaskU8Mat?.release()
            dncnnBlendedU8Mat?.release()

            dncnnResQ4Up = Mat(h, w, CvType.CV_16SC1)
            dncnnYQ4Mat = Mat(h, w, CvType.CV_16SC1)
            dncnnU8Mat = Mat(h, w, CvType.CV_8UC1)
            dncnnBlurU8Mat = Mat(h, w, CvType.CV_8UC1)
            dncnnDiffU8Mat = Mat(h, w, CvType.CV_8UC1)
            dncnnFlatMaskU8Mat = Mat(h, w, CvType.CV_8UC1)
            dncnnBlendedU8Mat = Mat(h, w, CvType.CV_8UC1)
        }
    }

    private fun ensureTileBuffers(tileSize: Int) {
        if (allocTileSize != tileSize) {
            runCatching { tileCachedTensorA?.close() }
            runCatching { tileCachedTensorB?.close() }
            tileCachedTensorA = null
            tileCachedTensorB = null
            tileChannelsA?.forEach { it.release() }
            tileChannelsB?.forEach { it.release() }
            tileOutPlaneWraps?.forEach { it.release() }
            resizedOutChannels?.forEach { it.release() }

            allocTileSize = tileSize
            val px = tileSize * tileSize
            val planeBytes = px * 4
            val inBytes = 3 * planeBytes
            val inByteA = ByteBuffer.allocateDirect(inBytes).order(ByteOrder.nativeOrder())
            val inByteB = ByteBuffer.allocateDirect(inBytes).order(ByteOrder.nativeOrder())
            val fBufA = inByteA.asFloatBuffer()
            val fBufB = inByteB.asFloatBuffer()
            tileInByteBufA = inByteA
            tileInByteBufB = inByteB
            tileInBufA = fBufA
            tileInBufB = fBufB

            val shape = longArrayOf(1L, 3L, tileSize.toLong(), tileSize.toLong())
            fBufA.clear()
            fBufB.clear()
            tileCachedTensorA = OnnxTensor.createTensor(env, fBufA, shape)
            tileCachedTensorB = OnnxTensor.createTensor(env, fBufB, shape)

            tileChannelsA = List(3) { c ->
                val slice = inByteA.duplicate().apply {
                    position(c * planeBytes)
                    limit((c + 1) * planeBytes)
                }.slice().order(ByteOrder.nativeOrder())
                Mat(tileSize, tileSize, CvType.CV_32FC1, slice)
            }
            tileChannelsB = List(3) { c ->
                val slice = inByteB.duplicate().apply {
                    position(c * planeBytes)
                    limit((c + 1) * planeBytes)
                }.slice().order(ByteOrder.nativeOrder())
                Mat(tileSize, tileSize, CvType.CV_32FC1, slice)
            }

            val outDim = tileSize * 4
            val outPlaneBytes = outDim * outDim * 4
            val outByteBuf = ByteBuffer.allocateDirect(3 * outPlaneBytes).order(ByteOrder.nativeOrder())
            tileOutDirectByteBuf = outByteBuf
            tileOutDirectFloatBuf = outByteBuf.asFloatBuffer()
            tileOutPlaneWraps = List(3) { c ->
                val slice = outByteBuf.duplicate().apply {
                    position(c * outPlaneBytes)
                    limit((c + 1) * outPlaneBytes)
                }.slice().order(ByteOrder.nativeOrder())
                Mat(outDim, outDim, CvType.CV_32FC1, slice)
            }

            resizedOutChannels = List(3) { Mat() }
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
        if (yuNetRunnerLazy.isInitialized()) {
            yuNetRunner.releaseWorkBuffers()
        }

        tilePrepExecutor?.shutdownNow()
        tilePrepExecutor = null

        dncnnAsyncExecutor?.let { exec ->
            exec.shutdown()
            runCatching { exec.awaitTermination(250L, TimeUnit.MILLISECONDS) }
            exec.shutdownNow()
        }
        dncnnAsyncExecutor = null

        allocTileSize = 0
        runCatching { tileCachedTensorA?.close() }; tileCachedTensorA = null
        runCatching { tileCachedTensorB?.close() }; tileCachedTensorB = null
        tileInByteBufA = null
        tileInByteBufB = null
        tileInBufA = null
        tileInBufB = null
        tileOutDirectByteBuf = null
        tileOutDirectFloatBuf = null

        for (s in 0 until 2) {
            runCatching { dncnnCachedInTensors[s]?.close() }
            dncnnCachedInTensors[s] = null
            dncnnInMats[s]?.release(); dncnnInMats[s] = null
            dncnnOutMats[s]?.release(); dncnnOutMats[s] = null
            dncnnPadMats[s]?.release(); dncnnPadMats[s] = null
            dncnnInByteBufs[s] = null
            dncnnInFloatBufs[s] = null
            dncnnOutByteBufs[s] = null
            dncnnOutFloatBufs[s] = null
        }
        dncnnAsyncBgrSmall?.release(); dncnnAsyncBgrSmall = null
        dncnnAsyncGrayTmp?.release(); dncnnAsyncGrayTmp = null
        dncnnResidualF?.release(); dncnnResidualF = null
        dncnnResQ4Mat256?.release(); dncnnResQ4Mat256 = null
        dncnnResQ4Up?.release(); dncnnResQ4Up = null
        dncnnYQ4Mat?.release(); dncnnYQ4Mat = null
        dncnnU8Mat?.release(); dncnnU8Mat = null
        dncnnBlurU8Mat?.release(); dncnnBlurU8Mat = null
        dncnnDiffU8Mat?.release(); dncnnDiffU8Mat = null
        dncnnFlatMaskU8Mat?.release(); dncnnFlatMaskU8Mat = null
        dncnnBlendedU8Mat?.release(); dncnnBlendedU8Mat = null

        tileChannelsA?.forEach { it.release() }; tileChannelsA = null
        tileChannelsB?.forEach { it.release() }; tileChannelsB = null
        tileOutPlaneWraps?.forEach { it.release() }; tileOutPlaneWraps = null
        resizedOutChannels?.forEach { it.release() }; resizedOutChannels = null
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
        baseS16Tmp?.release(); baseS16Tmp = null
        blurS16Tmp?.release(); blurS16Tmp = null
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
        morphKernel3x3?.release(); morphKernel3x3 = null
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
