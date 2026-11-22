package com.example.engine

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtProvider
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import android.content.Context
import android.os.Build
import com.example.model.DeviceCapabilities
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.File
import java.nio.FloatBuffer
import java.util.EnumSet
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * Pure geometry helpers for dynamic preset tile division and upscaled tile mapping
 * with guaranteed seam overlap, 1:1 aspect preservation, and zero cropped frame content.
 */
object AiTilingMath {

    data class TileRegion(
        val index: Int,
        val row: Int,
        val col: Int,
        val x: Int,
        val y: Int,
        val width: Int,
        val height: Int,
        val outX: Int = x,
        val outY: Int = y,
        val outWidth: Int = width,
        val outHeight: Int = height,
        val isFirstX: Boolean,
        val isLastX: Boolean,
        val isFirstY: Boolean,
        val isLastY: Boolean
    )

    data class TileGridPlan(
        val cols: Int,
        val rows: Int,
        val totalTiles: Int,
        val overlapX: Int,
        val overlapY: Int,
        val outOverlapX: Int = overlapX,
        val outOverlapY: Int = overlapY,
        val tiles: List<TileRegion>
    )

    /**
     * Dynamically divides a [frameW] x [frameH] frame into exactly [targetTilesPerFrame]
     * overlapping tiles (e.g., 4 -> 2x2, 6 -> 3x2/2x3, 12 -> 4x3/3x4, 25 -> 5x5) covering 100%
     * of the frame with zero cropped pixels.
     */
    fun planDynamicTileGrid(frameW: Int, frameH: Int, targetTilesPerFrame: Int): TileGridPlan {
        val safeW = frameW.coerceAtLeast(2)
        val safeH = frameH.coerceAtLeast(2)
        val target = targetTilesPerFrame.coerceAtLeast(1)

        val (cols, rows) = chooseGridDimensions(safeW, safeH, target)
        val xAxis = divideAxisWithOverlap(safeW, cols)
        val yAxis = divideAxisWithOverlap(safeH, rows)

        val tiles = ArrayList<TileRegion>(cols * rows)
        var idx = 0
        for (r in 0 until rows) {
            val (y0, tileH) = yAxis.segments[r]
            for (c in 0 until cols) {
                val (x0, tileW) = xAxis.segments[c]
                tiles.add(
                    TileRegion(
                        index = idx++,
                        row = r,
                        col = c,
                        x = x0,
                        y = y0,
                        width = tileW,
                        height = tileH,
                        outX = x0,
                        outY = y0,
                        outWidth = tileW,
                        outHeight = tileH,
                        isFirstX = (c == 0),
                        isLastX = (c == cols - 1),
                        isFirstY = (r == 0),
                        isLastY = (r == rows - 1)
                    )
                )
            }
        }

        return TileGridPlan(
            cols = cols,
            rows = rows,
            totalTiles = tiles.size,
            overlapX = xAxis.overlap,
            overlapY = yAxis.overlap,
            outOverlapX = xAxis.overlap,
            outOverlapY = yAxis.overlap,
            tiles = tiles
        )
    }

    /**
     * Plans an optimal aspect-preserving tile grid for input [inW]x[inH] -> output [outW]x[outH]
     * so every tile stays <= [tileSize] where possible (1:1 mirror-padded, zero distortion) and
     * maps cleanly with guaranteed overlap into [outW]x[outH] output space with zero seams.
     */
    fun planUpscaleTileGrid(
        inW: Int,
        inH: Int,
        outW: Int,
        outH: Int,
        tileSize: Int,
        maxTilesCap: Int = 25
    ): TileGridPlan {
        val safeInW = inW.coerceAtLeast(2)
        val safeInH = inH.coerceAtLeast(2)
        val safeOutW = outW.coerceAtLeast(safeInW)
        val safeOutH = outH.coerceAtLeast(safeInH)
        val safeTile = tileSize.coerceIn(128, 512)

        if (safeInW <= safeTile && safeInH <= safeTile) {
            return TileGridPlan(
                cols = 1,
                rows = 1,
                totalTiles = 1,
                overlapX = 0,
                overlapY = 0,
                outOverlapX = 0,
                outOverlapY = 0,
                tiles = listOf(
                    TileRegion(
                        index = 0,
                        row = 0,
                        col = 0,
                        x = 0,
                        y = 0,
                        width = safeInW,
                        height = safeInH,
                        outX = 0,
                        outY = 0,
                        outWidth = safeOutW,
                        outHeight = safeOutH,
                        isFirstX = true,
                        isLastX = true,
                        isFirstY = true,
                        isLastY = true
                    )
                )
            )
        }

        val overlapEstimate = (safeTile * 0.12f).roundToInt().coerceIn(16, 48)
        val step = (safeTile - overlapEstimate).coerceAtLeast(64)
        var cols = if (safeInW <= safeTile) 1 else ceil((safeInW - overlapEstimate).toDouble() / step.toDouble()).toInt().coerceAtLeast(2)
        var rows = if (safeInH <= safeTile) 1 else ceil((safeInH - overlapEstimate).toDouble() / step.toDouble()).toInt().coerceAtLeast(2)

        val safeCap = maxTilesCap.coerceAtLeast(1)
        while (cols * rows > safeCap && (cols > 1 || rows > 1)) {
            if (cols >= rows && cols > 1) cols-- else if (rows > 1) rows--
        }

        val xAxis = divideAxisWithOverlap(safeInW, cols)
        val yAxis = divideAxisWithOverlap(safeInH, rows)

        val scaleX = safeOutW.toDouble() / safeInW.toDouble()
        val scaleY = safeOutH.toDouble() / safeInH.toDouble()

        val tiles = ArrayList<TileRegion>(cols * rows)
        var idx = 0
        for (r in 0 until rows) {
            val (y0, inTileH) = yAxis.segments[r]
            val outY0 = if (r == 0) 0 else floor(y0 * scaleY).toInt().coerceIn(0, safeOutH - 2)
            val outY1 = if (r == rows - 1) safeOutH else ceil((y0 + inTileH) * scaleY).toInt().coerceIn(outY0 + 2, safeOutH)
            val outTileH = outY1 - outY0

            for (c in 0 until cols) {
                val (x0, inTileW) = xAxis.segments[c]
                val outX0 = if (c == 0) 0 else floor(x0 * scaleX).toInt().coerceIn(0, safeOutW - 2)
                val outX1 = if (c == cols - 1) safeOutW else ceil((x0 + inTileW) * scaleX).toInt().coerceIn(outX0 + 2, safeOutW)
                val outTileW = outX1 - outX0

                tiles.add(
                    TileRegion(
                        index = idx++,
                        row = r,
                        col = c,
                        x = x0,
                        y = y0,
                        width = inTileW,
                        height = inTileH,
                        outX = outX0,
                        outY = outY0,
                        outWidth = outTileW,
                        outHeight = outTileH,
                        isFirstX = (c == 0),
                        isLastX = (c == cols - 1),
                        isFirstY = (r == 0),
                        isLastY = (r == rows - 1)
                    )
                )
            }
        }

        return TileGridPlan(
            cols = cols,
            rows = rows,
            totalTiles = tiles.size,
            overlapX = xAxis.overlap,
            overlapY = yAxis.overlap,
            outOverlapX = (xAxis.overlap * scaleX).roundToInt().coerceAtLeast(if (cols > 1) 12 else 0),
            outOverlapY = (yAxis.overlap * scaleY).roundToInt().coerceAtLeast(if (rows > 1) 12 else 0),
            tiles = tiles
        )
    }

    private fun chooseGridDimensions(w: Int, h: Int, targetTiles: Int): Pair<Int, Int> {
        val isPortrait = h > w
        return when (targetTiles) {
            4 -> 2 to 2
            6 -> if (isPortrait) 2 to 3 else 3 to 2
            12 -> if (isPortrait) 3 to 4 else 4 to 3
            25 -> 5 to 5
            else -> {
                val aspect = w.toDouble() / h.toDouble()
                var bestCols = 1
                var bestRows = targetTiles
                var bestDiff = Double.MAX_VALUE
                val limit = sqrt(targetTiles.toDouble()).toInt() + 1
                for (a in 1..limit) {
                    if (targetTiles % a == 0) {
                        val b = targetTiles / a
                        for ((c, r) in listOf(a to b, b to a)) {
                            val diff = abs((c.toDouble() / r.toDouble()) - aspect)
                            if (diff < bestDiff) {
                                bestDiff = diff
                                bestCols = c
                                bestRows = r
                            }
                        }
                    }
                }
                bestCols to bestRows
            }
        }
    }

    private data class AxisDivision(
        val overlap: Int,
        val segments: List<Pair<Int, Int>>
    )

    private fun divideAxisWithOverlap(totalLen: Int, count: Int): AxisDivision {
        if (count <= 1) {
            return AxisDivision(overlap = 0, segments = listOf(0 to totalLen))
        }
        val baseStride = totalLen.toDouble() / count.toDouble()
        val maxSafeOverlap = ((totalLen / count) - 1).coerceAtLeast(1)
        val desiredOverlap = (baseStride * 0.16).roundToInt()
        val overlap = desiredOverlap.coerceIn(min(16, maxSafeOverlap), maxSafeOverlap)

        val tileSpan = ceil((totalLen + (count - 1) * overlap) / count.toDouble())
            .toInt()
            .coerceIn(1, totalLen)

        val spanStep = if (count > 1) {
            (totalLen - tileSpan).toDouble() / (count - 1).toDouble()
        } else {
            0.0
        }

        val segments = (0 until count).map { i ->
            val start = if (i == count - 1) {
                (totalLen - tileSpan).coerceAtLeast(0)
            } else {
                (i * spanStep).roundToInt().coerceIn(0, (totalLen - tileSpan).coerceAtLeast(0))
            }
            start to tileSpan
        }
        return AxisDivision(overlap = overlap, segments = segments)
    }

    fun featherWeight(
        pos: Int,
        len: Int,
        isFirst: Boolean,
        isLast: Boolean,
        feather: Int
    ): Float {
        if (feather <= 0) return 1f
        var w = 1f
        if (!isFirst && pos < feather) {
            w = min(w, (pos + 1).toFloat() / feather.toFloat())
        }
        if (!isLast && pos >= len - feather) {
            w = min(w, (len - pos).toFloat() / feather.toFloat())
        }
        return w.coerceIn(0.05f, 1f)
    }
}

/**
 * Safeguard policy that prioritizes real ONNX AI execution (stepping down tile sizes if needed)
 * and only triggers non-neural fallback under catastrophic multi-second-per-tile hangs.
 */
object AiSpeedPolicy {
    const val MAX_ACCEPTABLE_AI_FRAME_MS = 12_000L
    const val MAX_ACCEPTABLE_SINGLE_TILE_MS_HEAVY = 3_000L

    fun shouldEngageLightweightFallback(
        measuredTileMs: Long,
        tilesPerFrame: Int,
        estimatedFrameCount: Int
    ): Boolean {
        if (tilesPerFrame <= 0 || measuredTileMs <= 0L) return false
        if (measuredTileMs > MAX_ACCEPTABLE_SINGLE_TILE_MS_HEAVY) return true
        val projectedPerFrameMs = measuredTileMs * tilesPerFrame.toLong()
        return estimatedFrameCount >= 180 && projectedPerFrameMs > MAX_ACCEPTABLE_AI_FRAME_MS
    }
}

data class TileProcessOutcome(
    val outputBgr: Mat,
    val usedLightweightFallback: Boolean,
    val fallbackReason: String? = null,
    val activeAccelerationLabel: String,
    val faceAiMs: Long = 0L
)

class OnnxAiInferenceEngine(private val context: Context) {

    private val env: OrtEnvironment by lazy { OrtEnvironment.getEnvironment() }
    val gpuProcessor: GlesGpuImageProcessor = GlesGpuImageProcessor()

    private var cpuFallbackSession: OrtSession? = null
    private var activeSrSession: OrtSession? = null
    private var dncnnSession: OrtSession? = null

    private var activeProviderDescription: String = "CPU XNNPACK"
    private var benchmarkDetails: String = "Not benchmarked"
    private var lastInitDurationMs: Long = 0L
    private var nnapiDisabledDueToRuntimeError = false
    private var glesGpuAvailable = false

    // Reusable Real-ESRGAN ONNX tile buffers
    private var allocTileSize: Int = 0
    private var tileInArr: FloatArray = FloatArray(0)
    private var tileInBuf: FloatBuffer = FloatBuffer.wrap(tileInArr)
    private var tileInPlaneTmp: FloatArray = FloatArray(0)
    private var tileOutPlaneTmp: FloatArray = FloatArray(0)

    // Reusable DnCNN ONNX [1, 1, 256, 256] buffers
    private val dncnnTileSize = 256
    private val dncnnTilePixels = dncnnTileSize * dncnnTileSize
    private var dncnnInArr: FloatArray? = null
    private var dncnnInBuf: FloatBuffer? = null
    private var dncnnOutArr: FloatArray? = null
    private var dncnnPadMat: Mat? = null
    private var dncnnFloatMat: Mat? = null
    private var dncnnOutTileMat: Mat? = null
    private var dncnnAccumMat: Mat? = null
    private var dncnnWeightMat: Mat? = null
    private var dncnnWeightedMat: Mat? = null
    private var dncnnU8Mat: Mat? = null

    // Reusable OpenCV native C++ scratch Mats (zero per-tile/per-frame allocations)
    private var srcRgbMat: Mat? = null
    private var baseRgb8u: Mat? = null
    private var tilePaddedMat: Mat? = null
    private var tileFloatHwc: Mat? = null
    private var tileChannels: List<Mat>? = null
    private var outFullPlaneMat: Mat? = null
    private var resizedOutChannels: List<Mat>? = null
    private var outResizedFloat: Mat? = null
    private var weightedTileFloat: Mat? = null
    private var accumRgbMat: Mat? = null
    private var accumWeightMat: Mat? = null
    private var accumChSplit: MutableList<Mat>? = null
    private var weightClampFloorMat: Mat? = null
    private var outRgb8u: Mat? = null

    // Face ROI super-resolution scratch buffers
    private var faceSrFloatMat: Mat? = null
    private var faceSr8uMat: Mat? = null
    private var faceMaskFloatMat: Mat? = null

    // Lightweight OpenCV SIMD fallback scratch Mats
    private var lwBlurMat: Mat? = null
    private var lwDetailMat: Mat? = null

    // Feather weight Mat cache keyed by tile dimensions & overlap
    private val featherMaskCache = HashMap<Long, Mat>()

    val isRealEsrganReady: Boolean get() = activeSrSession != null
    val isDncnnReady: Boolean get() = dncnnSession != null
    val isGlesGpuReady: Boolean get() = glesGpuAvailable
    val providerLabel: String get() = activeProviderDescription
    val benchmarkSummary: String get() = benchmarkDetails
    val initDurationMs: Long get() = lastInitDurationMs

    @Synchronized
    fun initialize(
        capabilities: DeviceCapabilities,
        logger: EngineDiagnosticsLogger? = null
    ) {
        if (activeSrSession != null && dncnnSession != null) {
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
        // Copy external weights beside dncnn.onnx so ORT resolves dncnn.data cleanly
        val dncnnDataFile = ensureAssetCopied(
            "ai_models/denoise/dncnn.data",
            File(modelDir, "denoise/dncnn.data")
        )

        glesGpuAvailable = gpuProcessor.ensureInitialized(logger)

        val availableProviders = try {
            OrtEnvironment.getAvailableProviders()
        } catch (_: Throwable) {
            EnumSet.of(OrtProvider.CPU)
        }
        logger?.info("Available ORT providers: ${availableProviders.joinToString()} • GLES31=$glesGpuAvailable")

        val notes = mutableListOf<String>()
        var bestThreads = capabilities.inferenceThreads.coerceIn(2, 6)

        if (activeSrSession == null && srFile.exists() && srFile.length() > 0L) {
            val candidates = mutableListOf<Triple<String, OrtSession, Long>>()
            val threadCandidates = DeviceCapabilityDetector.benchmarkThreadCandidates(
                cpuCores = capabilities.cpuCores,
                maxTierThreads = 8
            )
            var bestCpuPair: Pair<String, OrtSession>? = null
            var bestCpuMs = Long.MAX_VALUE

            for (threads in threadCandidates) {
                var session: OrtSession? = null
                val label = "CPU XNNPACK (${threads}T)"
                try {
                    session = createXnnpackSession(srFile.absolutePath, threads)
                    val ms = benchmarkSrSession(session, tile = 128, timeoutSec = 6L, logger = logger)
                    if (ms != null) {
                        notes.add("XNN-${threads}T=${ms}ms")
                        candidates.add(Triple(label, session, ms))
                        if (ms < bestCpuMs) {
                            bestCpuMs = ms
                            bestCpuPair = label to session
                            bestThreads = threads
                        }
                    } else {
                        notes.add("XNN-${threads}T=FAIL")
                        session.close()
                    }
                } catch (t: Throwable) {
                    notes.add("XNN-${threads}T=ERR")
                    logger?.warn("$label init failed (${t.message})")
                    try {
                        session?.close()
                    } catch (_: Throwable) {
                    }
                }
            }

            if (bestCpuPair == null) {
                try {
                    val std = createPlainCpuSession(srFile.absolutePath, capabilities.inferenceThreads.coerceIn(2, 8))
                    val ms = benchmarkSrSession(std, tile = 128, timeoutSec = 6L, logger = logger) ?: 500L
                    val label = "CPU ORT (${capabilities.inferenceThreads}T)"
                    notes.add("CPU=${ms}ms")
                    candidates.add(Triple(label, std, ms))
                    bestCpuPair = label to std
                } catch (t: Throwable) {
                    logger?.error("CPU ONNX session creation failed", t)
                }
            }

            cpuFallbackSession = bestCpuPair?.second

            val nnapiAllowed = !nnapiDisabledDueToRuntimeError &&
                capabilities.nnapiSupported &&
                Build.VERSION.SDK_INT >= Build.VERSION_CODES.P &&
                availableProviders.contains(OrtProvider.NNAPI)

            if (nnapiAllowed) {
                val nnapiAttempts = listOf(
                    "Vulkan/NNAPI GPU FP16" to EnumSet.of(NNAPIFlags.USE_FP16, NNAPIFlags.USE_NCHW, NNAPIFlags.CPU_DISABLED),
                    "NNAPI Hardware FP16" to EnumSet.of(NNAPIFlags.USE_FP16),
                    "NNAPI Standard" to EnumSet.noneOf(NNAPIFlags::class.java)
                )
                for ((nnLabel, flags) in nnapiAttempts) {
                    var nnSession: OrtSession? = null
                    try {
                        nnSession = createNnapiSession(srFile.absolutePath, capabilities, flags)
                        val ms = benchmarkSrSession(nnSession, tile = 128, timeoutSec = 5L, logger = logger)
                        if (ms != null) {
                            notes.add("$nnLabel=${ms}ms")
                            candidates.add(Triple(nnLabel, nnSession, ms))
                            break
                        } else {
                            nnSession.close()
                        }
                    } catch (_: Throwable) {
                        try {
                            nnSession?.close()
                        } catch (_: Throwable) {
                        }
                    }
                }
            }

            val winner = candidates.minByOrNull { it.third }
            if (winner != null) {
                activeSrSession = winner.second
                activeProviderDescription = "${winner.first} (${winner.third}ms/128p)"
                for ((_, sess, _) in candidates) {
                    if (sess !== winner.second && sess !== cpuFallbackSession) {
                        try {
                            sess.close()
                        } catch (_: Throwable) {
                        }
                    }
                }
            } else {
                activeSrSession = cpuFallbackSession
                activeProviderDescription = "CPU Fallback"
            }
        } else if (!srFile.exists()) {
            logger?.error("Missing bundled Real-ESRGAN ONNX asset")
        }

        // Initialize DnCNN grayscale neural denoiser (dncnn.onnx + dncnn.data)
        if (dncnnSession == null && dncnnFile.exists() && dncnnFile.length() > 0L && dncnnDataFile.exists()) {
            try {
                val sess = try {
                    createXnnpackSession(dncnnFile.absolutePath, bestThreads)
                } catch (_: Throwable) {
                    createPlainCpuSession(dncnnFile.absolutePath, bestThreads)
                }
                val dnMs = benchmarkDnCnnSession(sess, timeoutSec = 5L, logger = logger)
                if (dnMs != null) {
                    dncnnSession = sess
                    notes.add("DnCNN=${dnMs}ms")
                } else {
                    sess.close()
                    notes.add("DnCNN=FAIL")
                }
            } catch (t: Throwable) {
                logger?.warn("DnCNN session initialization failed: ${t.message}")
                notes.add("DnCNN=ERR")
            }
        }

        benchmarkDetails = notes.joinToString(" | ").ifBlank { "No benchmark" }
        lastInitDurationMs = System.currentTimeMillis() - t0
        logger?.info(
            "AI Init complete (${lastInitDurationMs}ms): SR=${isRealEsrganReady}, DnCNN=${isDncnnReady}, " +
                "Provider=$activeProviderDescription • Benchmarks=[$benchmarkDetails]"
        )
    }

    private fun createXnnpackSession(modelPath: String, threads: Int): OrtSession {
        val opts = OrtSession.SessionOptions().apply {
            setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setInterOpNumThreads(1)
            setIntraOpNumThreads(1)
            addConfigEntry("session.intra_op.allow_spinning", "0")
            addXnnpack(mapOf("intra_op_num_threads" to threads.toString()))
        }
        return env.createSession(modelPath, opts)
    }

    private fun createPlainCpuSession(modelPath: String, threads: Int): OrtSession {
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(threads.coerceIn(1, 8))
            setInterOpNumThreads(1)
            setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        return env.createSession(modelPath, opts)
    }

    private fun createNnapiSession(
        modelPath: String,
        capabilities: DeviceCapabilities,
        flags: EnumSet<NNAPIFlags>
    ): OrtSession {
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(capabilities.inferenceThreads.coerceIn(2, 6))
            setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            if (flags.isEmpty()) addNnapi() else addNnapi(flags)
        }
        return env.createSession(modelPath, opts)
    }

    private fun benchmarkSrSession(
        session: OrtSession,
        tile: Int,
        timeoutSec: Long,
        logger: EngineDiagnosticsLogger?
    ): Long? {
        val executor = Executors.newSingleThreadExecutor()
        return try {
            val future = executor.submit<Long?> {
                val inputName = session.inputNames.first()
                val buf = FloatBuffer.allocate(3 * tile * tile)
                for (i in 0 until buf.capacity()) {
                    buf.put(i, 0.5f)
                }
                buf.rewind()
                val shape = longArrayOf(1L, 3L, tile.toLong(), tile.toLong())

                OnnxTensor.createTensor(env, buf, shape).use { tWarm ->
                    session.run(mapOf(inputName to tWarm)).use { res ->
                        val out = res[0] as OnnxTensor
                        val outBuf = out.floatBuffer
                        if (outBuf.remaining() < 3 * tile * tile * 16) return@submit null
                        val probe = outBuf.get(0)
                        if (probe.isNaN() || probe.isInfinite()) return@submit null
                    }
                }

                buf.rewind()
                val t0 = System.nanoTime()
                OnnxTensor.createTensor(env, buf, shape).use { tRun ->
                    session.run(mapOf(inputName to tRun)).use { res ->
                        val out = res[0] as OnnxTensor
                        val outBuf = out.floatBuffer
                        val probe = outBuf.get(outBuf.remaining() / 2)
                        if (probe.isNaN() || probe.isInfinite()) return@submit null
                    }
                }
                ((System.nanoTime() - t0) / 1_000_000L).coerceAtLeast(1L)
            }
            future.get(timeoutSec, TimeUnit.SECONDS)
        } catch (t: Throwable) {
            logger?.warn("SR session benchmark failed/timed out: ${t.message}")
            null
        } finally {
            executor.shutdownNow()
        }
    }

    private fun benchmarkDnCnnSession(
        session: OrtSession,
        timeoutSec: Long,
        logger: EngineDiagnosticsLogger?
    ): Long? {
        val executor = Executors.newSingleThreadExecutor()
        return try {
            val future = executor.submit<Long?> {
                val inputName = session.inputNames.first()
                val buf = FloatBuffer.allocate(dncnnTilePixels)
                for (i in 0 until dncnnTilePixels) {
                    buf.put(i, 0.5f)
                }
                buf.rewind()
                val shape = longArrayOf(1L, 1L, dncnnTileSize.toLong(), dncnnTileSize.toLong())
                val t0 = System.nanoTime()
                OnnxTensor.createTensor(env, buf, shape).use { tensor ->
                    session.run(mapOf(inputName to tensor)).use { res ->
                        val out = res[0] as OnnxTensor
                        val outBuf = out.floatBuffer
                        if (outBuf.remaining() < dncnnTilePixels) return@submit null
                        val probe = outBuf.get(dncnnTilePixels / 2)
                        if (probe.isNaN() || probe.isInfinite()) return@submit null
                    }
                }
                ((System.nanoTime() - t0) / 1_000_000L).coerceAtLeast(1L)
            }
            future.get(timeoutSec, TimeUnit.SECONDS)
        } catch (t: Throwable) {
            logger?.warn("DnCNN session benchmark failed: ${t.message}")
            null
        } finally {
            executor.shutdownNow()
        }
    }

    /**
     * Runs genuine DnCNN neural denoising ([1, 1, 256, 256] luminance Y tensor) over [yChan]
     * with BORDER_REFLECT_101 mirror-padding and feathered tile stitching, preserving sharp
     * edges and textures while stripping film grain, sensor noise, and macroblock ringing.
     */
    fun denoiseLuminanceYWithDnCnn(
        yChan: Mat,
        strength: Float,
        logger: EngineDiagnosticsLogger? = null
    ): Boolean {
        val session = dncnnSession ?: return false
        val blend = strength.toDouble().coerceIn(0.05, 0.95)
        val w = yChan.cols()
        val h = yChan.rows()
        if (w < 8 || h < 8) return false

        ensureDnCnnBuffers(w, h)
        val inArr = dncnnInArr!!
        val inBuf = dncnnInBuf!!
        val outArr = dncnnOutArr!!
        val padMat = dncnnPadMat!!
        val floatMat = dncnnFloatMat!!
        val outTileMat = dncnnOutTileMat!!
        val accum = dncnnAccumMat!!
        val weightAccum = dncnnWeightMat!!
        val weighted = dncnnWeightedMat!!
        val u8Out = dncnnU8Mat!!

        val shape = longArrayOf(1L, 1L, dncnnTileSize.toLong(), dncnnTileSize.toLong())
        val inputName = session.inputNames.first()

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
                    val sub = Mat(yChan, cropRect)

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
                    sub.release()

                    floatMat.get(0, 0, inArr)
                    inBuf.rewind()

                    OnnxTensor.createTensor(env, inBuf, shape).use { tensor ->
                        session.run(mapOf(inputName to tensor)).use { results ->
                            val outTensor = results[0] as OnnxTensor
                            val outBuf = outTensor.floatBuffer
                            outBuf.position(0)
                            outBuf.get(outArr, 0, dncnnTilePixels)
                        }
                    }

                    outTileMat.put(0, 0, outArr)
                    val validOut = if (cropW == dncnnTileSize && cropH == dncnnTileSize) {
                        outTileMat
                    } else {
                        Mat(outTileMat, Rect(0, 0, cropW, cropH))
                    }

                    val featherTile = AiTilingMath.TileRegion(
                        index = r * cols + c,
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
            Core.min(accum, Scalar(1.0), accum)
            Core.max(accum, Scalar(0.0), accum)
            accum.convertTo(u8Out, CvType.CV_8UC1, 255.0)

            Core.addWeighted(u8Out, blend, yChan, 1.0 - blend, 0.0, yChan)
            true
        } catch (t: Throwable) {
            logger?.warn("DnCNN inference fallback to temporal cleanup (${t.message})")
            false
        }
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
     * Genuine AI Super-Resolution + Dedicated Face ROI AI Refinement:
     * - Preserves 1:1 pixel aspect ratio via BORDER_REFLECT_101 mirror-padding (never squishes tiles)
     * - Uses Real-ESRGAN x4v3 neural reconstruction directly to [outW] x [outH]
     * - Supersamples down via INTER_AREA for 1x/2x/3x and outputs true 4x detail for 4x
     * - Enhances detected faces separately with dedicated face ROI SR + soft elliptical feather mask
     * - Uses memory-lean CV_8UC3 base + single-channel weight division to keep 4K RAM clean
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
        isCancelled: () -> Boolean,
        logger: EngineDiagnosticsLogger? = null,
        onTileProgress: (done: Int, total: Int, tileSize: Int) -> Unit = { _, _, _ -> }
    ): TileProcessOutcome {
        val inW = srcBgr.cols()
        val inH = srcBgr.rows()
        val safeOutW = outW.coerceAtLeast(inW)
        val safeOutH = outH.coerceAtLeast(inH)

        if (forceLightweightFallback || activeSrSession == null) {
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

        // Pre-compute clean bicubic base for blending (only ~25MB at 4K)
        if (inW == safeOutW && inH == safeOutH) {
            localRgb.copyTo(baseRgb8u!!)
        } else {
            Imgproc.resize(localRgb, baseRgb8u!!, Size(safeOutW.toDouble(), safeOutH.toDouble()), 0.0, 0.0, Imgproc.INTER_CUBIC)
        }

        accumRgbMat!!.setTo(Scalar(0.0, 0.0, 0.0))
        accumWeightMat!!.setTo(Scalar(0.0))

        val fixedTilePixels = safeTileSize * safeTileSize
        val fixedShape = longArrayOf(1L, 3L, safeTileSize.toLong(), safeTileSize.toLong())
        val chIn = tileChannels!!
        val paddedTile = tilePaddedMat!!
        val floatTileHwc = tileFloatHwc!!
        val outResized = outResizedFloat!!
        val weightedTile = weightedTileFloat!!

        var tileTimingSumMs = 0L
        var timedTiles = 0

        for (tile in grid.tiles) {
            if (isCancelled()) throw InterruptedException("Cancelled during AI tile enhancement")
            val tTile0 = System.currentTimeMillis()

            // 1. Aspect-Preserving Input Tile Preparation
            val inRect = Rect(tile.x, tile.y, tile.width, tile.height)
            val roi = Mat(localRgb, inRect)
            val validSrW: Int
            val validSrH: Int

            if (tile.width <= safeTileSize && tile.height <= safeTileSize) {
                if (tile.width == safeTileSize && tile.height == safeTileSize) {
                    roi.copyTo(paddedTile)
                } else {
                    Core.copyMakeBorder(
                        roi,
                        paddedTile,
                        0,
                        safeTileSize - tile.height,
                        0,
                        safeTileSize - tile.width,
                        Core.BORDER_REFLECT_101
                    )
                }
                validSrW = tile.width * 4
                validSrH = tile.height * 4
            } else {
                // Only reached when a single axis exceeds safeTileSize under a hard tile cap:
                // still resize cleanly with INTER_AREA so high-res detail maps properly.
                Imgproc.resize(roi, paddedTile, Size(safeTileSize.toDouble(), safeTileSize.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
                validSrW = safeTileSize * 4
                validSrH = safeTileSize * 4
            }
            roi.release()

            paddedTile.convertTo(floatTileHwc, CvType.CV_32FC3, 1.0 / 255.0)
            Core.split(floatTileHwc, chIn)
            chIn[0].get(0, 0, tileInPlaneTmp)
            System.arraycopy(tileInPlaneTmp, 0, tileInArr, 0, fixedTilePixels)
            chIn[1].get(0, 0, tileInPlaneTmp)
            System.arraycopy(tileInPlaneTmp, 0, tileInArr, fixedTilePixels, fixedTilePixels)
            chIn[2].get(0, 0, tileInPlaneTmp)
            System.arraycopy(tileInPlaneTmp, 0, tileInArr, 2 * fixedTilePixels, fixedTilePixels)
            tileInBuf.rewind()

            // 2. Real-ESRGAN x4v3 Neural Inference + Valid Region Crop + Area/Cubic Scaling to outWidth x outHeight
            val dstSize = Size(tile.outWidth.toDouble(), tile.outHeight.toDouble())
            runSrTileAndResizeDirect(
                shape = fixedShape,
                tileSize = safeTileSize,
                validSrW = validSrW,
                validSrH = validSrH,
                dstSize = dstSize,
                outResizedHwc = outResized,
                logger = logger
            )

            // 3. Cached Overlap Feather Accumulation in Output Coordinate Space
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

            val tileElapsed = (System.currentTimeMillis() - tTile0).coerceAtLeast(1L)
            tileTimingSumMs += tileElapsed
            timedTiles++
            onTileProgress(tile.index + 1, grid.totalTiles, safeTileSize)

            // Only abandon ONNX if hardware/driver is catastrophically hanging (>3000ms per tile)
            if (timedTiles == 2 && AiSpeedPolicy.shouldEngageLightweightFallback(
                    measuredTileMs = tileTimingSumMs / timedTiles,
                    tilesPerFrame = grid.totalTiles,
                    estimatedFrameCount = estimatedTotalFrames
                )
            ) {
                val avgTileMs = tileTimingSumMs / timedTiles
                val reason = "Emergency GPU/SIMD Fallback: ${safeTileSize}px neural tile stalled at ${avgTileMs}ms/tile"
                logger?.warn(reason)
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
                    faceAiMs = 0L
                )
            }
        }

        // 4. Low-RAM Per-Channel Normalization + Progressive AI Blend with Bicubic Base
        Core.max(accumWeightMat!!, weightClampFloorMat!!, accumWeightMat!!)
        val splitCh = accumChSplit!!
        Core.split(accumRgbMat!!, splitCh)
        for (c in 0 until 3) {
            Core.divide(splitCh[c], accumWeightMat!!, splitCh[c])
            Core.min(splitCh[c], Scalar(1.0), splitCh[c])
            Core.max(splitCh[c], Scalar(0.0), splitCh[c])
        }
        Core.merge(splitCh, accumRgbMat!!)
        accumRgbMat!!.convertTo(outRgb8u!!, CvType.CV_8UC3, 255.0)

        val aiBlend = aiStrength.toDouble().coerceIn(0.20, 0.96)
        Core.addWeighted(outRgb8u!!, aiBlend, baseRgb8u!!, 1.0 - aiBlend, 0.0, outRgb8u!!)

        // 5. Dedicated Face ROI Neural Super-Resolution Refinement (eyes, lips, beard, hair, skin)
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
            faceAiMs = faceMs
        )
    }

    /**
     * Runs dedicated full-fidelity Real-ESRGAN x4v3 inference over each detected face ROI
     * (expanded symmetrically for hair, forehead, ears & beard) and blends softly with
     * an elliptical raised-cosine mask so facial features gain crisp clarity without seams.
     */
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
        val chIn = tileChannels!!
        val paddedTile = tilePaddedMat!!
        val floatTileHwc = tileFloatHwc!!
        val fixedTilePixels = safeTileSize * safeTileSize
        val scaleX = outW.toDouble() / inW.toDouble()
        val scaleY = outH.toDouble() / inH.toDouble()
        val faceSrF = faceSrFloatMat ?: Mat().also { faceSrFloatMat = it }
        val faceSrU8 = faceSr8uMat ?: Mat().also { faceSr8uMat = it }
        val maskF = faceMaskFloatMat ?: Mat().also { faceMaskFloatMat = it }

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
                    roi,
                    paddedTile,
                    0,
                    safeTileSize - fh,
                    0,
                    safeTileSize - fw,
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
            chIn[0].get(0, 0, tileInPlaneTmp)
            System.arraycopy(tileInPlaneTmp, 0, tileInArr, 0, fixedTilePixels)
            chIn[1].get(0, 0, tileInPlaneTmp)
            System.arraycopy(tileInPlaneTmp, 0, tileInArr, fixedTilePixels, fixedTilePixels)
            chIn[2].get(0, 0, tileInPlaneTmp)
            System.arraycopy(tileInPlaneTmp, 0, tileInArr, 2 * fixedTilePixels, fixedTilePixels)
            tileInBuf.rewind()

            val outX0 = floor(fx0 * scaleX).toInt().coerceIn(0, outW - 2)
            val outY0 = floor(fy0 * scaleY).toInt().coerceIn(0, outH - 2)
            val outX1 = ceil(fx1 * scaleX).toInt().coerceIn(outX0 + 2, outW)
            val outY1 = ceil(fy1 * scaleY).toInt().coerceIn(outY0 + 2, outH)
            val outFw = outX1 - outX0
            val outFh = outY1 - outY0

            runSrTileAndResizeDirect(
                shape = fixedShape,
                tileSize = safeTileSize,
                validSrW = validSrW,
                validSrH = validSrH,
                dstSize = Size(outFw.toDouble(), outFh.toDouble()),
                outResizedHwc = faceSrF,
                logger = logger
            )

            faceSrF.convertTo(faceSrU8, CvType.CV_8UC3, 255.0)

            // Elliptical cosine feather blend so face clarity merges seamlessly into neck/hair/bg
            val weightData = FloatArray(outFw * outFh)
            val halfW = outFw * 0.5f
            val halfH = outFh * 0.5f
            val maxWeight = strength.coerceIn(0.25f, 0.92f)
            for (y in 0 until outFh) {
                val dy = (y - halfH) / halfH.coerceAtLeast(1f)
                val rowOff = y * outFw
                for (x in 0 until outFw) {
                    val dx = (x - halfW) / halfW.coerceAtLeast(1f)
                    val radial = sqrt(dx * dx + dy * dy).coerceIn(0f, 1f)
                    val w = if (radial < 0.45f) {
                        maxWeight
                    } else {
                        val t = (radial - 0.45f) / 0.55f
                        maxWeight * 0.5f * (1f + cos(PI.toFloat() * t))
                    }
                    weightData[rowOff + x] = w
                }
            }

            maskF.create(outFh, outFw, CvType.CV_32FC1)
            maskF.put(0, 0, weightData)
            val mask3 = Mat(outFh, outFw, CvType.CV_32FC3)
            val invMask3 = Mat(outFh, outFw, CvType.CV_32FC3)
            val dstSub = Mat(targetRgb8u, Rect(outX0, outY0, outFw, outFh))
            val dstSubF = Mat()
            val srWeightedF = Mat()
            val dstWeightedF = Mat()
            try {
                Core.merge(listOf(maskF, maskF, maskF), mask3)
                Core.subtract(Mat.ones(outFh, outFw, CvType.CV_32FC3), mask3, invMask3)
                dstSub.convertTo(dstSubF, CvType.CV_32FC3)
                faceSrF.convertTo(srWeightedF, CvType.CV_32FC3, 255.0)
                Core.multiply(srWeightedF, mask3, srWeightedF)
                Core.multiply(dstSubF, invMask3, dstWeightedF)
                Core.add(srWeightedF, dstWeightedF, dstSubF)
                dstSubF.convertTo(dstSub, CvType.CV_8UC3)
            } finally {
                mask3.release()
                invMask3.release()
                dstSub.release()
                dstSubF.release()
                srWeightedF.release()
                dstWeightedF.release()
            }
        }
    }

    /**
     * Hardware-first lightweight enhancement fallback:
     * Uses GLES 3.1 Compute Shader on GPU if supported, else native OpenCV C++ SIMD.
     */
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
        shape: LongArray,
        tileSize: Int,
        validSrW: Int,
        validSrH: Int,
        dstSize: Size,
        outResizedHwc: Mat,
        logger: EngineDiagnosticsLogger?
    ) {
        val current = activeSrSession ?: throw IllegalStateException("Real-ESRGAN ONNX model is not loaded.")
        val inputName = current.inputNames.first()
        var tensor = OnnxTensor.createTensor(env, tileInBuf, shape)
        try {
            current.run(mapOf(inputName to tensor)).use { results ->
                val outTensor = results[0] as OnnxTensor
                extractValidPlanesAndResize(outTensor.floatBuffer, tileSize, validSrW, validSrH, dstSize, outResizedHwc)
            }
        } catch (t: Throwable) {
            tensor.close()
            val cpu = cpuFallbackSession
            if (current !== cpu && cpu != null) {
                logger?.warn("NNAPI SR runtime error (${t.message}); switching to CPU XNNPACK")
                nnapiDisabledDueToRuntimeError = true
                activeSrSession = cpu
                activeProviderDescription = "CPU XNNPACK (Runtime fallback)"
                tileInBuf.rewind()
                tensor = OnnxTensor.createTensor(env, tileInBuf, shape)
                try {
                    cpu.run(mapOf(cpu.inputNames.first() to tensor)).use { results ->
                        val outTensor = results[0] as OnnxTensor
                        extractValidPlanesAndResize(outTensor.floatBuffer, tileSize, validSrW, validSrH, dstSize, outResizedHwc)
                    }
                } finally {
                    tensor.close()
                }
                return
            }
            throw t
        }
        tensor.close()
    }

    /**
     * Streams NCHW 4x output planes directly from ORT buffer, slices the valid aspect-preserved
     * region [validSrW x validSrH] (discarding mirror padding), and resizes cleanly via
     * INTER_AREA (downscale supersampling) or INTER_CUBIC (upscale) to [dstSize].
     */
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
        if (dncnnInArr == null) {
            val arr = FloatArray(dncnnTilePixels)
            dncnnInArr = arr
            dncnnInBuf = FloatBuffer.wrap(arr)
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
            tileInArr = FloatArray(3 * px)
            tileInBuf = FloatBuffer.wrap(tileInArr)
            tileInPlaneTmp = FloatArray(px)

            val outDim = tileSize * 4
            val outPx = outDim * outDim
            tileOutPlaneTmp = FloatArray(outPx)

            tileChannels?.forEach { it.release() }
            resizedOutChannels?.forEach { it.release() }
            tileChannels = List(3) { Mat(tileSize, tileSize, CvType.CV_32FC1) }
            resizedOutChannels = List(3) { Mat() }
            outFullPlaneMat?.release()
            outFullPlaneMat = Mat(outDim, outDim, CvType.CV_32FC1)
            tilePaddedMat?.release()
            tilePaddedMat = Mat(tileSize, tileSize, CvType.CV_8UC3)
            tileFloatHwc?.release()
            tileFloatHwc = Mat(tileSize, tileSize, CvType.CV_32FC3)
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
            accumChSplit?.forEach { it.release() }
            weightClampFloorMat?.release()
            outRgb8u?.release()
            clearFeatherCache()

            baseRgb8u = Mat(outH, outW, CvType.CV_8UC3)
            accumRgbMat = Mat(outH, outW, CvType.CV_32FC3)
            accumWeightMat = Mat(outH, outW, CvType.CV_32FC1)
            accumChSplit = ArrayList<Mat>(3).apply {
                repeat(3) { add(Mat(outH, outW, CvType.CV_32FC1)) }
            }
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
        allocTileSize = 0
        tileInArr = FloatArray(0)
        tileInBuf = FloatBuffer.wrap(tileInArr)
        tileInPlaneTmp = FloatArray(0)
        tileOutPlaneTmp = FloatArray(0)

        dncnnInArr = null
        dncnnInBuf = null
        dncnnOutArr = null
        dncnnPadMat?.release(); dncnnPadMat = null
        dncnnFloatMat?.release(); dncnnFloatMat = null
        dncnnOutTileMat?.release(); dncnnOutTileMat = null
        dncnnAccumMat?.release(); dncnnAccumMat = null
        dncnnWeightMat?.release(); dncnnWeightMat = null
        dncnnWeightedMat?.release(); dncnnWeightedMat = null
        dncnnU8Mat?.release(); dncnnU8Mat = null

        tileChannels?.forEach { it.release() }
        tileChannels = null
        resizedOutChannels?.forEach { it.release() }
        resizedOutChannels = null
        outFullPlaneMat?.release(); outFullPlaneMat = null
        tilePaddedMat?.release(); tilePaddedMat = null
        tileFloatHwc?.release(); tileFloatHwc = null
        outResizedFloat?.release(); outResizedFloat = null
        weightedTileFloat?.release(); weightedTileFloat = null

        srcRgbMat?.release(); srcRgbMat = null
        baseRgb8u?.release(); baseRgb8u = null
        accumRgbMat?.release(); accumRgbMat = null
        accumWeightMat?.release(); accumWeightMat = null
        accumChSplit?.forEach { it.release() }
        accumChSplit = null
        weightClampFloorMat?.release(); weightClampFloorMat = null
        outRgb8u?.release(); outRgb8u = null

        faceSrFloatMat?.release(); faceSrFloatMat = null
        faceSr8uMat?.release(); faceSr8uMat = null
        faceMaskFloatMat?.release(); faceMaskFloatMat = null

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

    fun close() {
        releaseWorkBuffers()
        gpuProcessor.release()
        try {
            activeSrSession?.takeIf { it !== cpuFallbackSession }?.close()
            cpuFallbackSession?.close()
            dncnnSession?.close()
        } catch (_: Exception) {
        }
        cpuFallbackSession = null
        activeSrSession = null
        dncnnSession = null
    }
}
