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
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * Pure geometry helpers for dynamic preset tile division (6 / 12 / 25 tiles per frame)
 * with guaranteed overlap and zero cropped or discarded frame content.
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
        val tiles: List<TileRegion>
    )

    /**
     * Dynamically divides a [frameW] x [frameH] frame into exactly [targetTilesPerFrame]
     * overlapping tiles (e.g., 6 -> 3x2/2x3, 12 -> 4x3/3x4, 25 -> 5x5) covering 100%
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
            tiles = tiles
        )
    }

    private fun chooseGridDimensions(w: Int, h: Int, targetTiles: Int): Pair<Int, Int> {
        val isPortrait = h > w
        return when (targetTiles) {
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
        val overlap = desiredOverlap.coerceIn(min(8, maxSafeOverlap), maxSafeOverlap)

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
 * Pure latency threshold policy determining when AI tiled neural inference is too slow
 * for responsive offline processing, triggering the GPU/SIMD lightweight enhancer.
 */
object AiSpeedPolicy {
    const val MAX_ACCEPTABLE_AI_FRAME_MS = 1_400L
    const val MAX_ACCEPTABLE_SINGLE_TILE_MS_HEAVY = 220L

    fun shouldEngageLightweightFallback(
        measuredTileMs: Long,
        tilesPerFrame: Int,
        estimatedFrameCount: Int
    ): Boolean {
        if (tilesPerFrame <= 0 || measuredTileMs <= 0L) return false
        val projectedPerFrameMs = measuredTileMs * tilesPerFrame.toLong()
        if (projectedPerFrameMs > MAX_ACCEPTABLE_AI_FRAME_MS) return true
        // For multi-second high-frame videos with >=12 tiles, enforce tighter responsiveness ceiling
        if (estimatedFrameCount >= 90 && tilesPerFrame >= 12 && measuredTileMs > MAX_ACCEPTABLE_SINGLE_TILE_MS_HEAVY) {
            return true
        }
        return false
    }
}

data class TileProcessOutcome(
    val outputBgr: Mat,
    val usedLightweightFallback: Boolean,
    val fallbackReason: String? = null,
    val activeAccelerationLabel: String
)

class OnnxAiInferenceEngine(private val context: Context) {

    private val env: OrtEnvironment by lazy { OrtEnvironment.getEnvironment() }
    val gpuProcessor: GlesGpuImageProcessor = GlesGpuImageProcessor()

    private var cpuFallbackSession: OrtSession? = null
    private var activeSrSession: OrtSession? = null

    private var activeProviderDescription: String = "CPU XNNPACK"
    private var benchmarkDetails: String = "Not benchmarked"
    private var lastInitDurationMs: Long = 0L
    private var nnapiDisabledDueToRuntimeError = false
    private var glesGpuAvailable = false

    // Reusable ONNX tile buffers (memory-optimized for 512px tiles: single-plane streaming eliminates ~127MB of 2048x2048 buffers)
    private var allocTileSize: Int = 0
    private var tileInArr: FloatArray = FloatArray(0)
    private var tileInBuf: FloatBuffer = FloatBuffer.wrap(tileInArr)
    private var tileInPlaneTmp: FloatArray = FloatArray(0)
    private var tileOutPlaneTmp: FloatArray = FloatArray(0)

    // Reusable OpenCV native C++ scratch Mats (zero per-tile/per-frame allocations)
    private var rgbMat: Mat? = null
    private var origFloatMat: Mat? = null
    private var tileResizedMat: Mat? = null
    private var tileFloatHwc: Mat? = null
    private var tileChannels: List<Mat>? = null
    private var outSinglePlaneMat: Mat? = null
    private var resizedOutChannels: List<Mat>? = null
    private var outResizedFloat: Mat? = null
    private var weightedTileFloat: Mat? = null
    private var accumRgbMat: Mat? = null
    private var accumWeightMat: Mat? = null
    private var accumWeight3CMat: Mat? = null
    private var weightClampFloorMat: Mat? = null
    private var blendedRgbFloat: Mat? = null
    private var outRgb8u: Mat? = null

    // Lightweight OpenCV SIMD detail scratch Mats
    private var lwBlurMat: Mat? = null
    private var lwDetailMat: Mat? = null

    // Feather weight Mat cache keyed by (isFirstX, isLastX, isFirstY, isLastY, cropW, cropH, ovX, ovY)
    private val featherMaskCache = HashMap<Long, Mat>()

    val isRealEsrganReady: Boolean get() = activeSrSession != null
    val isDncnnReady: Boolean get() = false
    val isGlesGpuReady: Boolean get() = glesGpuAvailable
    val providerLabel: String get() = activeProviderDescription
    val benchmarkSummary: String get() = benchmarkDetails
    val initDurationMs: Long get() = lastInitDurationMs

    @Synchronized
    fun initialize(
        capabilities: DeviceCapabilities,
        logger: EngineDiagnosticsLogger? = null
    ) {
        if (activeSrSession != null) {
            return
        }
        val t0 = System.currentTimeMillis()
        val modelDir = File(context.filesDir, "ai_models").apply { mkdirs() }
        val srFile = ensureAssetCopied(
            "ai_models/realesrgan/realesr_general_x4v3.onnx",
            File(modelDir, "realesrgan/realesr_general_x4v3.onnx")
        )

        glesGpuAvailable = gpuProcessor.ensureInitialized(logger)

        val availableProviders = try {
            OrtEnvironment.getAvailableProviders()
        } catch (_: Throwable) {
            EnumSet.of(OrtProvider.CPU)
        }
        logger?.info("Available ORT providers: ${availableProviders.joinToString()} • GLES31 Compute=$glesGpuAvailable")

        if (!srFile.exists() || srFile.length() == 0L) {
            logger?.error("Missing bundled Real-ESRGAN ONNX asset")
            lastInitDurationMs = System.currentTimeMillis() - t0
            return
        }

        val candidates = mutableListOf<Triple<String, OrtSession, Long>>()
        val notes = mutableListOf<String>()

        // 1. Benchmark CPU XNNPACK across 4, 6, and 8 threads (where supported by CPU cores)
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

        // Standard ORT CPU session fallback if XNNPACK config threw on this ABI
        if (bestCpuPair == null) {
            try {
                val std = createPlainCpuSession(srFile.absolutePath, capabilities.inferenceThreads.coerceIn(2, 8))
                val ms = benchmarkSrSession(std, tile = 128, timeoutSec = 6L, logger = logger) ?: 500L
                val label = "CPU ORT (${capabilities.inferenceThreads}T)"
                notes.add("CPU=${ms}ms")
                candidates.add(Triple(label, std, ms))
                bestCpuPair = label to std
                bestCpuMs = ms
            } catch (t: Throwable) {
                logger?.error("CPU ONNX session creation failed", t)
            }
        }

        cpuFallbackSession = bestCpuPair?.second

        // 2. Benchmark NNAPI / Vulkan-backed GPU HAL Hardware Acceleration
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
                        break // Found working hardware NNAPI session
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

        // 3. Pick fastest stable backend and close redundant sessions (keep cpuFallbackSession)
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

        benchmarkDetails = notes.joinToString(" | ").ifBlank { "No benchmark" }
        lastInitDurationMs = System.currentTimeMillis() - t0
        logger?.info("AI Backend selected: $activeProviderDescription • Benchmarks=[$benchmarkDetails] • Init=${lastInitDurationMs}ms")
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

    /**
     * High-speed tiled enhancement using native OpenCV SIMD C++ ROI/resize/feather accumulation
     * (zero Kotlin pixel loops) + automatic speed profiling that seamlessly shifts to
     * GPU/SIMD lightweight multi-scale detail reconstruction if neural tiles exceed the speed budget.
     */
    fun enhanceBgrWithPresetTiles(
        srcBgr: Mat,
        tileSize: Int,
        targetTilesPerFrame: Int,
        estimatedTotalFrames: Int,
        forceLightweightFallback: Boolean,
        isCancelled: () -> Boolean,
        logger: EngineDiagnosticsLogger? = null,
        onTileProgress: (done: Int, total: Int, tileSize: Int) -> Unit = { _, _, _ -> }
    ): TileProcessOutcome {
        if (forceLightweightFallback || activeSrSession == null) {
            val lw = enhanceFastLightweightCopy(srcBgr, sharpnessBoost = 0.34f, contrastBoost = 1.07f, satBoost = 1.05f)
            val label = if (glesGpuAvailable) "GLES 3.1 GPU Fast" else "OpenCV SIMD Fast"
            return TileProcessOutcome(
                outputBgr = lw,
                usedLightweightFallback = true,
                activeAccelerationLabel = label
            )
        }

        val inW = srcBgr.cols()
        val inH = srcBgr.rows()
        val safeTileSize = tileSize.coerceIn(128, 512)
        val grid = AiTilingMath.planDynamicTileGrid(inW, inH, targetTilesPerFrame)

        ensureNativeWorkMats(inW, inH, safeTileSize)
        ensureTileBuffers(safeTileSize)

        val localRgb = rgbMat!!
        Imgproc.cvtColor(srcBgr, localRgb, Imgproc.COLOR_BGR2RGB)
        localRgb.convertTo(origFloatMat!!, CvType.CV_32FC3, 1.0)

        accumRgbMat!!.setTo(Scalar(0.0, 0.0, 0.0))
        accumWeightMat!!.setTo(Scalar(0.0))

        val fixedTilePixels = safeTileSize * safeTileSize
        val fixedShape = longArrayOf(1L, 3L, safeTileSize.toLong(), safeTileSize.toLong())

        val resizeTarget = Size(safeTileSize.toDouble(), safeTileSize.toDouble())
        val chIn = tileChannels!!
        val resTile = tileResizedMat!!
        val floatTileHwc = tileFloatHwc!!
        val outResized = outResizedFloat!!
        val weightedTile = weightedTileFloat!!

        var tileTimingSumMs = 0L
        var timedTiles = 0

        for (tile in grid.tiles) {
            if (isCancelled()) throw InterruptedException("Cancelled during AI tile enhancement")

            val tTile0 = System.currentTimeMillis()

            // 1. Native SIMD ROI -> Resize -> Normalized Float -> NCHW bulk plane copy
            val cropRect = Rect(tile.x, tile.y, tile.width, tile.height)
            val roi = Mat(localRgb, cropRect)
            Imgproc.resize(roi, resTile, resizeTarget, 0.0, 0.0, Imgproc.INTER_LINEAR)
            roi.release()
            resTile.convertTo(floatTileHwc, CvType.CV_32FC3, 1.0 / 255.0)

            Core.split(floatTileHwc, chIn)
            chIn[0].get(0, 0, tileInPlaneTmp)
            System.arraycopy(tileInPlaneTmp, 0, tileInArr, 0, fixedTilePixels)
            chIn[1].get(0, 0, tileInPlaneTmp)
            System.arraycopy(tileInPlaneTmp, 0, tileInArr, fixedTilePixels, fixedTilePixels)
            chIn[2].get(0, 0, tileInPlaneTmp)
            System.arraycopy(tileInPlaneTmp, 0, tileInArr, 2 * fixedTilePixels, fixedTilePixels)
            tileInBuf.rewind()

            // 2. Memory-optimized ONNX Neural SR Inference + per-plane direct resize to tiny crop dimensions
            val cropSize = Size(tile.width.toDouble(), tile.height.toDouble())
            runSrTileAndResizeDirect(fixedShape, safeTileSize, cropSize, outResized, logger)

            // 3. Cached Overlap Feather Blend (prevents visible seams, preserves 100% frame content)
            val featherMat = getOrCreateFeatherMask(tile, grid.overlapX, grid.overlapY)
            val featherSingle = getOrCreateFeatherSingle(tile, grid.overlapX, grid.overlapY)

            Core.multiply(outResized, featherMat, weightedTile)
            val accumSub = Mat(accumRgbMat!!, cropRect)
            Core.add(accumSub, weightedTile, accumSub)
            accumSub.release()

            val weightSub = Mat(accumWeightMat!!, cropRect)
            Core.add(weightSub, featherSingle, weightSub)
            weightSub.release()

            val tileElapsed = (System.currentTimeMillis() - tTile0).coerceAtLeast(1L)
            tileTimingSumMs += tileElapsed
            timedTiles++
            onTileProgress(tile.index + 1, grid.totalTiles, safeTileSize)

            // Real-time Speed Profiler: quick bail after Tile 0 if massively over budget, else confirm after Tile 1
            val shouldFallbackNow = (timedTiles == 1 && tileElapsed * grid.totalTiles.toLong() > AiSpeedPolicy.MAX_ACCEPTABLE_AI_FRAME_MS * 2L) ||
                ((timedTiles == 2 || (timedTiles == grid.totalTiles && grid.totalTiles == 1)) &&
                    AiSpeedPolicy.shouldEngageLightweightFallback(tileTimingSumMs / timedTiles, grid.totalTiles, estimatedTotalFrames))

            if (shouldFallbackNow) {
                val avgTileMs = tileTimingSumMs / timedTiles
                val projectedMs = avgTileMs * grid.totalTiles
                val reason = "Auto-switched to Fast GPU/SIMD Engine: ${safeTileSize}px tile took ${avgTileMs}ms (~${projectedMs}ms/frame across ${grid.totalTiles} tiles)"
                logger?.warn(reason)
                val fastBgr = enhanceFastLightweightCopy(srcBgr, sharpnessBoost = 0.34f, contrastBoost = 1.07f, satBoost = 1.05f)
                val label = if (glesGpuAvailable) "GLES 3.1 GPU Fast" else "OpenCV SIMD Fast"
                activeProviderDescription = label
                return TileProcessOutcome(
                    outputBgr = fastBgr,
                    usedLightweightFallback = true,
                    fallbackReason = reason,
                    activeAccelerationLabel = label
                )
            }
        }

        // 4. Native SIMD Normalization + 80% AI / 20% Original crisp blend + BGR conversion
        Core.max(accumWeightMat!!, weightClampFloorMat!!, accumWeightMat!!)
        Core.merge(listOf(accumWeightMat!!, accumWeightMat!!, accumWeightMat!!), accumWeight3CMat!!)
        Core.divide(accumRgbMat!!, accumWeight3CMat!!, accumRgbMat!!)
        Core.min(accumRgbMat!!, Scalar(1.0, 1.0, 1.0), accumRgbMat!!)
        Core.max(accumRgbMat!!, Scalar(0.0, 0.0, 0.0), accumRgbMat!!)

        Core.addWeighted(accumRgbMat!!, 255.0 * 0.80, origFloatMat!!, 0.20, 0.0, blendedRgbFloat!!)
        blendedRgbFloat!!.convertTo(outRgb8u!!, CvType.CV_8UC3)

        val finalBgr = Mat(inH, inW, CvType.CV_8UC3)
        Imgproc.cvtColor(outRgb8u!!, finalBgr, Imgproc.COLOR_RGB2BGR)

        return TileProcessOutcome(
            outputBgr = finalBgr,
            usedLightweightFallback = false,
            activeAccelerationLabel = activeProviderDescription
        )
    }

    /**
     * Real lightweight hardware-first enhancement fallback:
     * Uses GLES 3.1 Compute Shader on GPU if supported, else native OpenCV C++ SIMD
     * multi-scale Laplacian detail reconstruction + crisp edge sharpening.
     */
    fun enhanceFastLightweightCopy(
        srcBgr: Mat,
        sharpnessBoost: Float = 0.32f,
        contrastBoost: Float = 1.06f,
        satBoost: Float = 1.05f
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
        // Native OpenCV SIMD fast high-frequency detail boost
        val blur = lwBlurMat ?: Mat().also { lwBlurMat = it }
        val detail = lwDetailMat ?: Mat().also { lwDetailMat = it }
        Imgproc.GaussianBlur(out, blur, Size(0.0, 0.0), 1.05)
        Core.addWeighted(out, 1.0 + sharpnessBoost, blur, -sharpnessBoost.toDouble(), 0.0, detail)
        detail.copyTo(out)
        return out
    }

    private fun runSrTileAndResizeDirect(
        shape: LongArray,
        tileSize: Int,
        cropSize: Size,
        outResizedHwc: Mat,
        logger: EngineDiagnosticsLogger?
    ) {
        val current = activeSrSession ?: throw IllegalStateException("Real-ESRGAN ONNX model is not loaded.")
        val inputName = current.inputNames.first()
        var tensor = OnnxTensor.createTensor(env, tileInBuf, shape)
        try {
            current.run(mapOf(inputName to tensor)).use { results ->
                val outTensor = results[0] as OnnxTensor
                extractPlanesAndMergeToCrop(outTensor.floatBuffer, tileSize, cropSize, outResizedHwc)
            }
        } catch (t: Throwable) {
            tensor.close()
            val cpu = cpuFallbackSession
            if (current !== cpu && cpu != null) {
                logger?.warn("NNAPI SR runtime error (${t.message}); switching permanently to CPU XNNPACK")
                nnapiDisabledDueToRuntimeError = true
                activeSrSession = cpu
                activeProviderDescription = "CPU XNNPACK (Runtime fallback)"
                tileInBuf.rewind()
                tensor = OnnxTensor.createTensor(env, tileInBuf, shape)
                try {
                    cpu.run(mapOf(cpu.inputNames.first() to tensor)).use { results ->
                        val outTensor = results[0] as OnnxTensor
                        extractPlanesAndMergeToCrop(outTensor.floatBuffer, tileSize, cropSize, outResizedHwc)
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
     * Streams each NCHW output plane directly from the ONNX FloatBuffer into a single
     * CV_32FC1 Mat, resizes down to the exact tile crop dimensions in native OpenCV SIMD,
     * and merges the 3 small channels — eliminating 127MB+ of full-resolution tile copies.
     */
    private fun extractPlanesAndMergeToCrop(
        outBuf: FloatBuffer,
        tileSize: Int,
        cropSize: Size,
        outResizedHwc: Mat
    ) {
        val outDim = tileSize * 4
        val outPixels = outDim * outDim
        val singlePlane = outSinglePlaneMat!!
        val smallCh = resizedOutChannels!!
        for (c in 0 until 3) {
            outBuf.position(c * outPixels)
            outBuf.get(tileOutPlaneTmp, 0, outPixels)
            singlePlane.put(0, 0, tileOutPlaneTmp)
            Imgproc.resize(singlePlane, smallCh[c], cropSize, 0.0, 0.0, Imgproc.INTER_LINEAR)
        }
        Core.merge(smallCh, outResizedHwc)
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
            outSinglePlaneMat?.release()
            outSinglePlaneMat = Mat(outDim, outDim, CvType.CV_32FC1)
            tileResizedMat?.release()
            tileResizedMat = Mat(tileSize, tileSize, CvType.CV_8UC3)
            tileFloatHwc?.release()
            tileFloatHwc = Mat(tileSize, tileSize, CvType.CV_32FC3)
        }
    }

    private fun ensureNativeWorkMats(frameW: Int, frameH: Int, tileSize: Int) {
        val currentW = rgbMat?.cols() ?: -1
        val currentH = rgbMat?.rows() ?: -1
        if (currentW == frameW && currentH == frameH) return

        rgbMat?.release()
        origFloatMat?.release()
        accumRgbMat?.release()
        accumWeightMat?.release()
        accumWeight3CMat?.release()
        weightClampFloorMat?.release()
        blendedRgbFloat?.release()
        outRgb8u?.release()
        clearFeatherCache()

        rgbMat = Mat(frameH, frameW, CvType.CV_8UC3)
        origFloatMat = Mat(frameH, frameW, CvType.CV_32FC3)
        accumRgbMat = Mat(frameH, frameW, CvType.CV_32FC3)
        accumWeightMat = Mat(frameH, frameW, CvType.CV_32FC1)
        accumWeight3CMat = Mat(frameH, frameW, CvType.CV_32FC3)
        weightClampFloorMat = Mat(frameH, frameW, CvType.CV_32FC1, Scalar(1e-5))
        blendedRgbFloat = Mat(frameH, frameW, CvType.CV_32FC3)
        outRgb8u = Mat(frameH, frameW, CvType.CV_8UC3)
        outResizedFloat = outResizedFloat ?: Mat()
        weightedTileFloat = weightedTileFloat ?: Mat()
    }

    private fun featherCacheKey(
        tile: AiTilingMath.TileRegion,
        ovX: Int,
        ovY: Int,
        singleChannel: Boolean
    ): Long {
        var h = tile.width.toLong()
        h = h * 31L + tile.height.toLong()
        h = h * 31L + ovX.toLong()
        h = h * 31L + ovY.toLong()
        val flags = (if (tile.isFirstX) 1 else 0) or
            (if (tile.isLastX) 2 else 0) or
            (if (tile.isFirstY) 4 else 0) or
            (if (tile.isLastY) 8 else 0) or
            (if (singleChannel) 16 else 0)
        return (h shl 5) or flags.toLong()
    }

    private fun getOrCreateFeatherSingle(tile: AiTilingMath.TileRegion, ovX: Int, ovY: Int): Mat {
        val key = featherCacheKey(tile, ovX, ovY, singleChannel = true)
        featherMaskCache[key]?.let { return it }

        val w = tile.width
        val h = tile.height
        val rowW = FloatArray(w) { dx ->
            AiTilingMath.featherWeight(dx, w, tile.isFirstX, tile.isLastX, ovX)
        }
        val colW = FloatArray(h) { dy ->
            AiTilingMath.featherWeight(dy, h, tile.isFirstY, tile.isLastY, ovY)
        }
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

    private fun getOrCreateFeatherMask(tile: AiTilingMath.TileRegion, ovX: Int, ovY: Int): Mat {
        val key = featherCacheKey(tile, ovX, ovY, singleChannel = false)
        featherMaskCache[key]?.let { return it }

        val single = getOrCreateFeatherSingle(tile, ovX, ovY)
        val mask3C = Mat(tile.height, tile.width, CvType.CV_32FC3)
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

        tileChannels?.forEach { it.release() }
        tileChannels = null
        resizedOutChannels?.forEach { it.release() }
        resizedOutChannels = null
        outSinglePlaneMat?.release(); outSinglePlaneMat = null
        tileResizedMat?.release(); tileResizedMat = null
        tileFloatHwc?.release(); tileFloatHwc = null
        outResizedFloat?.release(); outResizedFloat = null
        weightedTileFloat?.release(); weightedTileFloat = null

        rgbMat?.release(); rgbMat = null
        origFloatMat?.release(); origFloatMat = null
        accumRgbMat?.release(); accumRgbMat = null
        accumWeightMat?.release(); accumWeightMat = null
        accumWeight3CMat?.release(); accumWeight3CMat = null
        weightClampFloorMat?.release(); weightClampFloorMat = null
        blendedRgbFloat?.release(); blendedRgbFloat = null
        outRgb8u?.release(); outRgb8u = null
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
        } catch (_: Exception) {
        }
        cpuFallbackSession = null
        activeSrSession = null
    }
}
