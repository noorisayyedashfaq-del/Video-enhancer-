package com.example.engine

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtProvider
import ai.onnxruntime.OrtSession
import android.content.Context
import android.os.Build
import com.example.model.DeviceCapabilities
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.io.File
import java.nio.FloatBuffer
import java.util.EnumSet
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
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

class OnnxAiInferenceEngine(private val context: Context) {

    private val env: OrtEnvironment by lazy { OrtEnvironment.getEnvironment() }

    private var cpuSrSession: OrtSession? = null
    private var nnapiSrSession: OrtSession? = null
    private var activeSrSession: OrtSession? = null

    private var activeProviderDescription: String = "CPU XNNPACK"
    private var benchmarkDetails: String = "Not benchmarked"
    private var lastInitDurationMs: Long = 0L
    private var nnapiDisabledDueToRuntimeError = false

    // Reusable per-tile buffers (zero per-tile allocations)
    private var allocTileSize: Int = 0
    private var tileInArr: FloatArray = FloatArray(0)
    private var tileInBuf: FloatBuffer = FloatBuffer.wrap(tileInArr)
    private var tileOutArr: FloatArray = FloatArray(0)

    // Reusable original-resolution frame accumulators
    private var allocFrameW: Int = 0
    private var allocFrameH: Int = 0
    private var srcRgbBytes: ByteArray = ByteArray(0)
    private var targetAccumRgb: FloatArray = FloatArray(0)
    private var targetAccumWeight: FloatArray = FloatArray(0)
    private var targetOutBytes: ByteArray = ByteArray(0)

    // Reusable tile coordinate/feather LUTs
    private var cachedLutTileSize: Int = 0
    private var cachedLutCropW: Int = 0
    private var cachedLutCropH: Int = 0
    private var cachedLutOvX: Int = 0
    private var cachedLutOvY: Int = 0
    private var packX0: IntArray = IntArray(0)
    private var packX1: IntArray = IntArray(0)
    private var packFx: FloatArray = FloatArray(0)
    private var packY0: IntArray = IntArray(0)
    private var packY1: IntArray = IntArray(0)
    private var packFy: FloatArray = FloatArray(0)
    private var outTapX0: IntArray = IntArray(0)
    private var outTapX1: IntArray = IntArray(0)
    private var outTapY0: IntArray = IntArray(0)
    private var outTapY1: IntArray = IntArray(0)
    private var midFeatherX: FloatArray = FloatArray(0)
    private var midFeatherY: FloatArray = FloatArray(0)

    val isRealEsrganReady: Boolean get() = activeSrSession != null
    val isDncnnReady: Boolean get() = false // Fast path uses OpenCV temporal/spatial denoise; only preset AI tiles run
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

        val availableProviders = try {
            OrtEnvironment.getAvailableProviders()
        } catch (_: Throwable) {
            EnumSet.of(OrtProvider.CPU)
        }
        logger?.info("Available ORT providers: ${availableProviders.joinToString()}")

        if (cpuSrSession == null && srFile.exists() && srFile.length() > 0L) {
            cpuSrSession = createCpuSession(srFile.absolutePath, capabilities)
            activeSrSession = cpuSrSession
        }

        val srCpu = cpuSrSession
        if (srCpu != null) {
            val cpuBenchMs = benchmarkSrSession(srCpu, tile = 64, timeoutSec = 8L, logger = logger)
            var chosenLabel = "CPU XNNPACK (${capabilities.inferenceThreads}T)"
            var benchSummary = "CPU=${cpuBenchMs?.let { "${it}ms" } ?: "N/A"}"

            val nnapiCandidateAllowed = !nnapiDisabledDueToRuntimeError &&
                capabilities.nnapiSupported &&
                Build.VERSION.SDK_INT >= Build.VERSION_CODES.P &&
                availableProviders.contains(OrtProvider.NNAPI) &&
                capabilities.processingCapacityScore >= 50

            if (nnapiCandidateAllowed && cpuBenchMs != null) {
                var nnapiCandidate: OrtSession? = null
                try {
                    nnapiCandidate = createNnapiSession(srFile.absolutePath, capabilities)
                    val nnapiBenchMs = benchmarkSrSession(nnapiCandidate, tile = 64, timeoutSec = 6L, logger = logger)
                    benchSummary = "NNAPI=${nnapiBenchMs?.let { "${it}ms" } ?: "FAIL"} vs CPU=${cpuBenchMs}ms"
                    if (nnapiBenchMs != null && nnapiBenchMs < (cpuBenchMs * 0.90).toLong().coerceAtLeast(1L)) {
                        nnapiSrSession = nnapiCandidate
                        activeSrSession = nnapiCandidate
                        chosenLabel = "NNAPI Hardware (${nnapiBenchMs}ms/tile)"
                    } else {
                        nnapiCandidate.close()
                        logger?.info("NNAPI benchmark ($benchSummary) not faster than CPU; selected CPU XNNPACK")
                    }
                } catch (t: Throwable) {
                    logger?.warn("NNAPI init/benchmark failed (${t.message}); using CPU XNNPACK")
                    try {
                        nnapiCandidate?.close()
                    } catch (_: Throwable) {
                    }
                    activeSrSession = cpuSrSession
                }
            }

            activeProviderDescription = chosenLabel
            benchmarkDetails = benchSummary
        }

        lastInitDurationMs = System.currentTimeMillis() - t0
        logger?.info("AI Runtime initialized in ${lastInitDurationMs}ms • Provider: $activeProviderDescription • Benchmark: $benchmarkDetails")
    }

    private fun createCpuSession(modelPath: String, capabilities: DeviceCapabilities): OrtSession {
        val cpuOpts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(capabilities.inferenceThreads.coerceIn(1, 6))
            setInterOpNumThreads(1)
            setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        return env.createSession(modelPath, cpuOpts)
    }

    private fun createNnapiSession(modelPath: String, capabilities: DeviceCapabilities): OrtSession {
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(capabilities.inferenceThreads.coerceIn(1, 4))
            setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            addNnapi()
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
     * Enhances [srcBgr] at its exact original resolution ([inW] x [inH]) by dynamically dividing
     * the frame into [targetTilesPerFrame] overlapping tiles (6 for HD+, 12 for 2K, 25 for 4K),
     * running each tile through the ONNX AI model at [tileSize]x[tileSize] (512px / 384px / 256px),
     * and feather-blending back into reusable original-resolution buffers with zero cropping.
     */
    fun enhanceBgrWithPresetTiles(
        srcBgr: Mat,
        tileSize: Int,
        targetTilesPerFrame: Int,
        isCancelled: () -> Boolean,
        logger: EngineDiagnosticsLogger? = null,
        onTileProgress: (done: Int, total: Int, tileSize: Int) -> Unit = { _, _, _ -> }
    ): Mat {
        val inW = srcBgr.cols()
        val inH = srcBgr.rows()
        val safeTileSize = tileSize.coerceIn(128, 512)
        val grid = AiTilingMath.planDynamicTileGrid(inW, inH, targetTilesPerFrame)
        val firstTile = grid.tiles.first()

        ensureTileBuffers(safeTileSize)
        ensureFrameBuffers(inW, inH)
        ensureTileLuts(
            tileSize = safeTileSize,
            cropW = firstTile.width,
            cropH = firstTile.height,
            overlapX = grid.overlapX,
            overlapY = grid.overlapY
        )

        // Convert entire frame once to RGB bytes
        val rgbMat = Mat()
        Imgproc.cvtColor(srcBgr, rgbMat, Imgproc.COLOR_BGR2RGB)
        rgbMat.get(0, 0, srcRgbBytes)
        rgbMat.release()

        targetAccumRgb.fill(0f)
        targetAccumWeight.fill(0f)

        val fixedTilePixels = safeTileSize * safeTileSize
        val outModelSize = safeTileSize * 4
        val outChannelStride = outModelSize * outModelSize
        val fixedShape = longArrayOf(1L, 3L, safeTileSize.toLong(), safeTileSize.toLong())
        val inv255 = 1f / 255f

        for (tile in grid.tiles) {
            if (isCancelled()) throw InterruptedException("Cancelled during AI tile enhancement")

            // 1. Pack source crop [tile.x..tile.x+tile.width, tile.y..tile.y+tile.height] -> [1, 3, tileSize, tileSize]
            val baseX = tile.x
            val baseY = tile.y
            for (ty in 0 until safeTileSize) {
                val sy0 = (baseY + packY0[ty]) * inW
                val sy1 = (baseY + packY1[ty]) * inW
                val fy = packFy[ty]
                val invFy = 1f - fy
                val dstRow = ty * safeTileSize

                for (tx in 0 until safeTileSize) {
                    val sx0 = baseX + packX0[tx]
                    val sx1 = baseX + packX1[tx]
                    val fx = packFx[tx]
                    val invFx = 1f - fx

                    val w00 = invFx * invFy * inv255
                    val w10 = fx * invFy * inv255
                    val w01 = invFx * fy * inv255
                    val w11 = fx * fy * inv255

                    val p00 = (sy0 + sx0) * 3
                    val p10 = (sy0 + sx1) * 3
                    val p01 = (sy1 + sx0) * 3
                    val p11 = (sy1 + sx1) * 3

                    val dIdx = dstRow + tx
                    tileInArr[dIdx] =
                        (srcRgbBytes[p00].toInt() and 0xFF) * w00 +
                        (srcRgbBytes[p10].toInt() and 0xFF) * w10 +
                        (srcRgbBytes[p01].toInt() and 0xFF) * w01 +
                        (srcRgbBytes[p11].toInt() and 0xFF) * w11

                    tileInArr[fixedTilePixels + dIdx] =
                        (srcRgbBytes[p00 + 1].toInt() and 0xFF) * w00 +
                        (srcRgbBytes[p10 + 1].toInt() and 0xFF) * w10 +
                        (srcRgbBytes[p01 + 1].toInt() and 0xFF) * w01 +
                        (srcRgbBytes[p11 + 1].toInt() and 0xFF) * w11

                    tileInArr[2 * fixedTilePixels + dIdx] =
                        (srcRgbBytes[p00 + 2].toInt() and 0xFF) * w00 +
                        (srcRgbBytes[p10 + 2].toInt() and 0xFF) * w10 +
                        (srcRgbBytes[p01 + 2].toInt() and 0xFF) * w01 +
                        (srcRgbBytes[p11 + 2].toInt() and 0xFF) * w11
                }
            }
            tileInBuf.rewind()

            // 2. Run ONNX Real-ESRGAN inference at fixed [1, 3, tileSize, tileSize]
            runSrTileWithFallback(fixedShape, logger)

            // 3. Project enhanced tile back onto original tile coordinates with overlap feathering (no cropping!)
            val cropW = tile.width
            val cropH = tile.height
            val ovX = grid.overlapX
            val ovY = grid.overlapY

            for (dy in 0 until cropH) {
                val wy = when {
                    !tile.isFirstY && dy < ovY -> midFeatherY[dy]
                    !tile.isLastY && dy >= cropH - ovY -> midFeatherY[cropH - 1 - dy]
                    else -> 1f
                }
                val oy0 = outTapY0[dy] * outModelSize
                val oy1 = outTapY1[dy] * outModelSize
                val dstRow = (baseY + dy) * inW + baseX

                for (dx in 0 until cropW) {
                    val wx = when {
                        !tile.isFirstX && dx < ovX -> midFeatherX[dx]
                        !tile.isLastX && dx >= cropW - ovX -> midFeatherX[cropW - 1 - dx]
                        else -> 1f
                    }
                    val w = wx * wy
                    val ox0 = outTapX0[dx]
                    val ox1 = outTapX1[dx]

                    val i00 = oy0 + ox0
                    val i10 = oy0 + ox1
                    val i01 = oy1 + ox0
                    val i11 = oy1 + ox1

                    val r = ((tileOutArr[i00] + tileOutArr[i10] + tileOutArr[i01] + tileOutArr[i11]) * 0.25f).coerceIn(0f, 1f)
                    val g = ((tileOutArr[outChannelStride + i00] + tileOutArr[outChannelStride + i10] + tileOutArr[outChannelStride + i01] + tileOutArr[outChannelStride + i11]) * 0.25f).coerceIn(0f, 1f)
                    val b = ((tileOutArr[2 * outChannelStride + i00] + tileOutArr[2 * outChannelStride + i10] + tileOutArr[2 * outChannelStride + i01] + tileOutArr[2 * outChannelStride + i11]) * 0.25f).coerceIn(0f, 1f)

                    val dstPixel = dstRow + dx
                    val dstBase = dstPixel * 3
                    targetAccumRgb[dstBase] += r * w
                    targetAccumRgb[dstBase + 1] += g * w
                    targetAccumRgb[dstBase + 2] += b * w
                    targetAccumWeight[dstPixel] += w
                }
            }

            onTileProgress(tile.index + 1, grid.totalTiles, safeTileSize)
        }

        // Fast blend: combine AI reconstruction (80%) + crisp original input (20%) to preserve fine grain/speed
        val totalPixels = inW * inH
        for (i in 0 until totalPixels) {
            val w = targetAccumWeight[i].coerceAtLeast(1e-5f)
            val b = i * 3
            val aiR = (targetAccumRgb[b] / w) * 255f
            val aiG = (targetAccumRgb[b + 1] / w) * 255f
            val aiB = (targetAccumRgb[b + 2] / w) * 255f

            val origR = (srcRgbBytes[b].toInt() and 0xFF).toFloat()
            val origG = (srcRgbBytes[b + 1].toInt() and 0xFF).toFloat()
            val origB = (srcRgbBytes[b + 2].toInt() and 0xFF).toFloat()

            val outR = (aiR * 0.80f + origR * 0.20f).roundToInt().coerceIn(0, 255).toByte()
            val outG = (aiG * 0.80f + origG * 0.20f).roundToInt().coerceIn(0, 255).toByte()
            val outB = (aiB * 0.80f + origB * 0.20f).roundToInt().coerceIn(0, 255).toByte()

            targetOutBytes[b] = outB
            targetOutBytes[b + 1] = outG
            targetOutBytes[b + 2] = outR
        }

        val finalMat = Mat(inH, inW, srcBgr.type())
        finalMat.put(0, 0, targetOutBytes)
        return finalMat
    }

    private fun runSrTileWithFallback(shape: LongArray, logger: EngineDiagnosticsLogger?) {
        val current = activeSrSession ?: throw IllegalStateException("Real-ESRGAN ONNX model is not loaded.")
        val inputName = current.inputNames.first()
        var tensor = OnnxTensor.createTensor(env, tileInBuf, shape)
        try {
            current.run(mapOf(inputName to tensor)).use { results ->
                val outTensor = results[0] as OnnxTensor
                outTensor.floatBuffer.get(tileOutArr)
            }
        } catch (t: Throwable) {
            tensor.close()
            val cpu = cpuSrSession
            if (current !== cpu && cpu != null) {
                logger?.warn("NNAPI SR runtime error (${t.message}); falling back to CPU XNNPACK")
                nnapiDisabledDueToRuntimeError = true
                activeSrSession = cpu
                activeProviderDescription = "CPU XNNPACK (NNAPI runtime fallback)"
                tileInBuf.rewind()
                tensor = OnnxTensor.createTensor(env, tileInBuf, shape)
                try {
                    cpu.run(mapOf(cpu.inputNames.first() to tensor)).use { results ->
                        val outTensor = results[0] as OnnxTensor
                        outTensor.floatBuffer.get(tileOutArr)
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

    private fun ensureTileBuffers(tileSize: Int) {
        if (allocTileSize != tileSize) {
            allocTileSize = tileSize
            tileInArr = FloatArray(3 * tileSize * tileSize)
            tileInBuf = FloatBuffer.wrap(tileInArr)
            val outDim = tileSize * 4
            tileOutArr = FloatArray(3 * outDim * outDim)
        }
    }

    private fun ensureFrameBuffers(frameW: Int, frameH: Int) {
        if (allocFrameW != frameW || allocFrameH != frameH) {
            val pixels = frameW * frameH
            allocFrameW = frameW
            allocFrameH = frameH
            srcRgbBytes = ByteArray(pixels * 3)
            targetAccumRgb = FloatArray(pixels * 3)
            targetAccumWeight = FloatArray(pixels)
            targetOutBytes = ByteArray(pixels * 3)
        }
    }

    private fun ensureTileLuts(
        tileSize: Int,
        cropW: Int,
        cropH: Int,
        overlapX: Int,
        overlapY: Int
    ) {
        if (cachedLutTileSize == tileSize &&
            cachedLutCropW == cropW &&
            cachedLutCropH == cropH &&
            cachedLutOvX == overlapX &&
            cachedLutOvY == overlapY
        ) {
            return
        }
        cachedLutTileSize = tileSize
        cachedLutCropW = cropW
        cachedLutCropH = cropH
        cachedLutOvX = overlapX
        cachedLutOvY = overlapY

        // Input pack LUT: [0 until tileSize] -> [0 until cropW/cropH]
        packX0 = IntArray(tileSize)
        packX1 = IntArray(tileSize)
        packFx = FloatArray(tileSize)
        val scaleInX = if (tileSize > 1) (cropW - 1).toDouble() / (tileSize - 1).toDouble() else 0.0
        for (tx in 0 until tileSize) {
            val sx = (tx * scaleInX).coerceIn(0.0, (cropW - 1).toDouble())
            val i0 = floor(sx).toInt().coerceIn(0, cropW - 1)
            packX0[tx] = i0
            packX1[tx] = (i0 + 1).coerceAtMost(cropW - 1)
            packFx[tx] = (sx - i0).toFloat()
        }

        packY0 = IntArray(tileSize)
        packY1 = IntArray(tileSize)
        packFy = FloatArray(tileSize)
        val scaleInY = if (tileSize > 1) (cropH - 1).toDouble() / (tileSize - 1).toDouble() else 0.0
        for (ty in 0 until tileSize) {
            val sy = (ty * scaleInY).coerceIn(0.0, (cropH - 1).toDouble())
            val i0 = floor(sy).toInt().coerceIn(0, cropH - 1)
            packY0[ty] = i0
            packY1[ty] = (i0 + 1).coerceAtMost(cropH - 1)
            packFy[ty] = (sy - i0).toFloat()
        }

        // Output 2x2 area tap LUT: [0 until cropW/cropH] -> [0 until outModelSize]
        val outDim = tileSize * 4
        outTapX0 = IntArray(cropW)
        outTapX1 = IntArray(cropW)
        val scaleOutX = outDim.toDouble() / cropW.toDouble()
        for (dx in 0 until cropW) {
            val cx = ((dx + 0.5) * scaleOutX - 0.5).coerceIn(0.0, (outDim - 1).toDouble())
            val x0 = floor(cx).toInt().coerceIn(0, outDim - 1)
            outTapX0[dx] = x0
            outTapX1[dx] = (x0 + 1).coerceAtMost(outDim - 1)
        }

        outTapY0 = IntArray(cropH)
        outTapY1 = IntArray(cropH)
        val scaleOutY = outDim.toDouble() / cropH.toDouble()
        for (dy in 0 until cropH) {
            val cy = ((dy + 0.5) * scaleOutY - 0.5).coerceIn(0.0, (outDim - 1).toDouble())
            val y0 = floor(cy).toInt().coerceIn(0, outDim - 1)
            outTapY0[dy] = y0
            outTapY1[dy] = (y0 + 1).coerceAtMost(outDim - 1)
        }

        midFeatherX = FloatArray(overlapX.coerceAtLeast(1)) { d ->
            ((d + 1).toFloat() / overlapX.coerceAtLeast(1).toFloat()).coerceIn(0.05f, 1f)
        }
        midFeatherY = FloatArray(overlapY.coerceAtLeast(1)) { d ->
            ((d + 1).toFloat() / overlapY.coerceAtLeast(1).toFloat()).coerceIn(0.05f, 1f)
        }
    }

    fun releaseWorkBuffers() {
        allocTileSize = 0
        tileInArr = FloatArray(0)
        tileInBuf = FloatBuffer.wrap(tileInArr)
        tileOutArr = FloatArray(0)

        allocFrameW = 0
        allocFrameH = 0
        srcRgbBytes = ByteArray(0)
        targetAccumRgb = FloatArray(0)
        targetAccumWeight = FloatArray(0)
        targetOutBytes = ByteArray(0)

        cachedLutTileSize = 0
        cachedLutCropW = 0
        cachedLutCropH = 0
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
        try {
            nnapiSrSession?.takeIf { it !== cpuSrSession }?.close()
            cpuSrSession?.close()
        } catch (_: Exception) {
        }
        nnapiSrSession = null
        cpuSrSession = null
        activeSrSession = null
    }
}
