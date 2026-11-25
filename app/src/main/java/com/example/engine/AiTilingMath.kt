package com.example.engine

import kotlin.math.abs
import kotlin.math.ceil
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

    /**
     * Computes optimal neural working input dimensions for Real-ESRGAN x4v3 so that:
     * 1. It never feeds excessive full-res pixels that x4v3 would inflate 4x only to downsample.
     * 2. It fits cleanly within [maxTilesCap] tiles of [tileSize] without cutting field-of-view.
     */
    fun computeNeuralWorkingSize(
        inW: Int,
        inH: Int,
        outW: Int,
        outH: Int,
        tileSize: Int,
        maxTilesCap: Int
    ): Pair<Int, Int> {
        val safeInW = inW.coerceAtLeast(2)
        val safeInH = inH.coerceAtLeast(2)
        val safeTile = tileSize.coerceIn(128, 384)
        val cap = maxTilesCap.coerceAtLeast(1)
        val overlap = (safeTile * 0.10f).roundToInt().coerceIn(12, 32)
        val step = (safeTile - overlap).coerceAtLeast(64)

        // Ideal input size for 4x Real-ESRGAN model (~1/4 of target output), never exceeding input
        val idealW = ceil(outW.coerceAtLeast(safeInW) / 4.0).toInt().coerceAtMost(safeInW)
        val idealH = ceil(outH.coerceAtLeast(safeInH) / 4.0).toInt().coerceAtMost(safeInH)

        // Maximum span allowed cleanly by tile cap along dominant axis
        val (colsCap, rowsCap) = when {
            cap <= 1 -> 1 to 1
            cap == 2 -> if (safeInW >= safeInH) 2 to 1 else 1 to 2
            cap == 3 -> if (safeInW >= safeInH) 3 to 1 else 1 to 3
            cap <= 4 -> 2 to 2
            else -> if (safeInW >= safeInH) 3 to 2 else 2 to 3
        }
        val maxWByTiles = if (colsCap == 1) safeTile else safeTile + (colsCap - 1) * step
        val maxHByTiles = if (rowsCap == 1) safeTile else safeTile + (rowsCap - 1) * step

        val targetLongFloor = when (cap) {
            1 -> min(safeInW, safeTile)
            2 -> min(max(safeInW, idealW), max(maxWByTiles, maxHByTiles))
            else -> max(idealW, idealH).coerceAtLeast(min(max(safeInW, safeInH), safeTile * 2))
        }

        val scaleByW = min(safeInW.toDouble(), max(idealW, targetLongFloor).coerceAtMost(maxWByTiles).toDouble()) / safeInW.toDouble()
        val scaleByH = min(safeInH.toDouble(), max(idealH, targetLongFloor).coerceAtMost(maxHByTiles).toDouble()) / safeInH.toDouble()
        val scale = min(1.0, min(scaleByW, scaleByH)).coerceAtLeast(0.15)

        val workW = ((safeInW * scale).roundToInt() / 2 * 2).coerceIn(32, safeInW)
        val workH = ((safeInH * scale).roundToInt() / 2 * 2).coerceIn(32, safeInH)
        return workW to workH
    }
}

enum class SpeedExecutionTier(val label: String) {
    FULL_NEURAL_PER_FRAME("Full AI Per-Frame"),
    SINGLE_TILE_NEURAL("1-Tile AI Per-Frame"),
    KEYFRAME_NEURAL_HYBRID("Keyframe AI + GPU"),
    GPU_SHADER_FAST_PASS("GPU Shader Fast")
}

data class FrameSpeedDecision(
    val runNeuralSrNow: Boolean,
    val allowDnCnn: Boolean,
    val allowFaceDetect: Boolean,
    val allowFaceSr: Boolean,
    val maxTilesForFrame: Int,
    val tileSizeForFrame: Int,
    val tier: SpeedExecutionTier,
    val fastGpuMode: Boolean
)

/**
 * Enforces strict per-minute target ceilings across all 4 enhancement modes:
 * - HD <= 150s (2m 30s)
 * - HD+ <= 300s (5m 00s)
 * - FHD <= 450s (7m 30s)
 * - 2K <= 600s (10m 00s)
 * Dynamically tracks real EWMA ms/frame and steps down expensive ops cleanly to stay on budget.
 */
class SpeedBudgetGovernor(
    private val mode: com.example.model.EnhancementMode,
    durationMs: Long,
    estimatedFrames: Int,
    private val presetTileSize: Int,
    private val presetTilesCap: Int,
    measuredSrTileMs: Long,
    deviceCapacityScore: Int,
    measuredDnCnnMs: Long = 10L
) {
    val maxBudgetMs: Long
    val pacedBudgetMs: Long
    val targetMsPerFrame: Double
    private var currentTier: SpeedExecutionTier
    private var aiKeyframeInterval: Int
    private var emaFrameMs: Double = 0.0

    init {
        val maxPerMin = maxBudgetMsPerMinute(mode)
        val scaleFactor = (durationMs.coerceAtLeast(1_000L).toDouble() / 60_000.0)
        // Allow at least 12s overhead budget for very short clips (< 5s), scale linearly for longer videos
        maxBudgetMs = max(12_000L, (maxPerMin * scaleFactor).roundToInt().toLong())
        pacedBudgetMs = max(10_000L, (maxBudgetMs * 0.88).roundToInt().toLong())
        targetMsPerFrame = pacedBudgetMs.toDouble() / estimatedFrames.coerceAtLeast(1).toDouble()

        val estSingleTileMs = measuredSrTileMs.coerceIn(8L, 2_500L)
        // Double-buffered async DnCNN overlaps with post-processing/encoding (~50% effective critical path)
        val estDnCnnSerialMs = (measuredDnCnnMs.coerceIn(2L, 120L) * 0.50).roundToInt().toDouble()
        val fastGpuOverheadMs = (if (deviceCapacityScore >= 60) 12.0 else 22.0) + estDnCnnSerialMs

        when {
            // Fast GPU/NPU fits full multi-tile neural SR + mandatory per-frame DnCNN within strict budget
            estSingleTileMs * presetTilesCap + fastGpuOverheadMs <= targetMsPerFrame -> {
                currentTier = SpeedExecutionTier.FULL_NEURAL_PER_FRAME
                aiKeyframeInterval = 1
            }
            // Fits 1 compact neural SR tile + mandatory per-frame DnCNN on every frame
            estSingleTileMs + fastGpuOverheadMs <= targetMsPerFrame -> {
                currentTier = SpeedExecutionTier.SINGLE_TILE_NEURAL
                aiKeyframeInterval = 1
            }
            // Mid-speed NPU/CPU: run neural SR every K frames (K = 2..6), DnCNN ALWAYS runs on every frame
            estSingleTileMs + estDnCnnSerialMs <= targetMsPerFrame * 5.5 -> {
                currentTier = SpeedExecutionTier.KEYFRAME_NEURAL_HYBRID
                val k = ceil((estSingleTileMs - fastGpuOverheadMs) / (targetMsPerFrame - fastGpuOverheadMs).coerceAtLeast(6.0))
                    .toInt()
                    .coerceIn(2, 6)
                aiKeyframeInterval = k
            }
            // Slow device CPU fallback: use GLES 3.1 compute shader / SIMD fast SR + mandatory per-frame DnCNN
            else -> {
                currentTier = SpeedExecutionTier.GPU_SHADER_FAST_PASS
                aiKeyframeInterval = 9999
            }
        }
    }

    val activeTier: SpeedExecutionTier get() = currentTier
    val currentKeyframeInterval: Int get() = aiKeyframeInterval

    fun allowInterpolationByBaseline(): Boolean {
        return currentTier != SpeedExecutionTier.GPU_SHADER_FAST_PASS || targetMsPerFrame >= 45.0
    }

    fun decisionForFrame(frameIndex: Int, isSceneCut: Boolean, forceLightweight: Boolean): FrameSpeedDecision {
        // DnCNN noise/grain reduction runs on 100% of frames in ALL modes and ALL execution tiers (zero frame skipping).
        if (forceLightweight || currentTier == SpeedExecutionTier.GPU_SHADER_FAST_PASS) {
            return FrameSpeedDecision(
                runNeuralSrNow = false,
                allowDnCnn = true,
                allowFaceDetect = false,
                allowFaceSr = false,
                maxTilesForFrame = 0,
                tileSizeForFrame = presetTileSize,
                tier = SpeedExecutionTier.GPU_SHADER_FAST_PASS,
                fastGpuMode = true
            )
        }

        val isHighDetailMode = mode == com.example.model.EnhancementMode.FHD ||
            mode == com.example.model.EnhancementMode.TWO_K

        return when (currentTier) {
            SpeedExecutionTier.FULL_NEURAL_PER_FRAME -> {
                FrameSpeedDecision(
                    runNeuralSrNow = true,
                    allowDnCnn = true,
                    allowFaceDetect = isHighDetailMode && targetMsPerFrame >= 140.0,
                    allowFaceSr = mode == com.example.model.EnhancementMode.TWO_K && targetMsPerFrame >= 200.0,
                    maxTilesForFrame = presetTilesCap.coerceIn(1, 4),
                    tileSizeForFrame = presetTileSize,
                    tier = currentTier,
                    fastGpuMode = false
                )
            }
            SpeedExecutionTier.SINGLE_TILE_NEURAL -> FrameSpeedDecision(
                runNeuralSrNow = true,
                allowDnCnn = true,
                allowFaceDetect = false,
                allowFaceSr = false,
                maxTilesForFrame = 1,
                tileSizeForFrame = presetTileSize.coerceAtMost(192),
                tier = currentTier,
                fastGpuMode = false
            )
            SpeedExecutionTier.KEYFRAME_NEURAL_HYBRID -> {
                val interval = aiKeyframeInterval.coerceIn(2, 6)
                val runNow = (frameIndex % interval == 0) || isSceneCut
                FrameSpeedDecision(
                    runNeuralSrNow = runNow,
                    allowDnCnn = true,
                    allowFaceDetect = false,
                    allowFaceSr = false,
                    maxTilesForFrame = 1,
                    tileSizeForFrame = presetTileSize.coerceAtMost(192),
                    tier = currentTier,
                    fastGpuMode = !runNow
                )
            }
            SpeedExecutionTier.GPU_SHADER_FAST_PASS -> FrameSpeedDecision(
                runNeuralSrNow = false,
                allowDnCnn = true,
                allowFaceDetect = false,
                allowFaceSr = false,
                maxTilesForFrame = 0,
                tileSizeForFrame = presetTileSize,
                tier = currentTier,
                fastGpuMode = true
            )
        }
    }

    /**
     * Updates EMA speed and dynamically tightens cadence if remaining budget/frame shrinks below current speed.
     */
    fun onFrameCompleted(elapsedJobMs: Long, processedFrames: Int, totalFrames: Int, lastFrameMs: Long) {
        val sample = lastFrameMs.coerceAtLeast(1L).toDouble()
        emaFrameMs = if (emaFrameMs <= 0.0) sample else (emaFrameMs * 0.80 + sample * 0.20)
        if (processedFrames < 4) return

        val remainingFrames = (totalFrames - processedFrames).coerceAtLeast(1)
        val remainingBudgetMs = (pacedBudgetMs - elapsedJobMs).coerceAtLeast(0L).toDouble()
        val allowedPerRemainingFrame = remainingBudgetMs / remainingFrames.toDouble()

        if (emaFrameMs > allowedPerRemainingFrame * 1.08) {
            when (currentTier) {
                SpeedExecutionTier.FULL_NEURAL_PER_FRAME -> {
                    currentTier = SpeedExecutionTier.SINGLE_TILE_NEURAL
                }
                SpeedExecutionTier.SINGLE_TILE_NEURAL -> {
                    currentTier = SpeedExecutionTier.KEYFRAME_NEURAL_HYBRID
                    aiKeyframeInterval = 3
                }
                SpeedExecutionTier.KEYFRAME_NEURAL_HYBRID -> {
                    if (aiKeyframeInterval < 6) {
                        aiKeyframeInterval += 2
                    } else {
                        currentTier = SpeedExecutionTier.GPU_SHADER_FAST_PASS
                    }
                }
                SpeedExecutionTier.GPU_SHADER_FAST_PASS -> Unit
            }
        }
    }

    companion object {
        fun maxBudgetMsPerMinute(mode: com.example.model.EnhancementMode): Long = when (mode) {
            com.example.model.EnhancementMode.HD -> 150_000L // 2m 30s
            com.example.model.EnhancementMode.HD_PLUS -> 300_000L // 5m 00s
            com.example.model.EnhancementMode.FHD -> 450_000L // 7m 30s
            com.example.model.EnhancementMode.TWO_K -> 600_000L // 10m 00s
        }
    }
}

/**
 * Speed-first safeguard policy: protects the strict 150s-600s/min deadlines so heavy CPU stalls
 * transition rapidly into hardware GPU/SIMD acceleration instead of multi-hour runs.
 */
object AiSpeedPolicy {
    const val MAX_ACCEPTABLE_AI_FRAME_MS = 350L
    const val MAX_ACCEPTABLE_SINGLE_TILE_MS_HEAVY = 300L

    fun shouldEngageLightweightFallback(
        measuredTileMs: Long,
        tilesPerFrame: Int,
        estimatedFrameCount: Int,
        modeBudgetFrameMs: Long = MAX_ACCEPTABLE_AI_FRAME_MS
    ): Boolean {
        if (tilesPerFrame <= 0 || measuredTileMs <= 0L) return false
        if (measuredTileMs > MAX_ACCEPTABLE_SINGLE_TILE_MS_HEAVY) return true
        val projectedPerFrameMs = measuredTileMs * tilesPerFrame.toLong()
        val effectiveLimit = modeBudgetFrameMs.coerceIn(80L, MAX_ACCEPTABLE_AI_FRAME_MS)
        return estimatedFrameCount >= 60 && projectedPerFrameMs > effectiveLimit
    }
}
