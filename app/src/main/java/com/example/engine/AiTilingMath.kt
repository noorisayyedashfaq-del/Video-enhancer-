package com.example.engine

import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
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
}

/**
 * Safeguard policy that prioritizes real ONNX AI execution (stepping down hardware fallback chains
 * cleanly before ever considering non-neural emergency recovery on multi-second tile hangs).
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
