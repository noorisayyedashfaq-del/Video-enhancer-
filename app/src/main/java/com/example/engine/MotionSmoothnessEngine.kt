package com.example.engine

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.video.DISOpticalFlow
import org.opencv.video.Video
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.roundToLong

/**
 * Pure JVM-testable motion smoothness & optical flow frame-interpolation math:
 * - Decides cleanly when low-FPS cadence (<= 26.5 fps) benefits from 2x optical-flow interpolation
 *   while scaling target encoder FPS 2x so exact original playback speed & audio sync remain 1.0x.
 * - Detects VFR dropped-frame PTS gaps (>= 1.65 * median interval) for smooth repair synthesis.
 * - Detects accidentally frozen/stuck duplicate frames inside moving shots.
 * - Provides pure-Kotlin coarse block-motion + bidirectional bilinear optical-flow synthesis for
 *   deterministic JVM testing and zero-native fallback.
 */
object MotionSmoothMath {

    const val LOW_FPS_INTERPOLATION_THRESHOLD = 26.5
    const val VFR_DROP_GAP_MULTIPLIER = 1.65

    fun estimateMedianFrameIntervalUs(timestampsUs: List<Long>, fallbackFps: Double): Long {
        if (timestampsUs.size < 2) {
            return (1_000_000.0 / fallbackFps.coerceIn(10.0, 120.0)).roundToLong()
        }
        val deltas = ArrayList<Long>(timestampsUs.size - 1)
        for (i in 1 until timestampsUs.size) {
            val d = timestampsUs[i] - timestampsUs[i - 1]
            if (d in 4_000L..250_000L) deltas.add(d)
        }
        if (deltas.isEmpty()) {
            return (1_000_000.0 / fallbackFps.coerceIn(10.0, 120.0)).roundToLong()
        }
        deltas.sort()
        return deltas[deltas.size / 2]
    }

    fun shouldInterpolateCadence(sourceFps: Double, allowInterpolation: Boolean): Boolean {
        return allowInterpolation && sourceFps in 8.0..LOW_FPS_INTERPOLATION_THRESHOLD
    }

    fun targetOutputFps(sourceFps: Double, cadenceInterpolated: Boolean): Double {
        val safe = sourceFps.coerceIn(10.0, 120.0)
        return if (cadenceInterpolated) (safe * 2.0).coerceAtMost(60.0) else safe
    }

    fun detectDroppedFrameGap(prevPtsUs: Long, currPtsUs: Long, medianStepUs: Long): Int {
        if (prevPtsUs < 0L || currPtsUs <= prevPtsUs || medianStepUs <= 0L) return 0
        val delta = currPtsUs - prevPtsUs
        if (delta < (medianStepUs * VFR_DROP_GAP_MULTIPLIER).toLong()) return 0
        val missing = ((delta.toDouble() / medianStepUs.toDouble()) - 1.0).roundToInt()
        return missing.coerceIn(1, 2)
    }

    fun isStuckDuplicateFrame(prevMad: Float, currToPrevMad: Float): Boolean {
        return prevMad >= 2.2f && currToPrevMad < 0.35f
    }

    /**
     * Generates a monotonically spaced PTS list when interpolating midpoints between [ptsListUs].
     * Guarantees exact start/end duration lock (no slow-motion / fast-forward drift).
     */
    fun buildInterpolatedPtsSequence(ptsListUs: List<Long>, medianStepUs: Long): List<Long> {
        if (ptsListUs.isEmpty()) return emptyList()
        if (ptsListUs.size == 1) return ptsListUs
        val out = ArrayList<Long>(ptsListUs.size * 2)
        out.add(ptsListUs[0])
        for (i in 1 until ptsListUs.size) {
            val prev = out.last()
            val curr = max(ptsListUs[i], prev + 2_000L)
            val gap = curr - prev
            if (gap >= 4_000L) {
                val mid = prev + gap / 2L
                if (mid > prev && mid < curr) {
                    out.add(mid)
                }
            } else {
                out.add(prev + (medianStepUs / 2L).coerceAtLeast(1_000L))
            }
            out.add(max(curr, out.last() + 1_000L))
        }
        return out
    }

    /**
     * Pure-Kotlin block-motion optical flow + bidirectional warped midpoint synthesis.
     * Used directly in JVM unit tests and as an emergency fallback if OpenCV DISOpticalFlow cannot run.
     */
    fun synthesizeMidpointArgb(
        prevArgb: IntArray,
        nextArgb: IntArray,
        width: Int,
        height: Int,
        alpha: Float = 0.5f,
        blockSize: Int = 8,
        searchRadius: Int = 8
    ): IntArray {
        require(prevArgb.size == width * height && nextArgb.size == width * height) {
            "Frame dimensions do not match buffer sizes"
        }
        val lumPrev = IntArray(width * height)
        val lumNext = IntArray(width * height)
        for (i in prevArgb.indices) {
            val p = prevArgb[i]
            lumPrev[i] = (((p ushr 16) and 0xFF) * 77 + ((p ushr 8) and 0xFF) * 150 + (p and 0xFF) * 29) ushr 8
            val n = nextArgb[i]
            lumNext[i] = (((n ushr 16) and 0xFF) * 77 + ((n ushr 8) and 0xFF) * 150 + (n and 0xFF) * 29) ushr 8
        }

        val cols = (width + blockSize - 1) / blockSize
        val rows = (height + blockSize - 1) / blockSize
        val flowX = FloatArray(cols * rows)
        val flowY = FloatArray(cols * rows)

        for (by in 0 until rows) {
            val y0 = by * blockSize
            val bh = min(blockSize, height - y0)
            for (bx in 0 until cols) {
                val x0 = bx * blockSize
                val bw = min(blockSize, width - x0)

                var bestDx = 0
                var bestDy = 0
                var bestSad = Int.MAX_VALUE
                for (dy in -searchRadius..searchRadius) {
                    val ny0 = y0 + dy
                    if (ny0 < 0 || ny0 + bh > height) continue
                    for (dx in -searchRadius..searchRadius) {
                        val nx0 = x0 + dx
                        if (nx0 < 0 || nx0 + bw > width) continue
                        var sad = 0
                        for (py in 0 until bh step 2) {
                            val pRow = (y0 + py) * width + x0
                            val nRow = (ny0 + py) * width + nx0
                            for (px in 0 until bw step 2) {
                                sad += abs(lumPrev[pRow + px] - lumNext[nRow + px])
                            }
                        }
                        // Slight zero-motion tie-break preference prevents noisy vectors on flat areas
                        val penalty = (abs(dx) + abs(dy)) * 2
                        val score = sad + penalty
                        if (score < bestSad) {
                            bestSad = score
                            bestDx = dx
                            bestDy = dy
                        }
                    }
                }
                val bIdx = by * cols + bx
                flowX[bIdx] = bestDx.toFloat()
                flowY[bIdx] = bestDy.toFloat()
            }
        }

        val out = IntArray(width * height)
        val a = alpha.coerceIn(0f, 1f)
        val invA = 1f - a

        for (y in 0 until height) {
            val by = (y / blockSize).coerceIn(0, rows - 1)
            val rowOff = y * width
            for (x in 0 until width) {
                val bx = (x / blockSize).coerceIn(0, cols - 1)
                val bIdx = by * cols + bx
                val vx = flowX[bIdx]
                val vy = flowY[bIdx]

                // Bilinear sample along forward warp (-a*v from prev) & backward warp (+(1-a)*v from next)
                val px = (x - a * vx).coerceIn(0f, (width - 1).toFloat())
                val py = (y - a * vy).coerceIn(0f, (height - 1).toFloat())
                val nx = (x + invA * vx).coerceIn(0f, (width - 1).toFloat())
                val ny = (y + invA * vy).coerceIn(0f, (height - 1).toFloat())

                val pCol = sampleBilinearArgb(prevArgb, width, height, px, py)
                val nCol = sampleBilinearArgb(nextArgb, width, height, nx, ny)
                out[rowOff + x] = blendArgb(pCol, nCol, a)
            }
        }
        return out
    }

    private fun sampleBilinearArgb(buf: IntArray, w: Int, h: Int, fx: Float, fy: Float): Int {
        val x0 = fx.toInt().coerceIn(0, w - 1)
        val y0 = fy.toInt().coerceIn(0, h - 1)
        val x1 = min(x0 + 1, w - 1)
        val y1 = min(y0 + 1, h - 1)
        val tx = fx - x0
        val ty = fy - y0

        val c00 = buf[y0 * w + x0]
        val c10 = buf[y0 * w + x1]
        val c01 = buf[y1 * w + x0]
        val c11 = buf[y1 * w + x1]

        fun lerpCh(shift: Int): Int {
            val v00 = (c00 ushr shift) and 0xFF
            val v10 = (c10 ushr shift) and 0xFF
            val v01 = (c01 ushr shift) and 0xFF
            val v11 = (c11 ushr shift) and 0xFF
            val top = v00 + (v10 - v00) * tx
            val bot = v01 + (v11 - v01) * tx
            return (top + (bot - top) * ty).roundToInt().coerceIn(0, 255)
        }
        return (0xFF shl 24) or (lerpCh(16) shl 16) or (lerpCh(8) shl 8) or lerpCh(0)
    }

    private fun blendArgb(c0: Int, c1: Int, alpha: Float): Int {
        val w1 = alpha.coerceIn(0f, 1f)
        val w0 = 1f - w1
        val r = (((c0 ushr 16) and 0xFF) * w0 + ((c1 ushr 16) and 0xFF) * w1).roundToInt().coerceIn(0, 255)
        val g = (((c0 ushr 8) and 0xFF) * w0 + ((c1 ushr 8) and 0xFF) * w1).roundToInt().coerceIn(0, 255)
        val b = ((c0 and 0xFF) * w0 + (c1 and 0xFF) * w1).roundToInt().coerceIn(0, 255)
        return (0xFF shl 24) or (r shl 16) or (g shl 8) or b
    }
}

/**
 * Hardware/SIMD optical-flow motion smoothness engine built on OpenCV DISOpticalFlow:
 * 1. Computes fast dense optical flow on compact <= 192px proxy frames (~2-4ms per frame pair).
 * 2. Synthesizes crisp motion-compensated intermediate frames via OpenCV remap() warping
 *    for low-FPS cadence doubling and VFR dropped/stutter frame repairs.
 * 3. Applies motion-aware temporal micro-jitter & flicker stabilization when temporalConsistency is active.
 */
class MotionSmoothnessEngine {

    private var disFlow: DISOpticalFlow? = null
    private var prevGrayFullTmp: Mat? = null
    private var nextGrayFullTmp: Mat? = null
    private var prevGraySmall: Mat? = null
    private var nextGraySmall: Mat? = null
    private var flowSmall: Mat? = null
    private var flowFull: Mat? = null
    private var warpMapX: Mat? = null
    private var warpMapY: Mat? = null
    private var remapFwd: Mat? = null
    private var remapBwd: Mat? = null
    private var baseGridX: Mat? = null
    private var baseGridY: Mat? = null
    private var gridW: Int = 0
    private var gridH: Int = 0
    private var flowCh = ArrayList<Mat>(2)
    private var jitterPrevFrame: Mat? = null
    private var jitterDiff: Mat? = null
    private var jitterGrayDiff: Mat? = null
    private var jitterMask: Mat? = null
    private var jitterBlended: Mat? = null

    private fun getOrCreateDis(): DISOpticalFlow? {
        disFlow?.let { return it }
        return runCatching {
            DISOpticalFlow.create(DISOpticalFlow.PRESET_ULTRAFAST).apply {
                finestScale = 2
                patchSize = 8
                patchStride = 4
            }
        }.getOrNull()?.also { disFlow = it }
    }

    /**
     * Synthesizes an intermediate BGR frame at [alpha] between [prevBgr] and [nextBgr].
     * Returns true if optical-flow synthesis succeeded cleanly.
     */
    fun interpolateMidpointBgr(
        prevBgr: Mat,
        nextBgr: Mat,
        outBgr: Mat,
        alpha: Double = 0.5
    ): Boolean {
        if (prevBgr.empty() || nextBgr.empty()) return false
        val w = prevBgr.cols()
        val h = prevBgr.rows()
        if (nextBgr.cols() != w || nextBgr.rows() != h) {
            prevBgr.copyTo(outBgr)
            return false
        }

        return try {
            val flowLongEdge = 192.0
            val scale = (flowLongEdge / max(w, h).toDouble()).coerceIn(0.08, 1.0)
            val fw = max(32, ((w * scale).roundToInt() / 4) * 4)
            val fh = max(24, ((h * scale).roundToInt() / 4) * 4)

            val g0 = prevGraySmall ?: Mat().also { prevGraySmall = it }
            val g1 = nextGraySmall ?: Mat().also { nextGraySmall = it }
            val fSmall = flowSmall ?: Mat().also { flowSmall = it }
            val fFull = flowFull ?: Mat().also { flowFull = it }

            val pGrayTmp = prevGrayFullTmp ?: Mat().also { prevGrayFullTmp = it }
            val nGrayTmp = nextGrayFullTmp ?: Mat().also { nextGrayFullTmp = it }
            Imgproc.cvtColor(prevBgr, pGrayTmp, Imgproc.COLOR_BGR2GRAY)
            Imgproc.cvtColor(nextBgr, nGrayTmp, Imgproc.COLOR_BGR2GRAY)
            Imgproc.resize(pGrayTmp, g0, Size(fw.toDouble(), fh.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
            Imgproc.resize(nGrayTmp, g1, Size(fw.toDouble(), fh.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)

            val dis = getOrCreateDis()
            if (dis != null) {
                dis.calc(g0, g1, fSmall)
            } else {
                Video.calcOpticalFlowFarneback(g0, g1, fSmall, 0.5, 2, 11, 2, 5, 1.1, 0)
            }

            Imgproc.resize(fSmall, fFull, Size(w.toDouble(), h.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
            ensureIdentityGrids(w, h)

            flowCh.forEach { it.release() }
            flowCh.clear()
            Core.split(fFull, flowCh)

            val scaleFlowX = w.toDouble() / fw.toDouble()
            val scaleFlowY = h.toDouble() / fh.toDouble()
            val mapX = warpMapX!!
            val mapY = warpMapY!!
            val wFwd = remapFwd ?: Mat().also { remapFwd = it }
            val wBwd = remapBwd ?: Mat().also { remapBwd = it }

            // Forward warp prev -> t=alpha: sample prev at (x - alpha * vx, y - alpha * vy)
            Core.scaleAdd(flowCh[0], -alpha * scaleFlowX, baseGridX!!, mapX)
            Core.scaleAdd(flowCh[1], -alpha * scaleFlowY, baseGridY!!, mapY)
            Imgproc.remap(prevBgr, wFwd, mapX, mapY, Imgproc.INTER_LINEAR, Core.BORDER_REPLICATE)

            // Backward warp next -> t=alpha: sample next at (x + (1 - alpha) * vx, y + (1 - alpha) * vy)
            val invA = 1.0 - alpha
            Core.scaleAdd(flowCh[0], invA * scaleFlowX, baseGridX!!, mapX)
            Core.scaleAdd(flowCh[1], invA * scaleFlowY, baseGridY!!, mapY)
            Imgproc.remap(nextBgr, wBwd, mapX, mapY, Imgproc.INTER_LINEAR, Core.BORDER_REPLICATE)

            Core.addWeighted(wFwd, invA, wBwd, alpha, 0.0, outBgr)
            true
        } catch (_: Throwable) {
            // Safe linear blend fallback if native flow fails
            Core.addWeighted(prevBgr, 1.0 - alpha, nextBgr, alpha, 0.0, outBgr)
            false
        }
    }

    /**
     * Dampens temporal flicker & high-frequency shimmer on near-static regions without smearing moving objects.
     */
    fun stabilizeTemporalMicroJitter(
        bgr: Mat,
        strength: Float,
        isSceneCut: Boolean
    ) {
        if (bgr.empty() || strength < 0.05f) return
        val prev = jitterPrevFrame
        if (isSceneCut || prev == null || prev.cols() != bgr.cols() || prev.rows() != bgr.rows()) {
            jitterPrevFrame?.release()
            jitterPrevFrame = bgr.clone()
            return
        }

        val diff = jitterDiff ?: Mat().also { jitterDiff = it }
        val grayDiff = jitterGrayDiff ?: Mat().also { jitterGrayDiff = it }
        val mask = jitterMask ?: Mat().also { jitterMask = it }
        val blended = jitterBlended ?: Mat().also { jitterBlended = it }
        Core.absdiff(bgr, prev, diff)
        Imgproc.cvtColor(diff, grayDiff, Imgproc.COLOR_BGR2GRAY)
        // Only blend tiny flicker differences (< 9 luma levels) so rapid motion never trails/ghosts
        Imgproc.threshold(grayDiff, mask, 9.0, 255.0, Imgproc.THRESH_BINARY_INV)
        val alpha = (strength.toDouble() * 0.24).coerceIn(0.05, 0.20)
        Core.addWeighted(bgr, 1.0 - alpha, prev, alpha, 0.0, blended)
        blended.copyTo(bgr, mask)
        bgr.copyTo(prev)
    }

    private fun ensureIdentityGrids(w: Int, h: Int) {
        if (gridW == w && gridH == h && baseGridX != null && baseGridY != null) return
        baseGridX?.release()
        baseGridY?.release()
        warpMapX?.release()
        warpMapY?.release()

        val gx = FloatArray(w * h)
        val gy = FloatArray(w * h)
        for (y in 0 until h) {
            val rowOff = y * w
            val fy = y.toFloat()
            for (x in 0 until w) {
                gx[rowOff + x] = x.toFloat()
                gy[rowOff + x] = fy
            }
        }
        baseGridX = Mat(h, w, CvType.CV_32FC1).apply { put(0, 0, gx) }
        baseGridY = Mat(h, w, CvType.CV_32FC1).apply { put(0, 0, gy) }
        warpMapX = Mat(h, w, CvType.CV_32FC1)
        warpMapY = Mat(h, w, CvType.CV_32FC1)
        gridW = w
        gridH = h
    }

    fun release() {
        prevGrayFullTmp?.release(); prevGrayFullTmp = null
        nextGrayFullTmp?.release(); nextGrayFullTmp = null
        prevGraySmall?.release(); prevGraySmall = null
        nextGraySmall?.release(); nextGraySmall = null
        flowSmall?.release(); flowSmall = null
        flowFull?.release(); flowFull = null
        warpMapX?.release(); warpMapX = null
        warpMapY?.release(); warpMapY = null
        remapFwd?.release(); remapFwd = null
        remapBwd?.release(); remapBwd = null
        baseGridX?.release(); baseGridX = null
        baseGridY?.release(); baseGridY = null
        gridW = 0
        gridH = 0
        flowCh.forEach { it.release() }
        flowCh.clear()
        jitterPrevFrame?.release(); jitterPrevFrame = null
        jitterDiff?.release(); jitterDiff = null
        jitterGrayDiff?.release(); jitterGrayDiff = null
        jitterMask?.release(); jitterMask = null
        jitterBlended?.release(); jitterBlended = null
        disFlow = null
    }
}
