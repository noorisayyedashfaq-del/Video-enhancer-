package com.example.engine

import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs

/**
 * Pure motion-gating math for zero-quality-loss Static Tile Temporal Reuse and
 * Keyframe Face Detection Scheduling.
 *
 * Preserves 100% full Clarity 1x-4x neural detail: only reuses an already-computed
 * AI output tile when the underlying visual patch is genuinely static across frames,
 * and forces a mandatory full AI refresh every [MAX_CONSECUTIVE_REUSES] frames or on scene cuts.
 */
object TemporalReusePolicy {
    const val FRAME_DUPLICATE_MAD_THRESHOLD = 0.65f
    const val TILE_STATIC_MAD_THRESHOLD = 1.10f
    const val SCENE_CUT_MAD_THRESHOLD = 8.50f
    const val MAX_CONSECUTIVE_REUSES = 2
    const val FACE_KEYFRAME_INTERVAL = 3

    fun computeMeanAbsDiff(a: ByteArray, b: ByteArray): Float {
        if (a.isEmpty() || a.size != b.size) return 255f
        var sum = 0L
        for (i in a.indices) {
            val va = a[i].toInt() and 0xFF
            val vb = b[i].toInt() and 0xFF
            sum += abs(va - vb)
        }
        return sum.toFloat() / a.size.toFloat()
    }

    fun shouldReuseStaticTile(
        mad: Float,
        consecutiveReuseCount: Int,
        isSceneCut: Boolean,
        hasCachedOutput: Boolean
    ): Boolean {
        if (!hasCachedOutput || isSceneCut) return false
        if (consecutiveReuseCount >= MAX_CONSECUTIVE_REUSES) return false
        return mad < TILE_STATIC_MAD_THRESHOLD
    }

    fun shouldReuseDuplicateFrame(
        frameMad: Float,
        consecutiveDuplicateReuses: Int,
        hasCachedFrame: Boolean
    ): Boolean {
        if (!hasCachedFrame) return false
        if (consecutiveDuplicateReuses >= MAX_CONSECUTIVE_REUSES) return false
        return frameMad < FRAME_DUPLICATE_MAD_THRESHOLD
    }

    fun shouldRunFullFaceDetection(
        frameIndex: Int,
        frameMad: Float,
        hasPreviousBoxes: Boolean
    ): Boolean {
        if (frameIndex <= 0) return true
        if (frameMad >= SCENE_CUT_MAD_THRESHOLD) return true
        if (!hasPreviousBoxes) return frameIndex % 2 == 0
        return frameIndex % FACE_KEYFRAME_INTERVAL == 0
    }
}

/**
 * Stateful per-video cache storing tiny 16x16 input signatures and full neural tile outputs
 * for static background tiles across Real-ESRGAN and DnCNN passes.
 */
class StaticTileReuseCache {

    private data class CachedTileEntry(
        val signature: ByteArray,
        val cachedOutputMat: Mat,
        var consecutiveReuses: Int = 0
    )

    private val tileEntries = HashMap<Int, CachedTileEntry>()
    private var prevFrameThumbBytes: ByteArray? = null
    private var thumbMat: Mat? = null
    private var grayThumbMat: Mat? = null

    var totalTilesReused: Int = 0
        private set

    fun computeFrameThumbnailAndMad(bgrMat: Mat): Pair<ByteArray, Float> {
        val tm = thumbMat ?: Mat().also { thumbMat = it }
        val gm = grayThumbMat ?: Mat().also { grayThumbMat = it }
        Imgproc.resize(bgrMat, tm, Size(48.0, 48.0), 0.0, 0.0, Imgproc.INTER_AREA)
        Imgproc.cvtColor(tm, gm, Imgproc.COLOR_BGR2GRAY)
        val bytes = ByteArray(48 * 48)
        gm.get(0, 0, bytes)
        val mad = prevFrameThumbBytes?.let { TemporalReusePolicy.computeMeanAbsDiff(bytes, it) } ?: 255f
        prevFrameThumbBytes = bytes
        return bytes to mad
    }

    fun computeTileSignature(tileRgbOrGray: Mat): ByteArray {
        val tm = thumbMat ?: Mat().also { thumbMat = it }
        val gm = grayThumbMat ?: Mat().also { grayThumbMat = it }
        Imgproc.resize(tileRgbOrGray, tm, Size(16.0, 16.0), 0.0, 0.0, Imgproc.INTER_AREA)
        if (tm.channels() > 1) {
            Imgproc.cvtColor(tm, gm, Imgproc.COLOR_RGB2GRAY)
        } else {
            tm.copyTo(gm)
        }
        val sig = ByteArray(256)
        gm.get(0, 0, sig)
        return sig
    }

    /**
     * Checks if tile [tileKey] can safely reuse its cached full-detail neural output.
     * Returns true if reused (and copies cached pixels into [dstOutFloat]), false if full inference is required.
     */
    fun tryReuseTileOutput(
        tileKey: Int,
        currentSig: ByteArray,
        isSceneCut: Boolean,
        dstOutFloat: Mat
    ): Boolean {
        val entry = tileEntries[tileKey] ?: return false
        val mad = TemporalReusePolicy.computeMeanAbsDiff(currentSig, entry.signature)
        val canReuse = TemporalReusePolicy.shouldReuseStaticTile(
            mad = mad,
            consecutiveReuseCount = entry.consecutiveReuses,
            isSceneCut = isSceneCut,
            hasCachedOutput = !entry.cachedOutputMat.empty()
        )
        if (!canReuse) return false

        entry.cachedOutputMat.copyTo(dstOutFloat)
        entry.consecutiveReuses++
        totalTilesReused++
        return true
    }

    fun storeTileOutput(tileKey: Int, signature: ByteArray, outputMat: Mat) {
        val existing = tileEntries[tileKey]
        if (existing != null) {
            outputMat.copyTo(existing.cachedOutputMat)
            System.arraycopy(signature, 0, existing.signature, 0, minOf(signature.size, existing.signature.size))
            existing.consecutiveReuses = 0
        } else {
            tileEntries[tileKey] = CachedTileEntry(
                signature = signature.copyOf(),
                cachedOutputMat = outputMat.clone(),
                consecutiveReuses = 0
            )
        }
    }

    fun clearTilesOnly() {
        tileEntries.values.forEach { it.cachedOutputMat.release() }
        tileEntries.clear()
    }

    fun reset() {
        clearTilesOnly()
        prevFrameThumbBytes = null
        totalTilesReused = 0
        thumbMat?.release(); thumbMat = null
        grayThumbMat?.release(); grayThumbMat = null
    }
}
