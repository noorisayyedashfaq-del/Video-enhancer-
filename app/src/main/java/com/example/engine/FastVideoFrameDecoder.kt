package com.example.engine

import android.media.Image
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.media.MediaFormat
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.io.File
import java.nio.ByteBuffer
import java.util.ArrayDeque

enum class YuvFastLayout {
    NV12,
    NV21,
    PLANAR_I420
}

/**
 * Pure JVM-testable YUV packing & semi-planar layout heuristics for zero-disk MediaCodec decoding.
 */
object FastYuvCopyMath {

    fun detectLayout(
        uPixelStride: Int,
        vPixelStride: Int,
        uRemaining: Int,
        vRemaining: Int,
        uProbe0: Byte,
        uProbe1: Byte,
        uProbe2: Byte,
        uProbe3: Byte,
        vProbe0: Byte,
        vProbe1: Byte,
        vProbe2: Byte,
        vProbe3: Byte
    ): YuvFastLayout {
        if (uPixelStride == 2 && vPixelStride == 2) {
            if (uRemaining > vRemaining) return YuvFastLayout.NV12
            if (vRemaining > uRemaining) return YuvFastLayout.NV21
            val looksNv12 = (uProbe1 == vProbe0) && (uProbe3 == vProbe2)
            val looksNv21 = (vProbe1 == uProbe0) && (vProbe3 == uProbe2)
            if (looksNv12 && !looksNv21) return YuvFastLayout.NV12
            if (looksNv21 && !looksNv12) return YuvFastLayout.NV21
        }
        return YuvFastLayout.PLANAR_I420
    }

    fun copySemiPlanarUvPlane(
        interleavedBuf: ByteBuffer,
        cropLeftHalf: Int,
        cropTopHalf: Int,
        uvWidth: Int,
        uvHeight: Int,
        rowStride: Int,
        dst: ByteArray,
        dstOffset: Int
    ) {
        val buf = interleavedBuf.duplicate()
        val rowBytes = uvWidth * 2
        if (rowStride == rowBytes && cropLeftHalf == 0 && cropTopHalf == 0) {
            buf.position(0)
            val needed = rowBytes * uvHeight
            val avail = buf.remaining().coerceAtMost(needed)
            buf.get(dst, dstOffset, avail)
            return
        }
        val limit = buf.limit()
        var outPos = dstOffset
        for (r in 0 until uvHeight) {
            val srcPos = (cropTopHalf + r) * rowStride + cropLeftHalf * 2
            val avail = (limit - srcPos).coerceAtLeast(0).coerceAtMost(rowBytes)
            if (avail > 0) {
                buf.position(srcPos)
                buf.get(dst, outPos, avail)
            }
            outPos += rowBytes
        }
    }

    fun copyPlanarOrStridedPlane(
        bufSource: ByteBuffer,
        rowStride: Int,
        pixelStride: Int,
        cropLeft: Int,
        cropTop: Int,
        width: Int,
        height: Int,
        dst: ByteArray,
        dstOffset: Int,
        scratchRowBuf: ByteArray? = null
    ) {
        val buf = bufSource.duplicate()
        var outPos = dstOffset

        if (pixelStride == 1 && rowStride == width && cropLeft == 0 && cropTop == 0) {
            buf.position(0)
            val needed = width * height
            val avail = buf.remaining().coerceAtMost(needed)
            buf.get(dst, dstOffset, avail)
            return
        }

        val limit = buf.limit()
        if (pixelStride == 1) {
            for (r in 0 until height) {
                val srcPos = (cropTop + r) * rowStride + cropLeft
                val avail = (limit - srcPos).coerceAtLeast(0).coerceAtMost(width)
                if (avail > 0) {
                    buf.position(srcPos)
                    buf.get(dst, outPos, avail)
                }
                outPos += width
            }
        } else {
            val neededRow = ((width - 1) * pixelStride + 1).coerceAtLeast(1)
            val rowTmp = if (scratchRowBuf != null && scratchRowBuf.size >= neededRow) {
                scratchRowBuf
            } else {
                ByteArray(neededRow)
            }
            for (r in 0 until height) {
                val srcPos = (cropTop + r) * rowStride + cropLeft * pixelStride
                val readLen = rowTmp.size.coerceAtMost((limit - srcPos).coerceAtLeast(0))
                if (readLen > 0) {
                    buf.position(srcPos)
                    buf.get(rowTmp, 0, readLen)
                    for (c in 0 until width) {
                        val sIdx = c * pixelStride
                        dst[outPos + c] = if (sIdx < readLen) rowTmp[sIdx] else 0
                    }
                }
                outPos += width
            }
        }
    }
}

/**
 * High-speed zero-disk hardware video decoder built on MediaExtractor + MediaCodec.
 * Features:
 * - Bulk NV12 / NV21 semi-planar fast-path (avoids slow per-pixel UV loops)
 * - Synchronized BGR Mat ring pool (reuses native C++ frame Mats across frames)
 */
class FastVideoFrameDecoder(
    private val videoFile: File,
    private val rotationDegrees: Int
) {
    private var yuvBytes = ByteArray(0)
    private var stridedRowScratch = ByteArray(0)
    private var yuvMat: Mat? = null
    private var unrotatedScratchBgr: Mat? = null

    private val poolLock = Any()
    private val bgrPool = ArrayDeque<Mat>(8)

    fun recycleMat(mat: Mat?) {
        if (mat == null || mat.empty()) return
        synchronized(poolLock) {
            if (bgrPool.size < 8) {
                bgrPool.addLast(mat)
            } else {
                mat.release()
            }
        }
    }

    private fun obtainReusableBgrMat(h: Int, w: Int): Mat {
        synchronized(poolLock) {
            while (bgrPool.isNotEmpty()) {
                val m = bgrPool.removeFirst()
                if (m.rows() == h && m.cols() == w && m.type() == CvType.CV_8UC3) {
                    return m
                }
                m.release()
            }
        }
        return Mat(h, w, CvType.CV_8UC3)
    }

    private fun clearMatPool() {
        synchronized(poolLock) {
            while (bgrPool.isNotEmpty()) {
                bgrPool.removeFirst().release()
            }
        }
    }

    fun decodeAllFrames(
        isCancelled: () -> Boolean,
        onFrame: (frameIndex: Int, bgrMat: Mat, ptsUs: Long) -> Unit
    ): Int {
        val extractor = MediaExtractor()
        var codec: MediaCodec? = null
        var framesDecoded = 0

        try {
            extractor.setDataSource(videoFile.absolutePath)
            var trackIndex = -1
            var trackFormat: MediaFormat? = null
            for (i in 0 until extractor.trackCount) {
                val fmt = extractor.getTrackFormat(i)
                val mime = fmt.getString(MediaFormat.KEY_MIME).orEmpty()
                if (mime.startsWith("video/")) {
                    trackIndex = i
                    trackFormat = fmt
                    break
                }
            }
            if (trackIndex < 0 || trackFormat == null) {
                throw IllegalStateException("No video track found in MediaExtractor")
            }

            val mime = trackFormat.getString(MediaFormat.KEY_MIME)
                ?: throw IllegalStateException("Missing video MIME type")
            extractor.selectTrack(trackIndex)

            trackFormat.setInteger(
                MediaFormat.KEY_COLOR_FORMAT,
                MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible
            )

            codec = MediaCodec.createDecoderByType(mime)
            codec.configure(trackFormat, null, null, 0)
            codec.start()

            val bufferInfo = MediaCodec.BufferInfo()
            var inputDone = false
            var outputDone = false
            var idlePolls = 0

            while (!outputDone && !isCancelled()) {
                if (!inputDone) {
                    val inIdx = codec.dequeueInputBuffer(4_000L)
                    if (inIdx >= 0) {
                        val inBuf: ByteBuffer? = codec.getInputBuffer(inIdx)
                        val sampleSize = if (inBuf != null) extractor.readSampleData(inBuf, 0) else -1
                        if (sampleSize < 0) {
                            codec.queueInputBuffer(inIdx, 0, 0, 0L, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        } else {
                            val pts = extractor.sampleTime
                            codec.queueInputBuffer(inIdx, 0, sampleSize, pts, 0)
                            extractor.advance()
                        }
                    }
                }

                val outIdx = codec.dequeueOutputBuffer(bufferInfo, 4_000L)
                when {
                    outIdx >= 0 -> {
                        idlePolls = 0
                        val isEos = (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0
                        if (bufferInfo.size > 0 && !isCancelled()) {
                            val image = codec.getOutputImage(outIdx)
                            if (image != null) {
                                try {
                                    val bgrMat = imageYuv420ToBgrMat(image, rotationDegrees)
                                    onFrame(framesDecoded, bgrMat, bufferInfo.presentationTimeUs)
                                    framesDecoded++
                                } finally {
                                    image.close()
                                }
                            }
                        }
                        codec.releaseOutputBuffer(outIdx, false)
                        if (isEos) {
                            outputDone = true
                        }
                    }
                    outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        idlePolls = 0
                    }
                    outIdx == MediaCodec.INFO_TRY_AGAIN_LATER -> {
                        idlePolls++
                        if (idlePolls > 600 && inputDone) {
                            outputDone = true
                        }
                    }
                }
            }
            return framesDecoded
        } finally {
            yuvMat?.release()
            yuvMat = null
            unrotatedScratchBgr?.release()
            unrotatedScratchBgr = null
            yuvBytes = ByteArray(0)
            clearMatPool()
            try {
                codec?.stop()
            } catch (_: Throwable) {
            }
            try {
                codec?.release()
            } catch (_: Throwable) {
            }
            try {
                extractor.release()
            } catch (_: Throwable) {
            }
        }
    }

    private fun imageYuv420ToBgrMat(
        image: Image,
        rotationDeg: Int
    ): Mat {
        val crop = image.cropRect
        val rawW = if (crop.width() > 0) crop.width() else image.width
        val rawH = if (crop.height() > 0) crop.height() else image.height
        val w = if (rawW % 2 == 0) rawW else rawW - 1
        val h = if (rawH % 2 == 0) rawH else rawH - 1

        val ySize = w * h
        val uvSize = ySize / 4
        val totalLen = ySize + 2 * uvSize
        if (yuvBytes.size != totalLen) {
            yuvBytes = ByteArray(totalLen)
            yuvMat?.release()
            yuvMat = Mat(h + h / 2, w, CvType.CV_8UC1)
        }

        val planes = image.planes
        val neededRow = (w * 2).coerceAtLeast(16)
        if (stridedRowScratch.size < neededRow) {
            stridedRowScratch = ByteArray(neededRow)
        }
        FastYuvCopyMath.copyPlanarOrStridedPlane(
            bufSource = planes[0].buffer,
            rowStride = planes[0].rowStride,
            pixelStride = planes[0].pixelStride,
            cropLeft = crop.left,
            cropTop = crop.top,
            width = w,
            height = h,
            dst = yuvBytes,
            dstOffset = 0,
            scratchRowBuf = stridedRowScratch
        )

        val uPlane = planes[1]
        val vPlane = planes[2]
        val uBuf = uPlane.buffer
        val vBuf = vPlane.buffer
        val layout = if (uPlane.pixelStride == 2 && vPlane.pixelStride == 2 && uBuf.remaining() >= 4 && vBuf.remaining() >= 4) {
            FastYuvCopyMath.detectLayout(
                uPixelStride = uPlane.pixelStride,
                vPixelStride = vPlane.pixelStride,
                uRemaining = uBuf.remaining(),
                vRemaining = vBuf.remaining(),
                uProbe0 = uBuf.get(0),
                uProbe1 = uBuf.get(1),
                uProbe2 = uBuf.get(2),
                uProbe3 = uBuf.get(3),
                vProbe0 = vBuf.get(0),
                vProbe1 = vBuf.get(1),
                vProbe2 = vBuf.get(2),
                vProbe3 = vBuf.get(3)
            )
        } else {
            YuvFastLayout.PLANAR_I420
        }

        val normRot = ((rotationDeg % 360) + 360) % 360
        val needRotate = normRot == 90 || normRot == 180 || normRot == 270
        val outW = if (normRot == 90 || normRot == 270) h else w
        val outH = if (normRot == 90 || normRot == 270) w else h

        val cvColorCode = when (layout) {
            YuvFastLayout.NV12 -> {
                FastYuvCopyMath.copySemiPlanarUvPlane(
                    interleavedBuf = uBuf,
                    cropLeftHalf = crop.left / 2,
                    cropTopHalf = crop.top / 2,
                    uvWidth = w / 2,
                    uvHeight = h / 2,
                    rowStride = uPlane.rowStride,
                    dst = yuvBytes,
                    dstOffset = ySize
                )
                Imgproc.COLOR_YUV2BGR_NV12
            }
            YuvFastLayout.NV21 -> {
                FastYuvCopyMath.copySemiPlanarUvPlane(
                    interleavedBuf = vBuf,
                    cropLeftHalf = crop.left / 2,
                    cropTopHalf = crop.top / 2,
                    uvWidth = w / 2,
                    uvHeight = h / 2,
                    rowStride = vPlane.rowStride,
                    dst = yuvBytes,
                    dstOffset = ySize
                )
                Imgproc.COLOR_YUV2BGR_NV21
            }
            YuvFastLayout.PLANAR_I420 -> {
                FastYuvCopyMath.copyPlanarOrStridedPlane(
                    bufSource = uBuf,
                    rowStride = uPlane.rowStride,
                    pixelStride = uPlane.pixelStride,
                    cropLeft = crop.left / 2,
                    cropTop = crop.top / 2,
                    width = w / 2,
                    height = h / 2,
                    dst = yuvBytes,
                    dstOffset = ySize,
                    scratchRowBuf = stridedRowScratch
                )
                FastYuvCopyMath.copyPlanarOrStridedPlane(
                    bufSource = vBuf,
                    rowStride = vPlane.rowStride,
                    pixelStride = vPlane.pixelStride,
                    cropLeft = crop.left / 2,
                    cropTop = crop.top / 2,
                    width = w / 2,
                    height = h / 2,
                    dst = yuvBytes,
                    dstOffset = ySize + uvSize,
                    scratchRowBuf = stridedRowScratch
                )
                Imgproc.COLOR_YUV2BGR_I420
            }
        }

        val localYuv = yuvMat ?: Mat(h + h / 2, w, CvType.CV_8UC1).also { yuvMat = it }
        localYuv.put(0, 0, yuvBytes)

        if (!needRotate) {
            val outBgr = obtainReusableBgrMat(h, w)
            Imgproc.cvtColor(localYuv, outBgr, cvColorCode)
            return outBgr
        }

        val unrotatedBgr = unrotatedScratchBgr?.takeIf { it.rows() == h && it.cols() == w }
            ?: Mat(h, w, CvType.CV_8UC3).also {
                unrotatedScratchBgr?.release()
                unrotatedScratchBgr = it
            }
        Imgproc.cvtColor(localYuv, unrotatedBgr, cvColorCode)
        val rotated = obtainReusableBgrMat(outH, outW)
        val rotCode = when (normRot) {
            90 -> Core.ROTATE_90_CLOCKWISE
            180 -> Core.ROTATE_180
            else -> Core.ROTATE_90_COUNTERCLOCKWISE
        }
        Core.rotate(unrotatedBgr, rotated, rotCode)
        return rotated
    }
}
