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

/**
 * High-speed zero-disk hardware video decoder built on MediaExtractor + MediaCodec.
 * Decodes directly into YUV_420_888 Images and converts in-memory to OpenCV BGR Mat,
 * eliminating disk JPEG/PNG write+read overhead.
 */
class FastVideoFrameDecoder(
    private val videoFile: File,
    private val rotationDegrees: Int
) {
    private var i420Bytes = ByteArray(0)
    private var yuvMat: Mat? = null

    /**
     * Streams decoded BGR Mats sequentially into [onFrame].
     * Caller MUST invoke [releaseMat] or call `.release()` immediately inside [onFrame] if copied,
     * though each yielded [Mat] is freshly created so ownership transfers cleanly to caller.
     *
     * Returns total number of frames decoded, or throws if MediaCodec cannot decode cleanly
     * so caller can safely fall back to FFmpegKit.
     */
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
                    val inIdx = codec.dequeueInputBuffer(8_000L)
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

                val outIdx = codec.dequeueOutputBuffer(bufferInfo, 8_000L)
                when {
                    outIdx >= 0 -> {
                        idlePolls = 0
                        val isEos = (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0
                        if (bufferInfo.size > 0 && !isCancelled()) {
                            val image = codec.getOutputImage(outIdx)
                            if (image != null) {
                                try {
                                    val outFormat = codec.outputFormat
                                    val bgrMat = imageYuv420ToBgrMat(image, outFormat, rotationDegrees)
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
                            // Guard against rare decoder EOS hang once input has completed
                            outputDone = true
                        }
                    }
                }
            }
            return framesDecoded
        } finally {
            yuvMat?.release()
            yuvMat = null
            i420Bytes = ByteArray(0)
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
        outputFormat: MediaFormat,
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
        if (i420Bytes.size != totalLen) {
            i420Bytes = ByteArray(totalLen)
            yuvMat?.release()
            yuvMat = Mat(h + h / 2, w, CvType.CV_8UC1)
        }

        val planes = image.planes
        copyPlaneToI420(
            plane = planes[0],
            cropLeft = crop.left,
            cropTop = crop.top,
            width = w,
            height = h,
            dst = i420Bytes,
            dstOffset = 0
        )
        copyPlaneToI420(
            plane = planes[1],
            cropLeft = crop.left / 2,
            cropTop = crop.top / 2,
            width = w / 2,
            height = h / 2,
            dst = i420Bytes,
            dstOffset = ySize
        )
        copyPlaneToI420(
            plane = planes[2],
            cropLeft = crop.left / 2,
            cropTop = crop.top / 2,
            width = w / 2,
            height = h / 2,
            dst = i420Bytes,
            dstOffset = ySize + uvSize
        )

        val localYuv = yuvMat ?: Mat(h + h / 2, w, CvType.CV_8UC1).also { yuvMat = it }
        localYuv.put(0, 0, i420Bytes)
        val bgr = Mat(h, w, CvType.CV_8UC3)
        Imgproc.cvtColor(localYuv, bgr, Imgproc.COLOR_YUV2BGR_I420)

        val normRot = ((rotationDeg % 360) + 360) % 360
        if (normRot == 90 || normRot == 180 || normRot == 270) {
            val rotated = Mat()
            val code = when (normRot) {
                90 -> Core.ROTATE_90_CLOCKWISE
                180 -> Core.ROTATE_180
                else -> Core.ROTATE_90_COUNTERCLOCKWISE
            }
            Core.rotate(bgr, rotated, code)
            bgr.release()
            return rotated
        }
        return bgr
    }

    private fun copyPlaneToI420(
        plane: Image.Plane,
        cropLeft: Int,
        cropTop: Int,
        width: Int,
        height: Int,
        dst: ByteArray,
        dstOffset: Int
    ) {
        val buf = plane.buffer.duplicate()
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
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
                if (srcPos + width <= limit) {
                    buf.position(srcPos)
                    buf.get(dst, outPos, width)
                }
                outPos += width
            }
        } else {
            val rowTmp = ByteArray((width - 1) * pixelStride + 1)
            for (r in 0 until height) {
                val srcPos = (cropTop + r) * rowStride + cropLeft * pixelStride
                val readLen = rowTmp.size.coerceAtMost((limit - srcPos).coerceAtLeast(0))
                if (readLen > 0) {
                    buf.position(srcPos)
                    buf.get(rowTmp, 0, readLen)
                    var cIdx = 0
                    for (c in 0 until width) {
                        val sIdx = c * pixelStride
                        dst[outPos + c] = if (sIdx < readLen) rowTmp[sIdx] else 0
                        cIdx++
                    }
                }
                outPos += width
            }
        }
    }
}
