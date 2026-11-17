package com.example.engine

import android.graphics.Bitmap
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaCodecList
import android.media.MediaFormat
import android.media.MediaMuxer
import android.os.Build
import android.view.Surface
import java.io.File
import kotlin.math.roundToInt

/**
 * Streaming MediaCodec + MediaMuxer video encoder using COLOR_FormatSurface.
 * Attempts preferred codec (H.265 hardware -> H.264 hardware -> software MediaCodec fallback)
 * while maintaining exact source FPS via deterministic presentation timestamps.
 */
class HardwareVideoEncoder(
    private val outputFile: File,
    private val width: Int,
    private val height: Int,
    private val fps: Double,
    private val bitrateBps: Int,
    private val preferH265: Boolean
) {
    private var mediaCodec: MediaCodec? = null
    private var inputSurface: Surface? = null
    private var mediaMuxer: MediaMuxer? = null
    private var videoTrackIndex: Int = -1
    private var muxerStarted: Boolean = false
    private val bufferInfo = MediaCodec.BufferInfo()

    var activeCodecLabel: String = "H.264"
        private set

    fun start() {
        val candidates = buildList {
            if (preferH265) {
                add(MediaFormat.MIMETYPE_VIDEO_HEVC to true)
            }
            add(MediaFormat.MIMETYPE_VIDEO_AVC to true)
            if (preferH265) {
                add(MediaFormat.MIMETYPE_VIDEO_HEVC to false)
            }
            add(MediaFormat.MIMETYPE_VIDEO_AVC to false)
        }

        var lastError: Throwable? = null
        for ((mime, requireHardware) in candidates) {
            try {
                if (tryConfigure(mime, requireHardware)) {
                    activeCodecLabel = if (mime == MediaFormat.MIMETYPE_VIDEO_HEVC) {
                        if (requireHardware) "H.265 (Hardware)" else "H.265 (Software)"
                    } else {
                        if (requireHardware) "H.264 (Hardware)" else "H.264 (Software)"
                    }
                    return
                }
            } catch (t: Throwable) {
                lastError = t
                releaseInternal()
            }
        }
        throw IllegalStateException("Unable to initialize video encoder for ${width}x${height}: ${lastError?.message}", lastError)
    }

    private var outputFrameCounter: Long = 0L

    private fun tryConfigure(mime: String, requireHardware: Boolean): Boolean {
        val codecInfo = findMatchingEncoder(mime, requireHardware) ?: return false
        val format = MediaFormat.createVideoFormat(mime, width, height).apply {
            setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
            setInteger(MediaFormat.KEY_BIT_RATE, bitrateBps)
            setInteger(MediaFormat.KEY_FRAME_RATE, fps.roundToInt().coerceIn(15, 60))
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                setInteger(MediaFormat.KEY_MAX_B_FRAMES, 0)
            }
        }

        val codec = MediaCodec.createByCodecName(codecInfo.name)
        codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        val surface = codec.createInputSurface()
        codec.start()

        if (outputFile.exists()) outputFile.delete()
        val muxer = MediaMuxer(outputFile.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)

        mediaCodec = codec
        inputSurface = surface
        mediaMuxer = muxer
        videoTrackIndex = -1
        muxerStarted = false
        outputFrameCounter = 0L
        return true
    }

    fun encodeFrame(bitmap: Bitmap, frameIndex: Int) {
        val surface = inputSurface ?: throw IllegalStateException("Encoder surface not initialized")
        val canvas = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            surface.lockHardwareCanvas()
        } else {
            surface.lockCanvas(null)
        }
        try {
            canvas.drawBitmap(bitmap, 0f, 0f, null)
        } finally {
            surface.unlockCanvasAndPost(canvas)
        }
        drainEncoder(endOfStream = false)
    }

    fun finish() {
        val codec = mediaCodec ?: return
        try {
            codec.signalEndOfInputStream()
            drainEncoder(endOfStream = true)
        } finally {
            releaseInternal()
        }
    }

    fun abort() {
        releaseInternal()
        if (outputFile.exists()) {
            outputFile.delete()
        }
    }

    private fun drainEncoder(endOfStream: Boolean) {
        val codec = mediaCodec ?: return
        val muxer = mediaMuxer ?: return
        val timeoutUs = if (endOfStream) 15_000L else 5_000L
        val safeFps = fps.coerceIn(10.0, 120.0)

        while (true) {
            val outIndex = codec.dequeueOutputBuffer(bufferInfo, timeoutUs)
            when {
                outIndex == MediaCodec.INFO_TRY_AGAIN_LATER -> {
                    if (!endOfStream) break
                }
                outIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    if (muxerStarted) throw IllegalStateException("Format changed twice")
                    val newFormat = codec.outputFormat
                    videoTrackIndex = muxer.addTrack(newFormat)
                    muxer.start()
                    muxerStarted = true
                }
                outIndex >= 0 -> {
                    val encodedData = codec.getOutputBuffer(outIndex)
                    if ((bufferInfo.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG) != 0) {
                        bufferInfo.size = 0
                    }
                    if (bufferInfo.size > 0 && muxerStarted && encodedData != null) {
                        encodedData.position(bufferInfo.offset)
                        encodedData.limit(bufferInfo.offset + bufferInfo.size)
                        bufferInfo.presentationTimeUs = ((outputFrameCounter * 1_000_000.0) / safeFps).toLong()
                        outputFrameCounter++
                        muxer.writeSampleData(videoTrackIndex, encodedData, bufferInfo)
                    }
                    codec.releaseOutputBuffer(outIndex, false)
                    if ((bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                        break
                    }
                }
            }
        }
    }

    private fun findMatchingEncoder(mime: String, requireHardware: Boolean): MediaCodecInfo? {
        val list = MediaCodecList(MediaCodecList.REGULAR_CODECS)
        return list.codecInfos.firstOrNull { info ->
            if (!info.isEncoder) return@firstOrNull false
            if (!info.supportedTypes.any { it.equals(mime, ignoreCase = true) }) return@firstOrNull false
            val isHw = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                info.isHardwareAccelerated
            } else {
                val n = info.name.lowercase()
                !n.startsWith("omx.google.") && !n.startsWith("c2.android.")
            }
            if (requireHardware) isHw else !isHw
        }
    }

    private fun releaseInternal() {
        try {
            inputSurface?.release()
        } catch (_: Exception) {
        }
        inputSurface = null

        try {
            mediaCodec?.stop()
        } catch (_: Exception) {
        }
        try {
            mediaCodec?.release()
        } catch (_: Exception) {
        }
        mediaCodec = null

        try {
            if (muxerStarted) mediaMuxer?.stop()
        } catch (_: Exception) {
        }
        try {
            mediaMuxer?.release()
        } catch (_: Exception) {
        }
        mediaMuxer = null
        muxerStarted = false
        videoTrackIndex = -1
    }
}
