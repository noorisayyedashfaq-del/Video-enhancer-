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
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.Future
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.roundToInt

/**
 * Streaming MediaCodec + MediaMuxer video encoder using COLOR_FormatSurface.
 * Supports double-buffered asynchronous surface encoding via [enqueueFrameAsync]
 * so hardware video encoding overlaps concurrently with the next frame's AI inference.
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

    private var encodeExecutor: ExecutorService? = null
    private var pendingEncodeFuture: Future<*>? = null
    private val asyncError = AtomicReference<Throwable?>(null)

    var activeCodecLabel: String = "H.264"
        private set
    var activeProfileSummary: String = "Default"
        private set
    var keyframeIntervalSec: Int = 3
        private set

    private var outputFrameCounter: Long = 0L
    private var lastMuxedPtsUs: Long = -1L
    private var lastQueuedPtsUs: Long = -1L
    private val ptsLock = Any()
    private val ptsQueueUs = ArrayDeque<Long>()
    private val totalEncodeTimeMsAtomic = AtomicLong(0L)
    val totalEncodeTimeMs: Long get() = totalEncodeTimeMsAtomic.get()

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
                        if (requireHardware) "H.264 (Hardware) • $activeProfileSummary" else "H.264 (Software) • $activeProfileSummary"
                    }
                    encodeExecutor = Executors.newSingleThreadExecutor { r ->
                        Thread(r, "ClearKit-HwEncoder").apply { isDaemon = true }
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

    private fun tryConfigure(mime: String, requireHardware: Boolean): Boolean {
        val codecInfo = findMatchingEncoder(mime, requireHardware) ?: return false
        val codecCaps = runCatching { codecInfo.getCapabilitiesForType(mime) }.getOrNull()
        val encCaps = codecCaps?.encoderCapabilities
        val highestHighLevel = if (mime == MediaFormat.MIMETYPE_VIDEO_AVC && codecCaps != null) {
            codecCaps.profileLevels
                ?.filter { it.profile == MediaCodecInfo.CodecProfileLevel.AVCProfileHigh }
                ?.maxOfOrNull { it.level }
        } else {
            null
        }

        fun buildFormat(useHighProfile: Boolean, iFrameIntervalSec: Int): MediaFormat {
            return MediaFormat.createVideoFormat(mime, width, height).apply {
                setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
                setInteger(MediaFormat.KEY_BIT_RATE, bitrateBps)
                setInteger(MediaFormat.KEY_FRAME_RATE, fps.roundToInt().coerceIn(15, 60))
                setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, iFrameIntervalSec)
                if (encCaps?.isBitrateModeSupported(MediaCodecInfo.EncoderCapabilities.BITRATE_MODE_VBR) == true) {
                    setInteger(MediaFormat.KEY_BITRATE_MODE, MediaCodecInfo.EncoderCapabilities.BITRATE_MODE_VBR)
                }
                if (useHighProfile && highestHighLevel != null && mime == MediaFormat.MIMETYPE_VIDEO_AVC) {
                    setInteger(MediaFormat.KEY_PROFILE, MediaCodecInfo.CodecProfileLevel.AVCProfileHigh)
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && highestHighLevel > 0) {
                        setInteger(MediaFormat.KEY_LEVEL, highestHighLevel)
                    }
                }
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                    setInteger(MediaFormat.KEY_PRIORITY, 0)
                    setInteger(MediaFormat.KEY_OPERATING_RATE, 120)
                }
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    // Keep B-frames disabled on Surface input so lockHardwareCanvas() PTS stays strictly monotonic
                    setInteger(MediaFormat.KEY_MAX_B_FRAMES, 0)
                }
            }
        }

        val configs = listOf(
            (highestHighLevel != null) to 3,
            false to 3,
            false to 1
        ).distinct()

        var configuredCodec: MediaCodec? = null
        var chosenProfileLabel = "Default • GOP 1s"
        var chosenGop = 1
        var lastConfigError: Throwable? = null

        for ((tryHigh, gopSec) in configs) {
            val attemptCodec = MediaCodec.createByCodecName(codecInfo.name)
            try {
                attemptCodec.configure(buildFormat(tryHigh, gopSec), null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
                configuredCodec = attemptCodec
                chosenGop = gopSec
                chosenProfileLabel = if (tryHigh) {
                    "High Profile CABAC • GOP ${gopSec}s"
                } else {
                    "VBR • GOP ${gopSec}s"
                }
                break
            } catch (t: Throwable) {
                lastConfigError = t
                runCatching { attemptCodec.release() }
            }
        }

        val codec = configuredCodec ?: throw (lastConfigError ?: IllegalStateException("Codec configure failed"))
        val surface = codec.createInputSurface()
        codec.start()

        if (outputFile.exists()) outputFile.delete()
        val muxer = MediaMuxer(outputFile.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)

        mediaCodec = codec
        inputSurface = surface
        mediaMuxer = muxer
        activeProfileSummary = chosenProfileLabel
        keyframeIntervalSec = chosenGop
        videoTrackIndex = -1
        muxerStarted = false
        outputFrameCounter = 0L
        lastMuxedPtsUs = -1L
        lastQueuedPtsUs = -1L
        synchronized(ptsLock) { ptsQueueUs.clear() }
        totalEncodeTimeMsAtomic.set(0L)
        asyncError.set(null)
        return true
    }

    /**
     * Overlaps encoding of [bitmap] (double-buffered ping-pong bitmap) with AI processing of the next frame.
     * Waits only if the previous asynchronous encode has not yet finished drawing to the encoder Surface.
     */
    fun enqueueFrameAsync(bitmap: Bitmap, frameIndex: Int) {
        val derivedPts = ((frameIndex.toLong() * 1_000_000.0) / fps.coerceIn(10.0, 120.0)).toLong()
        enqueueFrameAsyncWithPts(bitmap, derivedPts)
    }

    fun enqueueFrameAsyncWithPts(bitmap: Bitmap, ptsUs: Long) {
        awaitPendingEncode()
        val exec = encodeExecutor
        if (exec == null || exec.isShutdown) {
            encodeFrameInternal(bitmap, ptsUs)
            return
        }
        pendingEncodeFuture = exec.submit {
            try {
                encodeFrameInternal(bitmap, ptsUs)
            } catch (t: Throwable) {
                asyncError.compareAndSet(null, t)
            }
        }
    }

    fun awaitPendingEncode() {
        val fut = pendingEncodeFuture
        pendingEncodeFuture = null
        if (fut != null) {
            try {
                fut.get()
            } catch (t: Throwable) {
                asyncError.compareAndSet(null, t)
            }
        }
        asyncError.getAndSet(null)?.let { throw it }
    }

    fun encodeFrame(bitmap: Bitmap, frameIndex: Int) {
        val derivedPts = ((frameIndex.toLong() * 1_000_000.0) / fps.coerceIn(10.0, 120.0)).toLong()
        encodeFrameWithPts(bitmap, derivedPts)
    }

    fun encodeFrameWithPts(bitmap: Bitmap, ptsUs: Long) {
        awaitPendingEncode()
        encodeFrameInternal(bitmap, ptsUs)
    }

    private fun encodeFrameInternal(bitmap: Bitmap, ptsUs: Long) {
        val t0 = System.currentTimeMillis()
        synchronized(ptsLock) {
            val safePts = if (ptsUs >= 0L) {
                if (lastQueuedPtsUs < 0L) ptsUs else maxOf(ptsUs, lastQueuedPtsUs + 1_000L)
            } else {
                if (lastQueuedPtsUs < 0L) 0L else lastQueuedPtsUs + (1_000_000.0 / fps.coerceIn(10.0, 120.0)).toLong()
            }
            lastQueuedPtsUs = safePts
            ptsQueueUs.addLast(safePts)
        }
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
        totalEncodeTimeMsAtomic.addAndGet(System.currentTimeMillis() - t0)
    }

    fun finish() {
        val t0 = System.currentTimeMillis()
        val codec = mediaCodec ?: return
        try {
            awaitPendingEncode()
            codec.signalEndOfInputStream()
            drainEncoder(endOfStream = true)
            if (outputFrameCounter == 0L) {
                throw IllegalStateException("Encoder produced 0 output frames for ${width}x${height} ($activeCodecLabel)")
            }
        } finally {
            totalEncodeTimeMsAtomic.addAndGet(System.currentTimeMillis() - t0)
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
        val timeoutUs = if (endOfStream) 10_000L else 0L
        val safeFps = fps.coerceIn(10.0, 120.0)
        val eosDeadlineNs = if (endOfStream) System.nanoTime() + 10_000_000_000L else Long.MAX_VALUE
        var eosTryAgainCount = 0

        while (true) {
            val outIndex = codec.dequeueOutputBuffer(bufferInfo, timeoutUs)
            when {
                outIndex == MediaCodec.INFO_TRY_AGAIN_LATER -> {
                    if (!endOfStream) break
                    eosTryAgainCount++
                    if (eosTryAgainCount > 300 || System.nanoTime() > eosDeadlineNs) {
                        break
                    }
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
                        val poppedPts = synchronized(ptsLock) {
                            if (ptsQueueUs.isNotEmpty()) ptsQueueUs.removeFirst() else -1L
                        }
                        val basePts = if (poppedPts >= 0L) {
                            poppedPts
                        } else {
                            ((outputFrameCounter * 1_000_000.0) / safeFps).toLong()
                        }
                        val monoPts = if (lastMuxedPtsUs < 0L) basePts else maxOf(basePts, lastMuxedPtsUs + 1_000L)
                        lastMuxedPtsUs = monoPts
                        bufferInfo.presentationTimeUs = monoPts
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
        pendingEncodeFuture?.cancel(true)
        pendingEncodeFuture = null
        encodeExecutor?.shutdownNow()
        encodeExecutor = null

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
