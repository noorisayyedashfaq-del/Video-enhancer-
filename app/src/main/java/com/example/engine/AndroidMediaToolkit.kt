package com.example.engine

import android.graphics.Bitmap
import android.media.AudioFormat
import android.media.Image
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaCodecList
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import android.media.MediaMuxer
import android.net.Uri
import android.os.Build
import com.example.model.VideoMetadata
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.roundToInt
import kotlin.math.roundToLong

/**
 * Pure JVM-testable helpers for Android MediaExtractor/MediaCodec probing,
 * FPS inference from packet timestamps, MP4 audio compatibility, and PCM channel downmixing.
 */
object MediaCodecProbeMath {

    fun mapVideoMimeToCodecLabel(mime: String?): String {
        val m = mime?.lowercase()?.trim().orEmpty()
        return when {
            m.contains("avc") || m.contains("h264") -> "h264"
            m.contains("hevc") || m.contains("h265") -> "hevc"
            m.contains("vp9") -> "vp9"
            m.contains("vp8") -> "vp8"
            m.contains("av01") || m.contains("av1") -> "av1"
            m.contains("mp4v") -> "mpeg4"
            m.contains("3gpp") -> "h263"
            m.startsWith("video/") -> m.removePrefix("video/")
            else -> "unknown"
        }
    }

    fun mapAudioMimeToCodecLabel(mime: String?): String? {
        val m = mime?.lowercase()?.trim() ?: return null
        if (m.isBlank()) return null
        return when {
            m == MediaFormat.MIMETYPE_AUDIO_AAC || m.contains("mp4a") -> "aac"
            m == MediaFormat.MIMETYPE_AUDIO_MPEG || m.contains("mp3") -> "mp3"
            m == MediaFormat.MIMETYPE_AUDIO_OPUS -> "opus"
            m == MediaFormat.MIMETYPE_AUDIO_VORBIS -> "vorbis"
            m == MediaFormat.MIMETYPE_AUDIO_FLAC -> "flac"
            m == MediaFormat.MIMETYPE_AUDIO_RAW -> "pcm"
            m == MediaFormat.MIMETYPE_AUDIO_AMR_NB -> "amr_nb"
            m == MediaFormat.MIMETYPE_AUDIO_AMR_WB -> "amr_wb"
            m.startsWith("audio/") -> m.removePrefix("audio/")
            else -> m
        }
    }

    fun isMp4PassthroughCompatibleAudioMime(mime: String?, hasCsd0: Boolean): Boolean {
        val m = mime?.lowercase()?.trim().orEmpty()
        return when (m) {
            MediaFormat.MIMETYPE_AUDIO_AAC -> hasCsd0
            MediaFormat.MIMETYPE_AUDIO_AMR_NB,
            MediaFormat.MIMETYPE_AUDIO_AMR_WB -> true
            else -> false
        }
    }

    fun estimateFpsFromPtsList(rawPtsUs: List<Long>): Double? {
        if (rawPtsUs.size < 2) return null
        val sorted = rawPtsUs.filter { it >= 0L }.sorted()
        if (sorted.size < 2) return null
        val deltas = ArrayList<Long>(sorted.size - 1)
        for (i in 1 until sorted.size) {
            val d = sorted[i] - sorted[i - 1]
            if (d in 4_000L..250_000L) {
                deltas.add(d)
            }
        }
        if (deltas.isEmpty()) return null
        deltas.sort()
        val medianDeltaUs = deltas[deltas.size / 2].toDouble()
        val fps = 1_000_000.0 / medianDeltaUs
        return if (fps in 5.0..120.0) fps else null
    }

    /**
     * Downmixes interleaved 16-bit signed PCM frames with >2 channels into stereo (2 channels)
     * by averaging left/front channels vs right/surround channels safely within Short bounds.
     */
    fun downmixInterleavedPcm16ToStereo(
        srcBytes: ByteArray,
        srcLen: Int,
        srcChannels: Int
    ): ByteArray {
        if (srcChannels <= 2 || srcLen <= 0) {
            return srcBytes.copyOf(srcLen)
        }
        val bytesPerFrame = srcChannels * 2
        val frameCount = srcLen / bytesPerFrame
        if (frameCount <= 0) return ByteArray(0)

        val inBuf = ByteBuffer.wrap(srcBytes, 0, frameCount * bytesPerFrame).order(ByteOrder.LITTLE_ENDIAN)
        val outBytes = ByteArray(frameCount * 4)
        val outBuf = ByteBuffer.wrap(outBytes).order(ByteOrder.LITTLE_ENDIAN)

        for (f in 0 until frameCount) {
            val ch0 = inBuf.short.toInt()
            val ch1 = inBuf.short.toInt()
            var centerOrMono = 0
            for (c in 2 until srcChannels) {
                val sample = inBuf.short.toInt()
                if (c == 2) centerOrMono = sample
            }
            val l = (ch0 + (centerOrMono * 7) / 10).coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt())
            val r = (ch1 + (centerOrMono * 7) / 10).coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt())
            outBuf.putShort(l.toShort())
            outBuf.putShort(r.toShort())
        }
        return outBytes
    }
}

/**
 * Pure Android Platform SDK (Apache-2.0) replacement for all media probing, fallback frame
 * decoding, representative frame seeking, MP4 muxing, and AAC-LC audio transcoding.
 * Uses MediaExtractor, MediaCodec, MediaMuxer, MediaMetadataRetriever, and OpenCV exclusively.
 */
object AndroidMediaToolkit {

    fun isMediaCodecSubsystemReady(): Boolean {
        return try {
            val infos = MediaCodecList(MediaCodecList.REGULAR_CODECS).codecInfos
            val hasDecoder = infos.any { !it.isEncoder && it.supportedTypes.any { t -> t.startsWith("video/") } }
            val hasEncoder = infos.any { it.isEncoder && it.supportedTypes.any { t -> t.startsWith("video/") } }
            hasDecoder && hasEncoder
        } catch (_: Throwable) {
            false
        }
    }

    fun probeVideoFile(uri: Uri, file: File, displayName: String): VideoMetadata {
        val retriever = MediaMetadataRetriever()
        var mmrWidth = 0
        var mmrHeight = 0
        var mmrRot = 0
        var mmrDurMs = 0L
        var mmrBitrate = 0L
        var mmrCaptureFps: Double? = null
        var mmrFrameCount: Int? = null
        try {
            retriever.setDataSource(file.absolutePath)
            mmrWidth = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toIntOrNull() ?: 0
            mmrHeight = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toIntOrNull() ?: 0
            mmrRot = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)?.toIntOrNull() ?: 0
            mmrDurMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
            mmrBitrate = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_BITRATE)?.toLongOrNull() ?: 0L
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                mmrCaptureFps = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE)
                    ?.toDoubleOrNull()
                    ?.takeIf { it in 5.0..120.0 }
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                mmrFrameCount = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)
                    ?.toIntOrNull()
                    ?.takeIf { it > 0 }
            }
        } catch (_: Throwable) {
        } finally {
            runCatching { retriever.release() }
        }

        var videoTrackIndex = -1
        var videoFormat: MediaFormat? = null
        var audioFormat: MediaFormat? = null
        var sampledFps: Double? = null

        val extractor = MediaExtractor()
        try {
            extractor.setDataSource(file.absolutePath)
            for (i in 0 until extractor.trackCount) {
                val fmt = extractor.getTrackFormat(i)
                val mime = fmt.getString(MediaFormat.KEY_MIME).orEmpty()
                if (mime.startsWith("video/") && videoFormat == null) {
                    videoTrackIndex = i
                    videoFormat = fmt
                } else if (mime.startsWith("audio/") && audioFormat == null) {
                    audioFormat = fmt
                }
            }

            val formatFps = videoFormat?.getFrameRateOrNull()
            if (formatFps == null && mmrCaptureFps == null && videoTrackIndex >= 0) {
                extractor.selectTrack(videoTrackIndex)
                val ptsList = ArrayList<Long>(32)
                while (ptsList.size < 32) {
                    val pts = extractor.sampleTime
                    if (pts < 0L) break
                    ptsList.add(pts)
                    if (!extractor.advance()) break
                }
                sampledFps = MediaCodecProbeMath.estimateFpsFromPtsList(ptsList)
            }
        } catch (_: Throwable) {
        } finally {
            runCatching { extractor.release() }
        }

        val fmtWidth = videoFormat?.getIntOrNull(MediaFormat.KEY_WIDTH) ?: 0
        val fmtHeight = videoFormat?.getIntOrNull(MediaFormat.KEY_HEIGHT) ?: 0
        val rawW = max(mmrWidth, fmtWidth)
        val rawH = max(mmrHeight, fmtHeight)
        if (rawW <= 0 || rawH <= 0) {
            throw IllegalArgumentException("Unsupported or unreadable video file.")
        }

        val fmtRot = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && mmrRot == 0) {
            videoFormat?.getIntOrNull(MediaFormat.KEY_ROTATION) ?: 0
        } else {
            mmrRot
        }
        val rotation = ((fmtRot % 360) + 360) % 360
        val swapDims = rotation == 90 || rotation == 270
        val displayW = if (swapDims) rawH else rawW
        val displayH = if (swapDims) rawW else rawH

        val fmtDurMs = videoFormat?.getLongOrNull(MediaFormat.KEY_DURATION)?.div(1000L) ?: 0L
        val durationMs = max(mmrDurMs, fmtDurMs).coerceAtLeast(100L)

        val avgFpsFromCount = mmrFrameCount?.let { count ->
            val f = count / (durationMs / 1000.0)
            if (f in 5.0..120.0) f else null
        }

        val fps = videoFormat?.getFrameRateOrNull()
            ?: mmrCaptureFps
            ?: sampledFps
            ?: avgFpsFromCount
            ?: 30.0

        val estFrames = mmrFrameCount ?: ((durationMs / 1000.0) * fps).roundToInt().coerceAtLeast(1)

        val fmtBitrate = videoFormat?.getIntOrNull(MediaFormat.KEY_BIT_RATE)?.toLong() ?: 0L
        val bitrate = when {
            mmrBitrate > 0L -> mmrBitrate
            fmtBitrate > 0L -> fmtBitrate
            else -> (file.length() * 8_000L) / durationMs
        }

        val videoMime = videoFormat?.getString(MediaFormat.KEY_MIME)
        val audioMime = audioFormat?.getString(MediaFormat.KEY_MIME)

        return VideoMetadata(
            uri = uri,
            displayName = displayName,
            width = rawW,
            height = rawH,
            rotationDegrees = rotation,
            displayWidth = displayW,
            displayHeight = displayH,
            fps = fps,
            durationMs = durationMs,
            estimatedFrameCount = estFrames,
            bitrateBps = bitrate,
            videoCodec = MediaCodecProbeMath.mapVideoMimeToCodecLabel(videoMime),
            hasAudio = audioFormat != null,
            audioCodec = MediaCodecProbeMath.mapAudioMimeToCodecLabel(audioMime),
            audioSampleRate = audioFormat?.getIntOrNull(MediaFormat.KEY_SAMPLE_RATE),
            audioChannels = audioFormat?.getIntOrNull(MediaFormat.KEY_CHANNEL_COUNT),
            fileSizeBytes = file.length()
        )
    }

    /**
     * Targeted keyframe-seek frame sampler using MediaExtractor + MediaCodec in memory.
     * Used when MediaMetadataRetriever cannot extract thumbnails from a container.
     * Caller owns and must release all returned Mat instances.
     */
    fun sampleRepresentativeFramesViaMediaCodec(
        videoFile: File,
        durationMs: Long,
        rotationDegrees: Int,
        fractions: List<Double> = listOf(0.15, 0.50, 0.85),
        maxAnalyticWidth: Int = 320
    ): List<Mat> {
        val extractor = MediaExtractor()
        var codec: MediaCodec? = null
        val sampledMats = ArrayList<Mat>(fractions.size)
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
            if (trackIndex < 0 || trackFormat == null) return emptyList()
            val mime = trackFormat.getString(MediaFormat.KEY_MIME) ?: return emptyList()
            extractor.selectTrack(trackIndex)

            codec = createDecoderPreferringCompatible(mime, preferSoftware = false)
            codec.configure(trackFormat, null, null, 0)
            codec.start()

            val durUs = durationMs.coerceAtLeast(300L) * 1000L
            val info = MediaCodec.BufferInfo()
            val normRot = ((rotationDegrees % 360) + 360) % 360

            for (frac in fractions) {
                val targetUs = (durUs * frac).toLong().coerceAtLeast(0L)
                extractor.seekTo(targetUs, MediaExtractor.SEEK_TO_CLOSEST_SYNC)
                codec.flush()

                var decodedOne = false
                var inputDone = false
                var attempts = 0
                while (!decodedOne && attempts < 80) {
                    attempts++
                    if (!inputDone) {
                        val inIdx = codec.dequeueInputBuffer(5_000L)
                        if (inIdx >= 0) {
                            val buf = codec.getInputBuffer(inIdx)
                            val size = if (buf != null) extractor.readSampleData(buf, 0) else -1
                            if (size < 0) {
                                codec.queueInputBuffer(inIdx, 0, 0, 0L, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                                inputDone = true
                            } else {
                                codec.queueInputBuffer(inIdx, 0, size, extractor.sampleTime, 0)
                                extractor.advance()
                            }
                        }
                    }

                    val outIdx = codec.dequeueOutputBuffer(info, 8_000L)
                    if (outIdx >= 0) {
                        if (info.size > 0) {
                            val bgr = decodeOutputBufferToBgrMat(codec, outIdx, normRot)
                            if (bgr != null && !bgr.empty()) {
                                val resized = resizeForMaxWidth(bgr, maxAnalyticWidth)
                                if (resized !== bgr) bgr.release()
                                sampledMats.add(resized)
                                decodedOne = true
                            } else {
                                bgr?.release()
                            }
                        }
                        codec.releaseOutputBuffer(outIdx, false)
                        if ((info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) break
                    }
                }
            }
            return sampledMats
        } catch (_: Throwable) {
            sampledMats.forEach { it.release() }
            return emptyList()
        } finally {
            runCatching { codec?.stop() }
            runCatching { codec?.release() }
            runCatching { extractor.release() }
        }
    }

    /**
     * Multi-tier pure Android fallback frame decoder when zero-copy COLOR_FormatYUV420Flexible fails:
     * 1. Software/native-layout MediaCodec + MediaExtractor (supports both Image and raw output ByteBuffer).
     * 2. MediaMetadataRetriever stepped frame extraction as an ultimate safety net.
     */
    fun decodeVideoFallbackFrames(
        videoFile: File,
        rotationDegrees: Int,
        durationMs: Long,
        safeFps: Double,
        isCancelled: () -> Boolean,
        onFrame: (frameIndex: Int, bgrMat: Mat, ptsUs: Long) -> Unit,
        logger: EngineDiagnosticsLogger? = null
    ): Pair<Int, String> {
        val normRot = ((rotationDegrees % 360) + 360) % 360

        // Tier 2: Software / unconstrained-format MediaCodec decoder
        val swCount = runCatching {
            decodeViaUnconstrainedMediaCodec(
                videoFile = videoFile,
                normRot = normRot,
                preferSoftware = true,
                isCancelled = isCancelled,
                onFrame = onFrame
            )
        }.onFailure { err ->
            logger?.warn("Software MediaCodec fallback failed (${err.message}); trying MediaMetadataRetriever")
        }.getOrDefault(0)

        if (swCount > 0 || isCancelled()) {
            return swCount to "MediaCodec Software Fallback"
        }

        // Tier 3: MediaMetadataRetriever stepped frame extraction
        val mmrCount = decodeViaMediaMetadataRetriever(
            videoFile = videoFile,
            durationMs = durationMs,
            safeFps = safeFps,
            isCancelled = isCancelled,
            onFrame = onFrame
        )
        return mmrCount to "MediaMetadataRetriever Fallback"
    }

    private fun decodeViaUnconstrainedMediaCodec(
        videoFile: File,
        normRot: Int,
        preferSoftware: Boolean,
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
                throw IllegalStateException("No video track found in source")
            }
            val mime = trackFormat.getString(MediaFormat.KEY_MIME)
                ?: throw IllegalStateException("Missing video MIME type")
            extractor.selectTrack(trackIndex)

            // Deliberately do NOT override KEY_COLOR_FORMAT so decoder uses its native YUV layout
            codec = createDecoderPreferringCompatible(mime, preferSoftware = preferSoftware)
            codec.configure(trackFormat, null, null, 0)
            codec.start()

            val info = MediaCodec.BufferInfo()
            var inputDone = false
            var outputDone = false
            var idlePolls = 0

            while (!outputDone && !isCancelled()) {
                if (!inputDone) {
                    val inIdx = codec.dequeueInputBuffer(6_000L)
                    if (inIdx >= 0) {
                        val inBuf = codec.getInputBuffer(inIdx)
                        val sampleSize = if (inBuf != null) extractor.readSampleData(inBuf, 0) else -1
                        if (sampleSize < 0) {
                            codec.queueInputBuffer(inIdx, 0, 0, 0L, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        } else {
                            codec.queueInputBuffer(inIdx, 0, sampleSize, extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }
                }

                val outIdx = codec.dequeueOutputBuffer(info, 6_000L)
                when {
                    outIdx >= 0 -> {
                        idlePolls = 0
                        val isEos = (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0
                        if (info.size > 0 && !isCancelled()) {
                            val bgrMat = decodeOutputBufferToBgrMat(codec, outIdx, normRot)
                            if (bgrMat != null && !bgrMat.empty()) {
                                onFrame(framesDecoded, bgrMat, info.presentationTimeUs)
                                framesDecoded++
                            } else {
                                bgrMat?.release()
                            }
                        }
                        codec.releaseOutputBuffer(outIdx, false)
                        if (isEos) outputDone = true
                    }
                    outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        idlePolls = 0
                    }
                    outIdx == MediaCodec.INFO_TRY_AGAIN_LATER -> {
                        idlePolls++
                        if (idlePolls > 700 && inputDone) outputDone = true
                    }
                }
            }
            return framesDecoded
        } finally {
            runCatching { codec?.stop() }
            runCatching { codec?.release() }
            runCatching { extractor.release() }
        }
    }

    private fun decodeViaMediaMetadataRetriever(
        videoFile: File,
        durationMs: Long,
        safeFps: Double,
        isCancelled: () -> Boolean,
        onFrame: (frameIndex: Int, bgrMat: Mat, ptsUs: Long) -> Unit
    ): Int {
        val retriever = MediaMetadataRetriever()
        var decoded = 0
        try {
            retriever.setDataSource(videoFile.absolutePath)
            val durUs = durationMs.coerceAtLeast(100L) * 1000L
            val stepUs = (1_000_000.0 / safeFps.coerceIn(5.0, 120.0)).roundToLong().coerceAtLeast(8_333L)
            var ptsUs = 0L
            while (ptsUs <= durUs && !isCancelled()) {
                val bmp: Bitmap? = try {
                    retriever.getFrameAtTime(ptsUs, MediaMetadataRetriever.OPTION_CLOSEST)
                } catch (_: Throwable) {
                    null
                }
                if (bmp != null) {
                    val rgbaMat = Mat()
                    val bgrMat = Mat()
                    try {
                        Utils.bitmapToMat(bmp, rgbaMat)
                        Imgproc.cvtColor(rgbaMat, bgrMat, Imgproc.COLOR_RGBA2BGR)
                        if (!bgrMat.empty()) {
                            onFrame(decoded, bgrMat, ptsUs)
                            decoded++
                        } else {
                            bgrMat.release()
                        }
                    } finally {
                        rgbaMat.release()
                        bmp.recycle()
                    }
                }
                ptsUs += stepUs
            }
            return decoded
        } finally {
            runCatching { retriever.release() }
        }
    }

    /**
     * Pure Android MediaMuxer audio+video finalizer:
     * 1. No audio -> direct MP4 copy (MediaMuxer already produced clean MP4 in rawVideoMp4).
     * 2. MP4-compatible audio (AAC with CSD-0, AMR-NB, AMR-WB) -> zero-reencode stream copy via MediaMuxer.
     * 3. Incompatible audio (Opus, Vorbis, MP3, FLAC, PCM) -> MediaCodec decode to PCM + AAC-LC 192kbps encode + MediaMuxer.
     * 4. Corrupt/unreadable audio -> graceful video-only MP4 fallback with warning log.
     */
    fun muxVideoWithOriginalAudio(
        rawVideoMp4: File,
        originalInput: File,
        finalMp4: File,
        hasAudio: Boolean,
        logger: EngineDiagnosticsLogger
    ) {
        if (!hasAudio) {
            rawVideoMp4.copyTo(finalMp4, overwrite = true)
            return
        }

        val audioInfo = inspectFirstAudioTrack(originalInput)
        if (audioInfo == null) {
            rawVideoMp4.copyTo(finalMp4, overwrite = true)
            return
        }
        val (audioMime, hasCsd0) = audioInfo

        // Fast path: stream copy without re-encoding if container track is MP4-muxer compatible
        if (MediaCodecProbeMath.isMp4PassthroughCompatibleAudioMime(audioMime, hasCsd0)) {
            val copyOk = runCatching {
                muxPassthroughWithMediaMuxer(rawVideoMp4, originalInput, finalMp4)
            }.onFailure { err ->
                logger.warn("Direct MediaMuxer audio passthrough failed (${err.message}); transcoding audio to AAC")
                if (finalMp4.exists()) finalMp4.delete()
            }.isSuccess

            if (copyOk && finalMp4.exists() && finalMp4.length() > 0L) {
                return
            }
        }

        // Transcode fallback: decode any Android-supported audio codec -> AAC-LC 192kbps -> MediaMuxer
        val aacOk = runCatching {
            muxWithAacTranscode(rawVideoMp4, originalInput, finalMp4)
        }.onFailure { err ->
            logger.warn("AAC audio transcode failed (${err.message}); saving video stream without audio track")
            if (finalMp4.exists()) finalMp4.delete()
        }.isSuccess

        if (!aacOk || !finalMp4.exists() || finalMp4.length() == 0L) {
            rawVideoMp4.copyTo(finalMp4, overwrite = true)
        }
    }

    private fun inspectFirstAudioTrack(file: File): Pair<String, Boolean>? {
        val extractor = MediaExtractor()
        return try {
            extractor.setDataSource(file.absolutePath)
            for (i in 0 until extractor.trackCount) {
                val fmt = extractor.getTrackFormat(i)
                val mime = fmt.getString(MediaFormat.KEY_MIME).orEmpty()
                if (mime.startsWith("audio/")) {
                    return mime to fmt.containsKey("csd-0")
                }
            }
            null
        } catch (_: Throwable) {
            null
        } finally {
            runCatching { extractor.release() }
        }
    }

    private fun muxPassthroughWithMediaMuxer(
        rawVideoMp4: File,
        originalInput: File,
        finalMp4: File
    ) {
        val videoExtractor = MediaExtractor()
        val audioExtractor = MediaExtractor()
        var muxer: MediaMuxer? = null
        var muxerStarted = false

        try {
            videoExtractor.setDataSource(rawVideoMp4.absolutePath)
            audioExtractor.setDataSource(originalInput.absolutePath)

            val (vIdx, vFmt) = findTrack(videoExtractor, "video/")
                ?: throw IllegalStateException("Enhanced video track missing")
            val (aIdx, aFmt) = findTrack(audioExtractor, "audio/")
                ?: throw IllegalStateException("Original audio track missing")

            videoExtractor.selectTrack(vIdx)
            audioExtractor.selectTrack(aIdx)

            if (finalMp4.exists()) finalMp4.delete()
            muxer = MediaMuxer(finalMp4.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
            val outVTrack = muxer.addTrack(vFmt)
            val outATrack = muxer.addTrack(aFmt)
            muxer.start()
            muxerStarted = true

            val vMaxSize = (vFmt.getIntOrNull(MediaFormat.KEY_MAX_INPUT_SIZE) ?: 0).coerceAtLeast(1024 * 1024)
            val aMaxSize = (aFmt.getIntOrNull(MediaFormat.KEY_MAX_INPUT_SIZE) ?: 0).coerceAtLeast(256 * 1024)
            var buffer = ByteBuffer.allocateDirect(max(vMaxSize, aMaxSize))
            val info = MediaCodec.BufferInfo()

            var maxVideoPtsUs = 0L
            var lastVideoPtsUs = -1L
            while (true) {
                buffer.clear()
                var sampleSize = videoExtractor.readSampleData(buffer, 0)
                if (sampleSize < 0) break
                val rawPts = videoExtractor.sampleTime.coerceAtLeast(0L)
                val safePts = if (lastVideoPtsUs < 0L) rawPts else max(rawPts, lastVideoPtsUs + 1L)
                lastVideoPtsUs = safePts
                maxVideoPtsUs = max(maxVideoPtsUs, safePts)

                info.offset = 0
                info.size = sampleSize
                info.presentationTimeUs = safePts
                info.flags = mapSampleFlags(videoExtractor.sampleFlags)
                muxer.writeSampleData(outVTrack, buffer, info)
                if (!videoExtractor.advance()) break
            }

            var lastAudioPtsUs = -1L
            while (true) {
                buffer.clear()
                val sampleSize = audioExtractor.readSampleData(buffer, 0)
                if (sampleSize < 0) break
                val rawPts = audioExtractor.sampleTime.coerceAtLeast(0L)
                if (maxVideoPtsUs > 0L && rawPts > maxVideoPtsUs) break
                val safePts = if (lastAudioPtsUs < 0L) rawPts else max(rawPts, lastAudioPtsUs + 1L)
                lastAudioPtsUs = safePts

                info.offset = 0
                info.size = sampleSize
                info.presentationTimeUs = safePts
                info.flags = mapSampleFlags(audioExtractor.sampleFlags)
                muxer.writeSampleData(outATrack, buffer, info)
                if (!audioExtractor.advance()) break
            }
        } finally {
            if (muxerStarted) {
                runCatching { muxer?.stop() }
            }
            runCatching { muxer?.release() }
            runCatching { videoExtractor.release() }
            runCatching { audioExtractor.release() }
        }
    }

    private fun muxWithAacTranscode(
        rawVideoMp4: File,
        originalInput: File,
        finalMp4: File
    ) {
        val videoExtractor = MediaExtractor()
        val audioExtractor = MediaExtractor()
        var decoder: MediaCodec? = null
        var encoder: MediaCodec? = null
        var muxer: MediaMuxer? = null
        var muxerStarted = false

        try {
            videoExtractor.setDataSource(rawVideoMp4.absolutePath)
            audioExtractor.setDataSource(originalInput.absolutePath)

            val (vIdx, vFmt) = findTrack(videoExtractor, "video/")
                ?: throw IllegalStateException("Enhanced video track missing")
            val (aIdx, aFmt) = findTrack(audioExtractor, "audio/")
                ?: throw IllegalStateException("Original audio track missing")
            val aMime = aFmt.getString(MediaFormat.KEY_MIME)
                ?: throw IllegalStateException("Missing audio MIME type")

            videoExtractor.selectTrack(vIdx)
            audioExtractor.selectTrack(aIdx)

            val initSampleRate = (aFmt.getIntOrNull(MediaFormat.KEY_SAMPLE_RATE) ?: 44_100).coerceIn(8_000, 96_000)
            val initChannels = (aFmt.getIntOrNull(MediaFormat.KEY_CHANNEL_COUNT) ?: 2).coerceAtLeast(1)

            decoder = MediaCodec.createDecoderByType(aMime)
            decoder.configure(aFmt, null, null, 0)
            decoder.start()

            if (finalMp4.exists()) finalMp4.delete()
            muxer = MediaMuxer(finalMp4.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
            val outVTrack = muxer.addTrack(vFmt)
            var outATrack = -1

            // Copy all enhanced video frames AFTER muxer starts; first find exact video duration to bound audio
            val videoDurationUs = (vFmt.getLongOrNull(MediaFormat.KEY_DURATION) ?: 0L).coerceAtLeast(0L)

            var decSampleRate = initSampleRate
            var decChannels = initChannels
            var decPcmFloat = false

            val decInfo = MediaCodec.BufferInfo()
            val encInfo = MediaCodec.BufferInfo()
            var decInputDone = false
            var decOutputDone = false
            var encOutputDone = false
            var totalSamplesEncoded = 0L
            var lastEncPtsUs = -1L

            fun ensureEncoderStarted(sampleRate: Int, channels: Int) {
                if (encoder != null) return
                val targetCh = channels.coerceIn(1, 2)
                val encFmt = MediaFormat.createAudioFormat(MediaFormat.MIMETYPE_AUDIO_AAC, sampleRate, targetCh).apply {
                    setInteger(MediaFormat.KEY_AAC_PROFILE, MediaCodecInfo.CodecProfileLevel.AACObjectLC)
                    setInteger(MediaFormat.KEY_BIT_RATE, 192_000)
                    setInteger(MediaFormat.KEY_MAX_INPUT_SIZE, 32 * 1024)
                }
                encoder = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_AUDIO_AAC).apply {
                    configure(encFmt, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
                    start()
                }
            }

            fun drainEncoderAndStartMuxerIfReady() {
                val enc = encoder ?: return
                while (true) {
                    val outIdx = enc.dequeueOutputBuffer(encInfo, 2_000L)
                    when {
                        outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                            if (!muxerStarted) {
                                outATrack = muxer.addTrack(enc.outputFormat)
                                muxer.start()
                                muxerStarted = true
                                copyAllVideoSamples(videoExtractor, muxer, outVTrack, vFmt)
                            }
                        }
                        outIdx >= 0 -> {
                            val encBuf = enc.getOutputBuffer(outIdx)
                            val isConfig = (encInfo.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG) != 0
                            if (!isConfig && encInfo.size > 0 && muxerStarted && outATrack >= 0 && encBuf != null) {
                                encBuf.position(encInfo.offset)
                                encBuf.limit(encInfo.offset + encInfo.size)
                                val pts = encInfo.presentationTimeUs.coerceAtLeast(0L)
                                if (videoDurationUs == 0L || pts <= videoDurationUs) {
                                    val safePts = if (lastEncPtsUs < 0L) pts else max(pts, lastEncPtsUs + 1L)
                                    lastEncPtsUs = safePts
                                    encInfo.presentationTimeUs = safePts
                                    muxer.writeSampleData(outATrack, encBuf, encInfo)
                                }
                            }
                            enc.releaseOutputBuffer(outIdx, false)
                            if ((encInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                                encOutputDone = true
                                break
                            }
                        }
                        else -> break
                    }
                }
            }

            fun feedPcmToEncoder(pcmBytes: ByteArray, sampleRate: Int, channels: Int) {
                ensureEncoderStarted(sampleRate, channels)
                val enc = encoder ?: return
                val targetCh = channels.coerceIn(1, 2)
                val readyBytes = if (channels > 2) {
                    MediaCodecProbeMath.downmixInterleavedPcm16ToStereo(pcmBytes, pcmBytes.size, channels)
                } else {
                    pcmBytes
                }
                val bytesPerFrame = targetCh * 2
                var offset = 0
                while (offset < readyBytes.size) {
                    val inIdx = enc.dequeueInputBuffer(5_000L)
                    if (inIdx >= 0) {
                        val inBuf = enc.getInputBuffer(inIdx) ?: break
                        inBuf.clear()
                        val chunkLen = minOf(inBuf.capacity(), readyBytes.size - offset)
                        inBuf.put(readyBytes, offset, chunkLen)
                        val ptsUs = (totalSamplesEncoded * 1_000_000L) / sampleRate.coerceAtLeast(8_000)
                        totalSamplesEncoded += (chunkLen / bytesPerFrame).coerceAtLeast(1)
                        enc.queueInputBuffer(inIdx, 0, chunkLen, ptsUs, 0)
                        offset += chunkLen
                    }
                    drainEncoderAndStartMuxerIfReady()
                }
            }

            while (!encOutputDone) {
                if (!decInputDone) {
                    val inIdx = decoder.dequeueInputBuffer(5_000L)
                    if (inIdx >= 0) {
                        val inBuf = decoder.getInputBuffer(inIdx)
                        val size = if (inBuf != null) audioExtractor.readSampleData(inBuf, 0) else -1
                        val pts = audioExtractor.sampleTime
                        if (size < 0 || (videoDurationUs > 0L && pts > videoDurationUs + 500_000L)) {
                            decoder.queueInputBuffer(inIdx, 0, 0, 0L, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            decInputDone = true
                        } else {
                            decoder.queueInputBuffer(inIdx, 0, size, pts.coerceAtLeast(0L), 0)
                            audioExtractor.advance()
                        }
                    }
                }

                if (!decOutputDone) {
                    val outIdx = decoder.dequeueOutputBuffer(decInfo, 5_000L)
                    when {
                        outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                            val outFmt = decoder.outputFormat
                            decSampleRate = (outFmt.getIntOrNull(MediaFormat.KEY_SAMPLE_RATE) ?: decSampleRate).coerceIn(8_000, 96_000)
                            decChannels = (outFmt.getIntOrNull(MediaFormat.KEY_CHANNEL_COUNT) ?: decChannels).coerceAtLeast(1)
                            decPcmFloat = Build.VERSION.SDK_INT >= Build.VERSION_CODES.N &&
                                outFmt.getIntOrNull(MediaFormat.KEY_PCM_ENCODING) == AudioFormat.ENCODING_PCM_FLOAT
                        }
                        outIdx >= 0 -> {
                            val decBuf = decoder.getOutputBuffer(outIdx)
                            if (decInfo.size > 0 && decBuf != null) {
                                decBuf.position(decInfo.offset)
                                decBuf.limit(decInfo.offset + decInfo.size)
                                val pcmBytes = if (decPcmFloat) {
                                    convertPcmFloatToPcm16Bytes(decBuf, decInfo.size)
                                } else {
                                    ByteArray(decInfo.size).also { decBuf.get(it) }
                                }
                                feedPcmToEncoder(pcmBytes, decSampleRate, decChannels)
                            }
                            decoder.releaseOutputBuffer(outIdx, false)
                            if ((decInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                                decOutputDone = true
                                ensureEncoderStarted(decSampleRate, decChannels)
                                val enc = encoder!!
                                while (true) {
                                    val eosIdx = enc.dequeueInputBuffer(5_000L)
                                    if (eosIdx >= 0) {
                                        val eosPts = (totalSamplesEncoded * 1_000_000L) / decSampleRate.coerceAtLeast(8_000)
                                        enc.queueInputBuffer(eosIdx, 0, 0, eosPts, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                                        break
                                    }
                                    drainEncoderAndStartMuxerIfReady()
                                }
                            }
                        }
                    }
                }
                drainEncoderAndStartMuxerIfReady()
                if (decOutputDone && encoder == null) break
            }

            if (!muxerStarted) {
                throw IllegalStateException("Audio AAC encoder never emitted output format")
            }
        } finally {
            if (muxerStarted) {
                runCatching { muxer?.stop() }
            }
            runCatching { muxer?.release() }
            runCatching { decoder?.stop() }
            runCatching { decoder?.release() }
            runCatching { encoder?.stop() }
            runCatching { encoder?.release() }
            runCatching { videoExtractor.release() }
            runCatching { audioExtractor.release() }
        }
    }

    private fun copyAllVideoSamples(
        videoExtractor: MediaExtractor,
        muxer: MediaMuxer,
        outVTrack: Int,
        vFmt: MediaFormat
    ) {
        val vMaxSize = (vFmt.getIntOrNull(MediaFormat.KEY_MAX_INPUT_SIZE) ?: 0).coerceAtLeast(1024 * 1024)
        val buffer = ByteBuffer.allocateDirect(vMaxSize)
        val info = MediaCodec.BufferInfo()
        var lastPts = -1L
        while (true) {
            buffer.clear()
            val sampleSize = videoExtractor.readSampleData(buffer, 0)
            if (sampleSize < 0) break
            val rawPts = videoExtractor.sampleTime.coerceAtLeast(0L)
            val safePts = if (lastPts < 0L) rawPts else max(rawPts, lastPts + 1L)
            lastPts = safePts

            info.offset = 0
            info.size = sampleSize
            info.presentationTimeUs = safePts
            info.flags = mapSampleFlags(videoExtractor.sampleFlags)
            muxer.writeSampleData(outVTrack, buffer, info)
            if (!videoExtractor.advance()) break
        }
    }

    private fun convertPcmFloatToPcm16Bytes(floatBuf: ByteBuffer, byteSize: Int): ByteArray {
        val sampleCount = byteSize / 4
        val out = ByteArray(sampleCount * 2)
        val src = floatBuf.slice().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        val dst = ByteBuffer.wrap(out).order(ByteOrder.LITTLE_ENDIAN)
        for (i in 0 until sampleCount) {
            val s = (src.get(i).coerceIn(-1f, 1f) * 32767f).roundToInt()
                .coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt())
            dst.putShort(s.toShort())
        }
        return out
    }

    private fun decodeOutputBufferToBgrMat(
        codec: MediaCodec,
        outIdx: Int,
        normRot: Int
    ): Mat? {
        val image = runCatching { codec.getOutputImage(outIdx) }.getOrNull()
        if (image != null) {
            return try {
                convertImageYuvToBgrMat(image, normRot)
            } finally {
                image.close()
            }
        }
        val outBuf = codec.getOutputBuffer(outIdx) ?: return null
        val fmt = codec.outputFormat
        return convertRawYuvBufferToBgrMat(outBuf, fmt, normRot)
    }

    private fun convertImageYuvToBgrMat(image: Image, normRot: Int): Mat? {
        val crop = image.cropRect
        val rawW = if (crop.width() > 0) crop.width() else image.width
        val rawH = if (crop.height() > 0) crop.height() else image.height
        val w = if (rawW % 2 == 0) rawW else rawW - 1
        val h = if (rawH % 2 == 0) rawH else rawH - 1
        if (w <= 0 || h <= 0) return null

        val ySize = w * h
        val uvSize = ySize / 4
        val yuvBytes = ByteArray(ySize + 2 * uvSize)
        val scratchRow = ByteArray((w * 2).coerceAtLeast(16))
        val planes = image.planes

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
            scratchRowBuf = scratchRow
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

        val cvCode = when (layout) {
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
                    scratchRowBuf = scratchRow
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
                    scratchRowBuf = scratchRow
                )
                Imgproc.COLOR_YUV2BGR_I420
            }
        }

        val yuvMat = Mat(h + h / 2, w, CvType.CV_8UC1)
        val bgrMat = Mat(h, w, CvType.CV_8UC3)
        return try {
            yuvMat.put(0, 0, yuvBytes)
            Imgproc.cvtColor(yuvMat, bgrMat, cvCode)
            applyRotationIfNeeded(bgrMat, normRot)
        } finally {
            yuvMat.release()
        }
    }

    private fun convertRawYuvBufferToBgrMat(
        rawBuf: ByteBuffer,
        fmt: MediaFormat,
        normRot: Int
    ): Mat? {
        val rawW = fmt.getIntOrNull(MediaFormat.KEY_WIDTH) ?: return null
        val rawH = fmt.getIntOrNull(MediaFormat.KEY_HEIGHT) ?: return null
        val w = if (rawW % 2 == 0) rawW else rawW - 1
        val h = if (rawH % 2 == 0) rawH else rawH - 1
        if (w <= 0 || h <= 0) return null

        val stride = (fmt.getIntOrNull(MediaFormat.KEY_STRIDE) ?: rawW).coerceAtLeast(w)
        val sliceH = (fmt.getIntOrNull(MediaFormat.KEY_SLICE_HEIGHT) ?: rawH).coerceAtLeast(h)
        val colorFmt = fmt.getIntOrNull(MediaFormat.KEY_COLOR_FORMAT)
            ?: MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420SemiPlanar

        val isSemiPlanar = colorFmt == MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420SemiPlanar ||
            colorFmt == MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420PackedSemiPlanar

        val ySize = w * h
        val uvSize = ySize / 4
        val yuvBytes = ByteArray(ySize + 2 * uvSize)
        val buf = rawBuf.duplicate()
        val limit = buf.limit()

        for (r in 0 until h) {
            val srcPos = r * stride
            val avail = (limit - srcPos).coerceAtLeast(0).coerceAtMost(w)
            if (avail > 0) {
                buf.position(srcPos)
                buf.get(yuvBytes, r * w, avail)
            }
        }

        val uvBase = stride * sliceH
        val cvCode = if (isSemiPlanar) {
            for (r in 0 until h / 2) {
                val srcPos = uvBase + r * stride
                val avail = (limit - srcPos).coerceAtLeast(0).coerceAtMost(w)
                if (avail > 0) {
                    buf.position(srcPos)
                    buf.get(yuvBytes, ySize + r * w, avail)
                }
            }
            Imgproc.COLOR_YUV2BGR_NV12
        } else {
            val uvStride = (stride / 2).coerceAtLeast(w / 2)
            val uvPlaneBytes = uvStride * (sliceH / 2)
            for (r in 0 until h / 2) {
                val uSrc = uvBase + r * uvStride
                val uAvail = (limit - uSrc).coerceAtLeast(0).coerceAtMost(w / 2)
                if (uAvail > 0) {
                    buf.position(uSrc)
                    buf.get(yuvBytes, ySize + r * (w / 2), uAvail)
                }
                val vSrc = uvBase + uvPlaneBytes + r * uvStride
                val vAvail = (limit - vSrc).coerceAtLeast(0).coerceAtMost(w / 2)
                if (vAvail > 0) {
                    buf.position(vSrc)
                    buf.get(yuvBytes, ySize + uvSize + r * (w / 2), vAvail)
                }
            }
            Imgproc.COLOR_YUV2BGR_I420
        }

        val yuvMat = Mat(h + h / 2, w, CvType.CV_8UC1)
        val bgrMat = Mat(h, w, CvType.CV_8UC3)
        return try {
            yuvMat.put(0, 0, yuvBytes)
            Imgproc.cvtColor(yuvMat, bgrMat, cvCode)
            applyRotationIfNeeded(bgrMat, normRot)
        } finally {
            yuvMat.release()
        }
    }

    private fun applyRotationIfNeeded(bgrMat: Mat, normRot: Int): Mat {
        if (normRot != 90 && normRot != 180 && normRot != 270) return bgrMat
        val rotated = Mat()
        val rotCode = when (normRot) {
            90 -> Core.ROTATE_90_CLOCKWISE
            180 -> Core.ROTATE_180
            else -> Core.ROTATE_90_COUNTERCLOCKWISE
        }
        Core.rotate(bgrMat, rotated, rotCode)
        bgrMat.release()
        return rotated
    }

    private fun resizeForMaxWidth(src: Mat, maxWidth: Int): Mat {
        if (src.cols() <= maxWidth) return src
        val scale = maxWidth.toDouble() / src.cols().toDouble()
        val dst = Mat()
        Imgproc.resize(
            src,
            dst,
            org.opencv.core.Size(maxWidth.toDouble(), (src.rows() * scale).coerceAtLeast(32.0)),
            0.0,
            0.0,
            Imgproc.INTER_AREA
        )
        return dst
    }

    private fun createDecoderPreferringCompatible(mime: String, preferSoftware: Boolean): MediaCodec {
        val infos = MediaCodecList(MediaCodecList.REGULAR_CODECS).codecInfos
        if (preferSoftware) {
            val sw = infos.firstOrNull { info ->
                !info.isEncoder &&
                    info.supportedTypes.any { it.equals(mime, ignoreCase = true) } &&
                    isSoftwareCodec(info)
            }
            if (sw != null) {
                return MediaCodec.createByCodecName(sw.name)
            }
        }
        return MediaCodec.createDecoderByType(mime)
    }

    private fun isSoftwareCodec(info: android.media.MediaCodecInfo): Boolean {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            return info.isSoftwareOnly
        }
        val n = info.name.lowercase()
        return n.startsWith("c2.android.") || n.startsWith("omx.google.")
    }

    private fun findTrack(extractor: MediaExtractor, mimePrefix: String): Pair<Int, MediaFormat>? {
        for (i in 0 until extractor.trackCount) {
            val fmt = extractor.getTrackFormat(i)
            val mime = fmt.getString(MediaFormat.KEY_MIME).orEmpty()
            if (mime.startsWith(mimePrefix)) {
                return i to fmt
            }
        }
        return null
    }

    private fun mapSampleFlags(extractorFlags: Int): Int {
        var flags = 0
        if ((extractorFlags and MediaExtractor.SAMPLE_FLAG_SYNC) != 0) {
            flags = flags or MediaCodec.BUFFER_FLAG_KEY_FRAME
        }
        if ((extractorFlags and MediaExtractor.SAMPLE_FLAG_PARTIAL_FRAME) != 0 &&
            Build.VERSION.SDK_INT >= Build.VERSION_CODES.O
        ) {
            flags = flags or MediaCodec.BUFFER_FLAG_PARTIAL_FRAME
        }
        return flags
    }

    private fun MediaFormat.getIntOrNull(key: String): Int? {
        if (!containsKey(key)) return null
        return runCatching { getInteger(key) }.getOrNull()
    }

    private fun MediaFormat.getLongOrNull(key: String): Long? {
        if (!containsKey(key)) return null
        return runCatching { getLong(key) }.getOrNull()
    }

    private fun MediaFormat.getFrameRateOrNull(): Double? {
        if (!containsKey(MediaFormat.KEY_FRAME_RATE)) return null
        val fps = runCatching { getFloat(MediaFormat.KEY_FRAME_RATE).toDouble() }.getOrNull()
            ?: runCatching { getInteger(MediaFormat.KEY_FRAME_RATE).toDouble() }.getOrNull()
            ?: return null
        return if (fps in 5.0..120.0) fps else null
    }
}
