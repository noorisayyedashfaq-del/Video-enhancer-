package com.example

import com.example.engine.AndroidMediaToolkit
import com.example.engine.MediaCodecProbeMath
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder

class AndroidMediaToolkitTest {

    @Test
    fun `maps video and audio MIME types to clean codec labels`() {
        assertEquals("h264", MediaCodecProbeMath.mapVideoMimeToCodecLabel("video/avc"))
        assertEquals("hevc", MediaCodecProbeMath.mapVideoMimeToCodecLabel("video/hevc"))
        assertEquals("vp9", MediaCodecProbeMath.mapVideoMimeToCodecLabel("video/x-vnd.on2.vp9"))
        assertEquals("av1", MediaCodecProbeMath.mapVideoMimeToCodecLabel("video/av01"))
        assertEquals("unknown", MediaCodecProbeMath.mapVideoMimeToCodecLabel(null))

        assertEquals("aac", MediaCodecProbeMath.mapAudioMimeToCodecLabel("audio/mp4a-latm"))
        assertEquals("mp3", MediaCodecProbeMath.mapAudioMimeToCodecLabel("audio/mpeg"))
        assertEquals("opus", MediaCodecProbeMath.mapAudioMimeToCodecLabel("audio/opus"))
        assertEquals("vorbis", MediaCodecProbeMath.mapAudioMimeToCodecLabel("audio/vorbis"))
        assertNull(MediaCodecProbeMath.mapAudioMimeToCodecLabel(null))
    }

    @Test
    fun `checks MP4 passthrough compatibility vs AAC transcode fallback`() {
        assertTrue(MediaCodecProbeMath.isMp4PassthroughCompatibleAudioMime("audio/mp4a-latm", hasCsd0 = true))
        assertFalse(MediaCodecProbeMath.isMp4PassthroughCompatibleAudioMime("audio/mp4a-latm", hasCsd0 = false))
        assertTrue(MediaCodecProbeMath.isMp4PassthroughCompatibleAudioMime("audio/3gpp", hasCsd0 = false))
        assertTrue(MediaCodecProbeMath.isMp4PassthroughCompatibleAudioMime("audio/amr-wb", hasCsd0 = false))
        assertFalse(MediaCodecProbeMath.isMp4PassthroughCompatibleAudioMime("audio/opus", hasCsd0 = true))
        assertFalse(MediaCodecProbeMath.isMp4PassthroughCompatibleAudioMime("audio/mpeg", hasCsd0 = false))
    }

    @Test
    fun `estimates FPS accurately even with B-frame presentation timestamp reordering`() {
        // 30fps (33_333 us intervals) presented out-of-order before sorting
        val bFramePts = listOf(0L, 66_667L, 33_333L, 133_333L, 100_000L, 166_667L)
        val fps = MediaCodecProbeMath.estimateFpsFromPtsList(bFramePts)
        assertNotNull(fps)
        assertEquals(30.0, fps!!, 0.2)

        assertNull(MediaCodecProbeMath.estimateFpsFromPtsList(listOf(0L)))
    }

    @Test
    fun `downmixes 5_1 interleaved PCM16 audio safely to stereo`() {
        val frameBytes = ByteArray(6 * 2)
        val inBuf = ByteBuffer.wrap(frameBytes).order(ByteOrder.LITTLE_ENDIAN)
        inBuf.putShort(1000)  // L
        inBuf.putShort(2000)  // R
        inBuf.putShort(1000)  // C (70% -> 700 mixed into L and R)
        inBuf.putShort(0)     // LFE
        inBuf.putShort(0)     // Ls
        inBuf.putShort(0)     // Rs

        val stereo = MediaCodecProbeMath.downmixInterleavedPcm16ToStereo(frameBytes, frameBytes.size, 6)
        assertEquals(4, stereo.size)
        val outBuf = ByteBuffer.wrap(stereo).order(ByteOrder.LITTLE_ENDIAN)
        assertEquals(1700.toShort(), outBuf.short)
        assertEquals(2700.toShort(), outBuf.short)
    }

    @Test
    fun `media codec readiness check executes safely without throwing on JVM`() {
        // Under pure JVM without Android framework, MediaCodecList throws RuntimeException which must be caught safely
        val ready = AndroidMediaToolkit.isMediaCodecSubsystemReady()
        assertFalse(ready)
    }
}
