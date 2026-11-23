package com.example.engine

import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLSurface
import android.opengl.GLES31
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Offscreen OpenGL ES 3.1 Compute Shader processor (backed by Vulkan/Adreno/Mali GPU drivers
 * on Android 7+/10+) for hardware-accelerated contrast/unsharp detail/saturation enhancement.
 * Probed safely once; gracefully disables itself if running in a headless JVM environment.
 */
class GlesGpuImageProcessor {

    private var eglDisplay: EGLDisplay? = null
    private var eglContext: EGLContext? = null
    private var eglSurface: EGLSurface? = null
    private var computeProgram: Int = 0
    private var texIn: Int = 0
    private var texOut: Int = 0
    private var uWidthLoc: Int = -1
    private var uHeightLoc: Int = -1
    private var uSharpnessLoc: Int = -1
    private var uContrastLoc: Int = -1
    private var uSaturationLoc: Int = -1

    private var initialized = false
    private val probeFailed = AtomicBoolean(false)
    private var rgbaScratchMat: Mat? = null
    private var inPixelBuf: ByteBuffer? = null
    private var outPixelBuf: ByteBuffer? = null
    private var rgbaIoBytes: ByteArray = ByteArray(0)
    private var fboIds = IntArray(1)

    val isAvailable: Boolean get() = initialized && !probeFailed.get()

    @Synchronized
    fun ensureInitialized(logger: EngineDiagnosticsLogger? = null): Boolean {
        if (initialized) return true
        if (probeFailed.get()) return false

        return try {
            val display = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
            if (display == null || display == EGL14.EGL_NO_DISPLAY) {
                probeFailed.set(true)
                return false
            }
            val version = IntArray(2)
            if (!EGL14.eglInitialize(display, version, 0, version, 1)) {
                probeFailed.set(true)
                return false
            }
            eglDisplay = display

            val configAttribs = intArrayOf(
                EGL14.EGL_RENDERABLE_TYPE, 0x0040, // EGL_OPENGL_ES3_BIT_KHR
                EGL14.EGL_SURFACE_TYPE, EGL14.EGL_PBUFFER_BIT,
                EGL14.EGL_RED_SIZE, 8,
                EGL14.EGL_GREEN_SIZE, 8,
                EGL14.EGL_BLUE_SIZE, 8,
                EGL14.EGL_ALPHA_SIZE, 8,
                EGL14.EGL_NONE
            )
            val configs = arrayOfNulls<EGLConfig>(1)
            val numConfigs = IntArray(1)
            if (!EGL14.eglChooseConfig(display, configAttribs, 0, configs, 0, 1, numConfigs, 0) || numConfigs[0] <= 0) {
                release()
                probeFailed.set(true)
                return false
            }
            val config = configs[0] ?: run {
                release()
                probeFailed.set(true)
                return false
            }

            val contextAttribs = intArrayOf(
                EGL14.EGL_CONTEXT_CLIENT_VERSION, 3,
                EGL14.EGL_NONE
            )
            val ctx = EGL14.eglCreateContext(display, config, EGL14.EGL_NO_CONTEXT, contextAttribs, 0)
            if (ctx == null || ctx == EGL14.EGL_NO_CONTEXT) {
                release()
                probeFailed.set(true)
                return false
            }
            eglContext = ctx

            val pbufferAttribs = intArrayOf(
                EGL14.EGL_WIDTH, 16,
                EGL14.EGL_HEIGHT, 16,
                EGL14.EGL_NONE
            )
            val surface = EGL14.eglCreatePbufferSurface(display, config, pbufferAttribs, 0)
            if (surface == null || surface == EGL14.EGL_NO_SURFACE) {
                release()
                probeFailed.set(true)
                return false
            }
            eglSurface = surface

            if (!EGL14.eglMakeCurrent(display, surface, surface, ctx)) {
                release()
                probeFailed.set(true)
                return false
            }

            val prog = createComputeProgram(COMPUTE_SHADER_SOURCE)
            if (prog == 0) {
                release()
                probeFailed.set(true)
                return false
            }
            computeProgram = prog
            uWidthLoc = GLES31.glGetUniformLocation(prog, "uWidth")
            uHeightLoc = GLES31.glGetUniformLocation(prog, "uHeight")
            uSharpnessLoc = GLES31.glGetUniformLocation(prog, "uSharpness")
            uContrastLoc = GLES31.glGetUniformLocation(prog, "uContrast")
            uSaturationLoc = GLES31.glGetUniformLocation(prog, "uSaturation")

            val textures = IntArray(2)
            GLES31.glGenTextures(2, textures, 0)
            texIn = textures[0]
            texOut = textures[1]
            GLES31.glGenFramebuffers(1, fboIds, 0)

            // Detach from init thread so processing thread can bind it smoothly
            EGL14.eglMakeCurrent(display, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT)
            initialized = true
            logger?.info("GLES 3.1 GPU Compute shader initialized successfully")
            true
        } catch (t: Throwable) {
            logger?.info("GLES 3.1 GPU compute unavailable (${t.message}); using native SIMD OpenCV")
            release()
            probeFailed.set(true)
            false
        }
    }

    /**
     * Benchmarks GLES 3.1 compute dispatch latency on a 256x256 RGBA patch.
     * Returns duration in ms or null if unsupported.
     */
    @Synchronized
    fun benchmarkMs(): Long? {
        if (!ensureInitialized()) return null
        val testMat = Mat(256, 256, org.opencv.core.CvType.CV_8UC3)
        return try {
            testMat.setTo(org.opencv.core.Scalar(90.0, 120.0, 150.0))
            // Warmup
            if (!enhanceBgrInPlace(testMat, sharpness = 0.25f, contrast = 1.08f, saturation = 1.06f)) {
                return null
            }
            val t0 = System.nanoTime()
            val ok = enhanceBgrInPlace(testMat, sharpness = 0.25f, contrast = 1.08f, saturation = 1.06f)
            if (!ok) null else ((System.nanoTime() - t0) / 1_000_000L).coerceAtLeast(1L)
        } catch (_: Throwable) {
            null
        } finally {
            testMat.release()
        }
    }

    @Synchronized
    fun enhanceBgrInPlace(
        bgr: Mat,
        sharpness: Float,
        contrast: Float,
        saturation: Float
    ): Boolean {
        if (!initialized || probeFailed.get()) return false
        val w = bgr.cols()
        val h = bgr.rows()
        if (w < 16 || h < 16) return false

        return try {
            if (!EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)) {
                return false
            }

            val rgba = rgbaScratchMat?.takeIf { it.cols() == w && it.rows() == h }
                ?: Mat(h, w, org.opencv.core.CvType.CV_8UC4).also {
                    rgbaScratchMat?.release()
                    rgbaScratchMat = it
                }
            Imgproc.cvtColor(bgr, rgba, Imgproc.COLOR_BGR2RGBA)

            val neededBytes = w * h * 4
            val inBuf = inPixelBuf?.takeIf { it.capacity() >= neededBytes }
                ?: ByteBuffer.allocateDirect(neededBytes).order(ByteOrder.nativeOrder()).also {
                    inPixelBuf = it
                }
            val outBuf = outPixelBuf?.takeIf { it.capacity() >= neededBytes }
                ?: ByteBuffer.allocateDirect(neededBytes).order(ByteOrder.nativeOrder()).also {
                    outPixelBuf = it
                }
            val rawBytes = if (rgbaIoBytes.size >= neededBytes) {
                rgbaIoBytes
            } else {
                ByteArray(neededBytes).also { rgbaIoBytes = it }
            }

            rgba.get(0, 0, rawBytes)
            inBuf.clear()
            inBuf.put(rawBytes, 0, neededBytes)
            inBuf.flip()

            GLES31.glBindTexture(GLES31.GL_TEXTURE_2D, texIn)
            GLES31.glTexParameteri(GLES31.GL_TEXTURE_2D, GLES31.GL_TEXTURE_MIN_FILTER, GLES31.GL_NEAREST)
            GLES31.glTexParameteri(GLES31.GL_TEXTURE_2D, GLES31.GL_TEXTURE_MAG_FILTER, GLES31.GL_NEAREST)
            GLES31.glTexStorage2D(GLES31.GL_TEXTURE_2D, 1, GLES31.GL_RGBA8, w, h)
            GLES31.glTexSubImage2D(
                GLES31.GL_TEXTURE_2D, 0, 0, 0, w, h,
                GLES31.GL_RGBA, GLES31.GL_UNSIGNED_BYTE, inBuf
            )

            GLES31.glBindTexture(GLES31.GL_TEXTURE_2D, texOut)
            GLES31.glTexParameteri(GLES31.GL_TEXTURE_2D, GLES31.GL_TEXTURE_MIN_FILTER, GLES31.GL_NEAREST)
            GLES31.glTexParameteri(GLES31.GL_TEXTURE_2D, GLES31.GL_TEXTURE_MAG_FILTER, GLES31.GL_NEAREST)
            GLES31.glTexStorage2D(GLES31.GL_TEXTURE_2D, 1, GLES31.GL_RGBA8, w, h)

            GLES31.glUseProgram(computeProgram)
            GLES31.glUniform1i(uWidthLoc, w)
            GLES31.glUniform1i(uHeightLoc, h)
            GLES31.glUniform1f(uSharpnessLoc, sharpness.coerceIn(0f, 0.8f))
            GLES31.glUniform1f(uContrastLoc, contrast.coerceIn(0.9f, 1.35f))
            GLES31.glUniform1f(uSaturationLoc, saturation.coerceIn(0.85f, 1.35f))

            GLES31.glBindImageTexture(0, texIn, 0, false, 0, GLES31.GL_READ_ONLY, GLES31.GL_RGBA8)
            GLES31.glBindImageTexture(1, texOut, 0, false, 0, GLES31.GL_WRITE_ONLY, GLES31.GL_RGBA8)

            val groupsX = (w + 15) / 16
            val groupsY = (h + 15) / 16
            GLES31.glDispatchCompute(groupsX, groupsY, 1)
            GLES31.glMemoryBarrier(GLES31.GL_FRAMEBUFFER_BARRIER_BIT or GLES31.GL_TEXTURE_FETCH_BARRIER_BIT or GLES31.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

            GLES31.glBindFramebuffer(GLES31.GL_FRAMEBUFFER, fboIds[0])
            GLES31.glFramebufferTexture2D(GLES31.GL_FRAMEBUFFER, GLES31.GL_COLOR_ATTACHMENT0, GLES31.GL_TEXTURE_2D, texOut, 0)
            outBuf.clear()
            GLES31.glReadPixels(0, 0, w, h, GLES31.GL_RGBA, GLES31.GL_UNSIGNED_BYTE, outBuf)
            GLES31.glBindFramebuffer(GLES31.GL_FRAMEBUFFER, 0)

            outBuf.rewind()
            outBuf.get(rawBytes, 0, neededBytes)
            rgba.put(0, 0, rawBytes)
            Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)

            EGL14.eglMakeCurrent(eglDisplay, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT)
            true
        } catch (_: Throwable) {
            try {
                EGL14.eglMakeCurrent(eglDisplay, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT)
            } catch (_: Throwable) {
            }
            false
        }
    }

    private fun createComputeProgram(source: String): Int {
        val shader = GLES31.glCreateShader(GLES31.GL_COMPUTE_SHADER)
        if (shader == 0) return 0
        GLES31.glShaderSource(shader, source)
        GLES31.glCompileShader(shader)
        val compiled = IntArray(1)
        GLES31.glGetShaderiv(shader, GLES31.GL_COMPILE_STATUS, compiled, 0)
        if (compiled[0] == 0) {
            GLES31.glDeleteShader(shader)
            return 0
        }
        val program = GLES31.glCreateProgram()
        if (program == 0) {
            GLES31.glDeleteShader(shader)
            return 0
        }
        GLES31.glAttachShader(program, shader)
        GLES31.glLinkProgram(program)
        GLES31.glDeleteShader(shader)
        val linked = IntArray(1)
        GLES31.glGetProgramiv(program, GLES31.GL_LINK_STATUS, linked, 0)
        if (linked[0] == 0) {
            GLES31.glDeleteProgram(program)
            return 0
        }
        return program
    }

    @Synchronized
    fun release() {
        try {
            rgbaScratchMat?.release()
            rgbaScratchMat = null
            inPixelBuf = null
            outPixelBuf = null
            rgbaIoBytes = ByteArray(0)
            val disp = eglDisplay
            if (disp != null && disp != EGL14.EGL_NO_DISPLAY) {
                val ctx = eglContext
                val surf = eglSurface
                if (ctx != null && ctx != EGL14.EGL_NO_CONTEXT && surf != null && surf != EGL14.EGL_NO_SURFACE) {
                    EGL14.eglMakeCurrent(disp, surf, surf, ctx)
                    if (computeProgram != 0) GLES31.glDeleteProgram(computeProgram)
                    if (texIn != 0 || texOut != 0) GLES31.glDeleteTextures(2, intArrayOf(texIn, texOut), 0)
                    if (fboIds[0] != 0) GLES31.glDeleteFramebuffers(1, fboIds, 0)
                }
                EGL14.eglMakeCurrent(disp, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT)
                if (surf != null && surf != EGL14.EGL_NO_SURFACE) EGL14.eglDestroySurface(disp, surf)
                if (ctx != null && ctx != EGL14.EGL_NO_CONTEXT) EGL14.eglDestroyContext(disp, ctx)
                EGL14.eglTerminate(disp)
            }
        } catch (_: Throwable) {
        } finally {
            eglDisplay = null
            eglContext = null
            eglSurface = null
            computeProgram = 0
            texIn = 0
            texOut = 0
            fboIds[0] = 0
            initialized = false
        }
    }

    companion object {
        private const val COMPUTE_SHADER_SOURCE = """
            #version 310 es
            layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
            layout(rgba8, binding = 0) uniform readonly highp image2D uInput;
            layout(rgba8, binding = 1) uniform writeonly highp image2D uOutput;
            uniform int uWidth;
            uniform int uHeight;
            uniform float uSharpness;
            uniform float uContrast;
            uniform float uSaturation;

            void main() {
                ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
                if (pos.x >= uWidth || pos.y >= uHeight) return;

                vec4 c = imageLoad(uInput, pos);
                vec4 n = imageLoad(uInput, ivec2(pos.x, max(pos.y - 1, 0)));
                vec4 s = imageLoad(uInput, ivec2(pos.x, min(pos.y + 1, uHeight - 1)));
                vec4 w = imageLoad(uInput, ivec2(max(pos.x - 1, 0), pos.y));
                vec4 e = imageLoad(uInput, ivec2(min(pos.x + 1, uWidth - 1), pos.y));

                vec3 blur = (n.rgb + s.rgb + w.rgb + e.rgb) * 0.25;
                vec3 detail = c.rgb - blur;
                vec3 res = clamp(c.rgb + detail * uSharpness, 0.0, 1.0);

                // Micro-contrast around mid-gray + luminance-preserving saturation boost
                res = clamp((res - 0.5) * uContrast + 0.5, 0.0, 1.0);
                float luma = dot(res, vec3(0.299, 0.587, 0.114));
                res = clamp(mix(vec3(luma), res, uSaturation), 0.0, 1.0);

                imageStore(uOutput, pos, vec4(res, 1.0));
            }
        """
    }
}
