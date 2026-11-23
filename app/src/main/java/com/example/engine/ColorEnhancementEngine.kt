package com.example.engine

import com.example.model.AdaptiveAdjustmentParams
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import java.util.Locale
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.roundToInt

/**
 * Pure JVM-testable color enhancement mathematics:
 * - Adaptive gray-world white balance correction (detects shifts >= 0.045 instead of ignoring casts < 0.18)
 * - Luminance-preserving selective Vibrance (boosts faded pixels ~ (1 - sat)^1.25 while leaving vivid pixels natural)
 * - Continuous YCrCb skin-tone shield so warm human skin hues resist over-saturation & clipping
 * - Highlight soft-shoulder compression + midtone brilliance & S-curve contrast without blown whites
 */
object ColorMath {

    fun computeAutoWhiteBalanceGains(
        wbShiftR: Float,
        wbShiftB: Float,
        tempShift: Float = 0f
    ): Pair<Float, Float> {
        val devR = wbShiftR - 1f
        val devB = wbShiftB - 1f
        val hasCastR = abs(devR) > 0.045f
        val hasCastB = abs(devB) > 0.045f
        val tempBias = (tempShift / 100f).coerceIn(-0.12f, 0.12f)
        val gainR = (if (hasCastR) 1f + devR * 0.62f else 1f) + tempBias
        val gainB = (if (hasCastB) 1f + devB * 0.62f else 1f) - tempBias
        return gainR.coerceIn(0.86f, 1.16f) to gainB.coerceIn(0.86f, 1.16f)
    }

    /**
     * Returns 0.0 for vivid skin-locus pixels (strong protection) up to 1.0 for non-skin sky/foliage/garments.
     * Skin protection reduces extra chroma gain to 25% of full vibrance/saturation boost.
     */
    fun skinNonMaskWeight(rNorm: Float, gNorm: Float, bNorm: Float): Float {
        val y = 0.299f * rNorm + 0.587f * gNorm + 0.114f * bNorm
        val cr = (rNorm - y) * 0.713f + 0.5f
        val cb = (bNorm - y) * 0.564f + 0.5f
        // Smooth distance to natural human skin YCrCb locus (Cr ~ 0.52..0.68, Cb ~ 0.30..0.49)
        val crDist = when {
            cr < 0.525f -> (0.525f - cr) / 0.04f
            cr > 0.675f -> (cr - 0.675f) / 0.04f
            else -> 0f
        }
        val cbDist = when {
            cb < 0.300f -> (0.300f - cb) / 0.04f
            cb > 0.495f -> (cb - 0.495f) / 0.04f
            else -> 0f
        }
        val locusOutside = max(crDist, cbDist).coerceIn(0f, 1f)
        val brightnessOk = if (y in 0.12f..0.92f) 1f else 0.4f
        val inSkinStrength = (1f - locusOutside) * brightnessOk
        // Retain 28% of boost on skin, 100% outside skin
        return (1f - inSkinStrength * 0.72f).coerceIn(0.28f, 1f)
    }

    /**
     * Enhances a single linear-8bit RGB pixel with White Balance, Contrast S-curve, Highlight Roll-off,
     * Brightness/Shadows lift, Selective Vibrance, Saturation, and Skin Protection.
     * Packed into ARGB int with original alpha preserved.
     */
    fun enhancePixelArgb(
        argb: Int,
        params: AdaptiveAdjustmentParams
    ): Int {
        val a = (argb ushr 24) and 0xFF
        val r0 = ((argb ushr 16) and 0xFF) / 255f
        val g0 = ((argb ushr 8) and 0xFF) / 255f
        val b0 = (argb and 0xFF) / 255f

        // 1. White balance correction
        var r = (r0 * params.wbGainR).coerceIn(0f, 1.25f)
        var g = (g0 * params.wbGainG).coerceIn(0f, 1.25f)
        var b = (b0 * params.wbGainB).coerceIn(0f, 1.25f)

        // 2. Luminance-domain shadow lift + contrast + highlight shoulder roll-off
        val lumIn = (0.299f * r + 0.587f * g + 0.114f * b).coerceAtLeast(1e-4f)
        var lumOut = lumIn
        if (params.shadowsLift > 0.002f) {
            val sW = (1f - lumOut).coerceIn(0f, 1f)
            lumOut += params.shadowsLift * 0.22f * sW * (1f - sW) * 4f
        }
        if (params.brillianceStrength > 0.002f) {
            val midW = (4f * lumOut * (1f - lumOut)).coerceIn(0f, 1f)
            lumOut += params.brillianceStrength * 0.14f * midW
        }
        lumOut = ((lumOut - 0.5f) * params.contrastGain + 0.5f) + (params.brightnessShift / 255f)
        if (params.exposureHighlightCompress > 0.002f && lumOut > 0.76f) {
            val excess = lumOut - 0.76f
            lumOut = 0.76f + excess * (1f - params.exposureHighlightCompress)
        }
        lumOut = lumOut.coerceIn(0f, 1f)
        val lumScale = lumOut / lumIn
        r *= lumScale
        g *= lumScale
        b *= lumScale

        // 3. Selective Vibrance + Skin-Protected Saturation
        val luma = 0.299f * r + 0.587f * g + 0.114f * b
        val maxC = max(r, max(g, b))
        val minC = min(r, min(g, b))
        val localSat = if (maxC > 1e-4f) ((maxC - minC) / maxC).coerceIn(0f, 1f) else 0f
        val skinFactor = if (params.skinToneProtection) skinNonMaskWeight(r, g, b) else 1f
        val vibranceTerm = params.vibranceGain * (1f - localSat).toDouble().pow(1.25).toFloat()
        val satTerm = (params.saturationGain - 1f) * (1f - 0.55f * localSat * localSat)
        val chromaGain = 1f + (vibranceTerm + satTerm) * skinFactor

        r = (luma + (r - luma) * chromaGain).coerceIn(0f, 1f)
        g = (luma + (g - luma) * chromaGain).coerceIn(0f, 1f)
        b = (luma + (b - luma) * chromaGain).coerceIn(0f, 1f)

        val rOut = (r * 255f).roundToInt().coerceIn(0, 255)
        val gOut = (g * 255f).roundToInt().coerceIn(0, 255)
        val bOut = (b * 255f).roundToInt().coerceIn(0, 255)
        return (a shl 24) or (rOut shl 16) or (gOut shl 8) or bOut
    }

    fun enhanceArgbFrameCopy(srcArgb: IntArray, params: AdaptiveAdjustmentParams): IntArray {
        return IntArray(srcArgb.size) { i -> enhancePixelArgb(srcArgb[i], params) }
    }

    /**
     * Builds a 256-entry centered-128 chroma LUT for YCrCb Cr/Cb channels that applies
     * selective Vibrance (strongest on low/mid chroma faded pixels) + saturation roll-off near gamut edges.
     */
    fun buildChromaVibranceLutArray(vibranceGain: Float, saturationGain: Float): ByteArray {
        val lut = ByteArray(256)
        val satDelta = (saturationGain - 1f).toDouble().coerceIn(-0.20, 0.45)
        val vibAmt = vibranceGain.toDouble().coerceIn(0.0, 0.85)
        for (i in 0..255) {
            val dev = i - 128.0
            val absNorm = (abs(dev) / 128.0).coerceIn(0.0, 1.0)
            // Faded colors (low absNorm) get full vibrance + saturation lift; near-clipped colors gently taper
            val selectiveBoost = vibAmt * (1.0 - absNorm).pow(1.25) + satDelta * (1.0 - 0.55 * absNorm * absNorm)
            val gain = (1.0 + selectiveBoost).coerceIn(0.75, 1.65)
            val out = (128.0 + dev * gain).roundToInt().coerceIn(0, 255)
            lut[i] = out.toByte()
        }
        return lut
    }
}

/**
 * Fast native OpenCV color mastering pass used whenever GLES 3.1 compute shader does not handle
 * the frame directly (e.g., neural SR frames without GLES post-pass, or headless/CPU devices).
 * Ensures White Balance, Tone/Brilliance LUT, Selective Vibrance, and Skin-Tone Protection
 * ALWAYS execute on every Clarity 1x-4x mode with zero skipping.
 */
class ColorEnhancementEngine {

    private var ycrcbMat: Mat? = null
    private var chromaFullLutMat: Mat? = null
    private var chromaSkinLutMat: Mat? = null
    private var cachedChromaKey: String = ""
    private var skinMaskMat: Mat? = null
    private var nonSkinMaskMat: Mat? = null
    private var crBoostMat: Mat? = null
    private var crSkinMat: Mat? = null
    private var cbBoostMat: Mat? = null
    private var cbSkinMat: Mat? = null

    fun applyFastColorEnhancementBgr(
        bgr: Mat,
        params: AdaptiveAdjustmentParams,
        applyToneLut: Boolean = true
    ) {
        if (bgr.empty()) return

        // 1. Auto White Balance
        if (abs(params.wbGainR - 1f) >= 0.008f || abs(params.wbGainB - 1f) >= 0.008f) {
            val bgrCh = ArrayList<Mat>(3)
            try {
                Core.split(bgr, bgrCh)
                if (abs(params.wbGainB - 1f) >= 0.008f) {
                    bgrCh[0].convertTo(bgrCh[0], CvType.CV_8UC1, params.wbGainB.toDouble(), 0.0)
                }
                if (abs(params.wbGainR - 1f) >= 0.008f) {
                    bgrCh[2].convertTo(bgrCh[2], CvType.CV_8UC1, params.wbGainR.toDouble(), 0.0)
                }
                Core.merge(bgrCh, bgr)
            } finally {
                bgrCh.forEach { it.release() }
            }
        }

        // 2. YCrCb Luma Tone LUT + Selective Skin-Protected Vibrance LUT
        val needChromaBoost = params.vibranceGain > 0.015f || abs(params.saturationGain - 1f) > 0.01f
        if (!applyToneLut && !needChromaBoost) return

        val ycrcb = ycrcbMat ?: Mat().also { ycrcbMat = it }
        val ch = ArrayList<Mat>(3)
        try {
            Imgproc.cvtColor(bgr, ycrcb, Imgproc.COLOR_BGR2YCrCb)
            Core.split(ycrcb, ch) // 0=Y, 1=Cr, 2=Cb

            if (applyToneLut) {
                val toneLutData = OpenCvFrameProcessor.buildToneLutArray(params)
                val toneLut = Mat(1, 256, CvType.CV_8UC1)
                try {
                    toneLut.put(0, 0, toneLutData)
                    Core.LUT(ch[0], toneLut, ch[0])
                } finally {
                    toneLut.release()
                }
            }

            if (needChromaBoost) {
                ensureChromaLuts(params.vibranceGain, params.saturationGain)
                val fullLut = chromaFullLutMat!!
                val skinLut = chromaSkinLutMat!!

                if (!params.skinToneProtection) {
                    Core.LUT(ch[1], fullLut, ch[1])
                    Core.LUT(ch[2], fullLut, ch[2])
                } else {
                    val skinMask = skinMaskMat ?: Mat().also { skinMaskMat = it }
                    val nonSkinMask = nonSkinMaskMat ?: Mat().also { nonSkinMaskMat = it }
                    val crB = crBoostMat ?: Mat().also { crBoostMat = it }
                    val crS = crSkinMat ?: Mat().also { crSkinMat = it }
                    val cbB = cbBoostMat ?: Mat().also { cbBoostMat = it }
                    val cbS = cbSkinMat ?: Mat().also { cbSkinMat = it }

                    // YCrCb human skin tone locus: Y=[30..235], Cr=[133..173], Cb=[77..127]
                    Core.inRange(ycrcb, Scalar(30.0, 133.0, 77.0), Scalar(235.0, 173.0, 127.0), skinMask)
                    Core.bitwise_not(skinMask, nonSkinMask)

                    Core.LUT(ch[1], fullLut, crB)
                    Core.LUT(ch[1], skinLut, crS)
                    Core.LUT(ch[2], fullLut, cbB)
                    Core.LUT(ch[2], skinLut, cbS)

                    crB.copyTo(ch[1], nonSkinMask)
                    crS.copyTo(ch[1], skinMask)
                    cbB.copyTo(ch[2], nonSkinMask)
                    cbS.copyTo(ch[2], skinMask)
                }
            }

            Core.merge(ch, ycrcb)
            Imgproc.cvtColor(ycrcb, bgr, Imgproc.COLOR_YCrCb2BGR)
        } finally {
            ch.forEach { it.release() }
        }
    }

    private fun ensureChromaLuts(vibranceGain: Float, satGain: Float) {
        val key = String.format(Locale.US, "%.3f|%.3f", vibranceGain, satGain)
        if (chromaFullLutMat != null && chromaSkinLutMat != null && cachedChromaKey == key) return

        val fullLut = chromaFullLutMat ?: Mat(1, 256, CvType.CV_8UC1).also { chromaFullLutMat = it }
        val skinLut = chromaSkinLutMat ?: Mat(1, 256, CvType.CV_8UC1).also { chromaSkinLutMat = it }

        fullLut.put(0, 0, ColorMath.buildChromaVibranceLutArray(vibranceGain, satGain))
        // Skin-shielded LUT receives 28% of boost so natural complexion never turns orange or flushed
        val skinSat = 1f + (satGain - 1f) * 0.28f
        val skinVib = vibranceGain * 0.28f
        skinLut.put(0, 0, ColorMath.buildChromaVibranceLutArray(skinVib, skinSat))
        cachedChromaKey = key
    }

    fun release() {
        ycrcbMat?.release(); ycrcbMat = null
        chromaFullLutMat?.release(); chromaFullLutMat = null
        chromaSkinLutMat?.release(); chromaSkinLutMat = null
        cachedChromaKey = ""
        skinMaskMat?.release(); skinMaskMat = null
        nonSkinMaskMat?.release(); nonSkinMaskMat = null
        crBoostMat?.release(); crBoostMat = null
        crSkinMat?.release(); crSkinMat = null
        cbBoostMat?.release(); cbBoostMat = null
        cbSkinMat?.release(); cbSkinMat = null
    }
}
