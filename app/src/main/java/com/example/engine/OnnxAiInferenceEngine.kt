package com.example.engine

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.os.Build
import com.example.model.DeviceCapabilities
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.ceil
import kotlin.math.min
import kotlin.math.roundToInt

class OnnxAiInferenceEngine(private val context: Context) {

    private val env: OrtEnvironment by lazy { OrtEnvironment.getEnvironment() }
    private var realEsrganSession: OrtSession? = null
    private var dncnnSession: OrtSession? = null
    private var activeProviderDescription: String = "ONNX CPU/XNNPACK"

    val isRealEsrganReady: Boolean get() = realEsrganSession != null
    val isDncnnReady: Boolean get() = dncnnSession != null
    val providerLabel: String get() = activeProviderDescription

    @Synchronized
    fun initialize(capabilities: DeviceCapabilities) {
        val modelDir = File(context.filesDir, "ai_models").apply { mkdirs() }
        val srFile = ensureAssetCopied("ai_models/realesrgan/realesr_general_x4v3.onnx", File(modelDir, "realesrgan/realesr_general_x4v3.onnx"))
        val dncnnOnnx = ensureAssetCopied("ai_models/denoise/dncnn.onnx", File(modelDir, "denoise/dncnn.onnx"))
        ensureAssetCopied("ai_models/denoise/dncnn.data", File(modelDir, "denoise/dncnn.data"))

        if (realEsrganSession == null && srFile.exists() && srFile.length() > 0) {
            realEsrganSession = createSession(srFile.absolutePath, capabilities)
        }
        if (dncnnSession == null && dncnnOnnx.exists() && dncnnOnnx.length() > 0) {
            dncnnSession = createSession(dncnnOnnx.absolutePath, capabilities)
        }
    }

    private fun createSession(modelPath: String, capabilities: DeviceCapabilities): OrtSession {
        val useNnapi = capabilities.nnapiSupported && capabilities.processingCapacityScore >= 55
        if (useNnapi) {
            try {
                val opts = OrtSession.SessionOptions().apply {
                    setIntraOpNumThreads(capabilities.inferenceThreads)
                    setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                    addNnapi()
                }
                activeProviderDescription = if (capabilities.vulkanSupported) {
                    "ONNX NNAPI + Vulkan/GPU"
                } else {
                    "ONNX NNAPI NPU/GPU"
                }
                return env.createSession(modelPath, opts)
            } catch (_: Exception) {
                // Fallback safely to CPU/XNNPACK if NNAPI rejects specific ops on a device
            }
        }

        val cpuOpts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(capabilities.inferenceThreads)
            setInterOpNumThreads(1)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        activeProviderDescription = "ONNX XNNPACK CPU (${capabilities.inferenceThreads}T)"
        return env.createSession(modelPath, cpuOpts)
    }

    /**
     * Runs genuine tiled Real-ESRGAN 4x AI super-resolution on a BGR Mat using overlapping tiles
     * with feathered cosine/linear margin blending, then resizes smoothly via INTER_AREA
     * to the exact target output dimensions (preserving aspect ratio without stretching).
     */
    fun upscaleBgrTiled(
        srcBgr: Mat,
        targetW: Int,
        targetH: Int,
        capabilities: DeviceCapabilities,
        isCancelled: () -> Boolean,
        onTileProgress: (done: Int, total: Int) -> Unit = { _, _ -> }
    ): Mat {
        val session = realEsrganSession
            ?: throw IllegalStateException("Real-ESRGAN ONNX model is not loaded.")

        val inW = srcBgr.cols()
        val inH = srcBgr.rows()
        val scale = 4 // realesr_general_x4v3 is 4x

        val tileSize = capabilities.adaptiveTileSize.coerceIn(64, 256)
        val overlap = capabilities.tileOverlap.coerceIn(4, tileSize / 4)
        val step = (tileSize - overlap * 2).coerceAtLeast(32)

        val srW = inW * scale
        val srH = inH * scale

        val outAccum = FloatArray(srW * srH * 3)
        val weightAccum = FloatArray(srW * srH)

        // Convert entire input frame once to RGB bytes
        val rgbMat = Mat()
        Imgproc.cvtColor(srcBgr, rgbMat, Imgproc.COLOR_BGR2RGB)
        val rgbBytes = ByteArray(inW * inH * 3)
        rgbMat.get(0, 0, rgbBytes)
        rgbMat.release()

        val xStarts = computeTileStarts(inW, tileSize, step)
        val yStarts = computeTileStarts(inH, tileSize, step)
        val totalTiles = xStarts.size * yStarts.size
        var tilesDone = 0

        val inputName = session.inputNames.first()

        for (y0 in yStarts) {
            val tileH = min(tileSize, inH - y0)
            for (x0 in xStarts) {
                if (isCancelled()) throw InterruptedException("Cancelled during AI super-resolution")
                val tileW = min(tileSize, inW - x0)

                val tilePixels = tileW * tileH
                val inputBuffer = FloatBuffer.allocate(3 * tilePixels)
                val inputArr = inputBuffer.array()

                // Pack NCHW RGB [0..1]
                for (ty in 0 until tileH) {
                    val srcRow = (y0 + ty) * inW
                    val dstRow = ty * tileW
                    for (tx in 0 until tileW) {
                        val sIdx = (srcRow + x0 + tx) * 3
                        val dIdx = dstRow + tx
                        inputArr[dIdx] = (rgbBytes[sIdx].toInt() and 0xFF) / 255f
                        inputArr[tilePixels + dIdx] = (rgbBytes[sIdx + 1].toInt() and 0xFF) / 255f
                        inputArr[2 * tilePixels + dIdx] = (rgbBytes[sIdx + 2].toInt() and 0xFF) / 255f
                    }
                }
                inputBuffer.rewind()

                val shape = longArrayOf(1L, 3L, tileH.toLong(), tileW.toLong())
                val tensor = OnnxTensor.createTensor(env, inputBuffer, shape)
                val outTileW = tileW * scale
                val outTileH = tileH * scale
                val outTilePixels = outTileW * outTileH

                try {
                    session.run(mapOf(inputName to tensor)).use { results ->
                        val outTensor = results[0] as OnnxTensor
                        val outBuf = outTensor.floatBuffer
                        val outArr = FloatArray(3 * outTilePixels)
                        outBuf.get(outArr)

                        val outX0 = x0 * scale
                        val outY0 = y0 * scale
                        val feather = (overlap * scale).coerceAtLeast(4)

                        for (oy in 0 until outTileH) {
                            val wy = featherWeight(oy, outTileH, y0 == 0, y0 + tileH >= inH, feather)
                            val dstY = outY0 + oy
                            for (ox in 0 until outTileW) {
                                val wx = featherWeight(ox, outTileW, x0 == 0, x0 + tileW >= inW, feather)
                                val w = wx * wy
                                val sIdx = oy * outTileW + ox
                                val dstPixel = dstY * srW + (outX0 + ox)
                                val dstBase = dstPixel * 3

                                outAccum[dstBase] += outArr[sIdx].coerceIn(0f, 1f) * w
                                outAccum[dstBase + 1] += outArr[outTilePixels + sIdx].coerceIn(0f, 1f) * w
                                outAccum[dstBase + 2] += outArr[2 * outTilePixels + sIdx].coerceIn(0f, 1f) * w
                                weightAccum[dstPixel] += w
                            }
                        }
                    }
                } finally {
                    tensor.close()
                }

                tilesDone++
                onTileProgress(tilesDone, totalTiles)
            }
        }

        // Normalize blended accumulator back into RGB byte array -> BGR Mat
        val srBytes = ByteArray(srW * srH * 3)
        for (i in 0 until (srW * srH)) {
            val w = weightAccum[i].coerceAtLeast(1e-5f)
            val b = i * 3
            val r = ((outAccum[b] / w) * 255f).roundToInt().coerceIn(0, 255).toByte()
            val g = ((outAccum[b + 1] / w) * 255f).roundToInt().coerceIn(0, 255).toByte()
            val bl = ((outAccum[b + 2] / w) * 255f).roundToInt().coerceIn(0, 255).toByte()
            // Store directly as BGR for OpenCV
            srBytes[b] = bl
            srBytes[b + 1] = g
            srBytes[b + 2] = r
        }

        val srBgr = Mat(srH, srW, CvType.CV_8UC3)
        srBgr.put(0, 0, srBytes)

        if (srW == targetW && srH == targetH) {
            return srBgr
        }

        val finalMat = Mat()
        val interp = if (srW > targetW) Imgproc.INTER_AREA else Imgproc.INTER_LANCZOS4
        Imgproc.resize(srBgr, finalMat, Size(targetW.toDouble(), targetH.toDouble()), 0.0, 0.0, interp)
        srBgr.release()
        return finalMat
    }

    /**
     * Runs genuine DnCNN ONNX denoising on the Y (luminance) channel using 256x256 tiles,
     * blending the AI-denoised luma with the original Y channel according to blendAlpha
     * to preserve natural film/skin texture without plastic smoothing.
     */
    fun denoiseLumaChannel(
        yChannel: Mat,
        blendAlpha: Float,
        isCancelled: () -> Boolean
    ) {
        val session = dncnnSession ?: return
        val alpha = blendAlpha.coerceIn(0f, 0.90f)
        if (alpha <= 0.05f) return

        val w = yChannel.cols()
        val h = yChannel.rows()
        val srcBytes = ByteArray(w * h)
        yChannel.get(0, 0, srcBytes)

        val tileSize = 256
        val step = 224
        val xStarts = computeTileStarts(w, min(tileSize, w), step)
        val yStarts = computeTileStarts(h, min(tileSize, h), step)

        val accum = FloatArray(w * h)
        val weight = FloatArray(w * h)
        val inputName = session.inputNames.first()
        val tileBuf = FloatArray(tileSize * tileSize)

        for (y0 in yStarts) {
            val copyH = min(tileSize, h - y0)
            for (x0 in xStarts) {
                if (isCancelled()) throw InterruptedException("Cancelled during AI denoising")
                val copyW = min(tileSize, w - x0)

                // Fill 256x256 fixed input tensor with edge-clamped reflection/border
                for (ty in 0 until tileSize) {
                    val sy = y0 + ty.coerceAtMost(copyH - 1)
                    val rowOff = sy * w
                    val dstOff = ty * tileSize
                    for (tx in 0 until tileSize) {
                        val sx = x0 + tx.coerceAtMost(copyW - 1)
                        tileBuf[dstOff + tx] = (srcBytes[rowOff + sx].toInt() and 0xFF) / 255f
                    }
                }

                val tensor = OnnxTensor.createTensor(
                    env,
                    FloatBuffer.wrap(tileBuf),
                    longArrayOf(1L, 1L, tileSize.toLong(), tileSize.toLong())
                )
                try {
                    session.run(mapOf(inputName to tensor)).use { results ->
                        val outTensor = results[0] as OnnxTensor
                        val outArr = FloatArray(tileSize * tileSize)
                        outTensor.floatBuffer.get(outArr)

                        for (ty in 0 until copyH) {
                            val wy = featherWeight(ty, copyH, y0 == 0, y0 + copyH >= h, 16)
                            val dy = y0 + ty
                            for (tx in 0 until copyW) {
                                val wx = featherWeight(tx, copyW, x0 == 0, x0 + copyW >= w, 16)
                                val wt = wx * wy
                                val idx = dy * w + (x0 + tx)
                                accum[idx] += outArr[ty * tileSize + tx].coerceIn(0f, 1f) * wt
                                weight[idx] += wt
                            }
                        }
                    }
                } finally {
                    tensor.close()
                }
            }
        }

        for (i in 0 until (w * h)) {
            val orig = (srcBytes[i].toInt() and 0xFF) / 255f
            val denoised = accum[i] / weight[i].coerceAtLeast(1e-5f)
            val blended = orig * (1f - alpha) + denoised * alpha
            srcBytes[i] = (blended * 255f).roundToInt().coerceIn(0, 255).toByte()
        }
        yChannel.put(0, 0, srcBytes)
    }

    private fun computeTileStarts(totalLen: Int, tileLen: Int, step: Int): List<Int> {
        if (totalLen <= tileLen) return listOf(0)
        val starts = mutableListOf<Int>()
        var pos = 0
        while (pos + tileLen < totalLen) {
            starts.add(pos)
            pos += step
        }
        val last = (totalLen - tileLen).coerceAtLeast(0)
        if (starts.isEmpty() || starts.last() != last) {
            starts.add(last)
        }
        return starts
    }

    private fun featherWeight(
        pos: Int,
        len: Int,
        isFirst: Boolean,
        isLast: Boolean,
        feather: Int
    ): Float {
        var w = 1f
        if (!isFirst && pos < feather) {
            w = min(w, (pos + 1).toFloat() / feather.toFloat())
        }
        if (!isLast && pos >= len - feather) {
            w = min(w, (len - pos).toFloat() / feather.toFloat())
        }
        return w.coerceIn(0.05f, 1f)
    }

    private fun ensureAssetCopied(assetPath: String, destFile: File): File {
        destFile.parentFile?.mkdirs()
        if (!destFile.exists() || destFile.length() == 0L) {
            context.assets.open(assetPath).use { input ->
                destFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        }
        return destFile
    }

    fun close() {
        try {
            realEsrganSession?.close()
            dncnnSession?.close()
        } catch (_: Exception) {
        }
        realEsrganSession = null
        dncnnSession = null
    }
}
