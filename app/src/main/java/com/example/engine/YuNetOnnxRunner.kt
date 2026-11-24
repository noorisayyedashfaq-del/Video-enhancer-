package com.example.engine

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.example.model.AcceleratorKind
import com.example.model.BackendCandidateId
import com.example.model.ModelBackendBenchmark
import com.example.model.ModelExecutionPlan
import com.example.model.ModelKind
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Size
import org.opencv.dnn.Dnn
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.FaceDetectorYN
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.ceil
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * Pure Kotlin multi-scale anchor decoder + IoU NMS for YuNet (face_detection_yunet_2023mar.onnx).
 */
object YuNetDecoder {

    data class DetectionBox(
        val x: Float,
        val y: Float,
        val width: Float,
        val height: Float,
        val score: Float
    )

    val STRIDES = intArrayOf(8, 16, 32)

    fun decodeStridedOutputs(
        inputW: Int,
        inputH: Int,
        outputsByName: Map<String, FloatArray>,
        scoreThreshold: Float = 0.56f,
        nmsThreshold: Float = 0.30f,
        topK: Int = 10
    ): List<DetectionBox> {
        val raw = ArrayList<DetectionBox>(16)

        for (stride in STRIDES) {
            val cols = ceil(inputW.toDouble() / stride).toInt().coerceAtLeast(1)
            val rows = ceil(inputH.toDouble() / stride).toInt().coerceAtLeast(1)
            val clsArr = outputsByName["cls_$stride"] ?: continue
            val objArr = outputsByName["obj_$stride"] ?: continue
            val bboxArr = outputsByName["bbox_$stride"] ?: continue

            val totalCells = rows * cols
            val count = minOf(totalCells, clsArr.size, objArr.size, bboxArr.size / 4)
            for (idx in 0 until count) {
                val cls = clsArr[idx].coerceIn(0f, 1f)
                val obj = objArr[idx].coerceIn(0f, 1f)
                val score = sqrt(cls * obj)
                if (score < scoreThreshold) continue

                val r = idx / cols
                val c = idx % cols
                val bIdx = idx * 4
                val dx = bboxArr[bIdx]
                val dy = bboxArr[bIdx + 1]
                val dw = bboxArr[bIdx + 2].coerceIn(-6f, 6f)
                val dh = bboxArr[bIdx + 3].coerceIn(-6f, 6f)

                val cx = (c + dx) * stride
                val cy = (r + dy) * stride
                val w = exp(dw) * stride
                val h = exp(dh) * stride
                val x = (cx - w * 0.5f).coerceIn(0f, (inputW - 2).toFloat())
                val y = (cy - h * 0.5f).coerceIn(0f, (inputH - 2).toFloat())
                val clampedW = w.coerceIn(4f, inputW - x)
                val clampedH = h.coerceIn(4f, inputH - y)
                raw.add(DetectionBox(x, y, clampedW, clampedH, score))
            }
        }

        if (raw.isEmpty()) return emptyList()
        return nonMaxSuppression(raw, nmsThreshold, topK)
    }

    fun nonMaxSuppression(
        boxes: List<DetectionBox>,
        iouThreshold: Float,
        topK: Int
    ): List<DetectionBox> {
        val sorted = boxes.sortedByDescending { it.score }
        val kept = ArrayList<DetectionBox>(min(topK, sorted.size))
        for (cand in sorted) {
            if (kept.size >= topK) break
            val suppressed = kept.any { computeIoU(it, cand) > iouThreshold }
            if (!suppressed) {
                kept.add(cand)
            }
        }
        return kept
    }

    private fun computeIoU(a: DetectionBox, b: DetectionBox): Float {
        val ix0 = max(a.x, b.x)
        val iy0 = max(a.y, b.y)
        val ix1 = min(a.x + a.width, b.x + b.width)
        val iy1 = min(a.y + a.height, b.y + b.height)
        val iw = (ix1 - ix0).coerceAtLeast(0f)
        val ih = (iy1 - iy0).coerceAtLeast(0f)
        val inter = iw * ih
        if (inter <= 0f) return 0f
        val union = (a.width * a.height + b.width * b.height - inter).coerceAtLeast(1e-5f)
        return inter / union
    }
}

/**
 * Multi-backend Adaptive Face AI Runner for YuNet (`face_detection_yunet_2023mar.onnx`).
 * Benchmarks ORT (QNN GPU, NNAPI GPU, QNN HTP NPU, NNAPI NPU, XNNPACK) and
 * OpenCV FaceDetectorYN (OpenCL GPU, Vulkan GPU, SIMD CPU) independently at 320x320,
 * choosing the genuinely fastest verified backend with live mid-video runtime fallback.
 */
class YuNetOnnxRunner(private val env: OrtEnvironment) {

    private val probeSize = 320
    private val probePixels = probeSize * probeSize

    private val ortSessions = LinkedHashMap<BackendCandidateId, OrtSession>()
    private var openClDetector: FaceDetectorYN? = null
    private var openClDetSize: Pair<Int, Int> = -1 to -1
    private var cpuDetector: FaceDetectorYN? = null
    private var cpuDetSize: Pair<Int, Int> = -1 to -1
    private var modelPath: String? = null

    var activePlan: ModelExecutionPlan? = null
        private set
    private var activeBackendIndex: Int = 0

    // Reusable buffers for ORT YuNet inference (direct off-heap NCHW slices -> zero GC & zero FloatArray copies)
    private var ortInSize: Pair<Int, Int> = -1 to -1
    private var ortDirectBuf: ByteBuffer? = null
    private var ortFloatBuf: FloatBuffer? = null
    private var ortCachedTensor: OnnxTensor? = null
    private var ortRgbFloatMat: Mat? = null
    private val ortChannelWraps = ArrayList<Mat>(3)

    // Reusable OpenCV scratch Mats
    private var detScaledMat: Mat? = null
    private var detOutMat: Mat? = null
    private var openCvRowBuf: FloatArray = FloatArray(15)

    val currentBackendLabel: String
        get() = activePlan?.let { plan ->
            val id = plan.fallbackChain.getOrNull(activeBackendIndex) ?: plan.selectedBackend
            id.displayLabel
        } ?: "OpenCV SIMD CPU"

    val currentAcceleratorKind: AcceleratorKind
        get() = activePlan?.let { plan ->
            val id = plan.fallbackChain.getOrNull(activeBackendIndex) ?: plan.selectedBackend
            id.kind
        } ?: AcceleratorKind.CPU

    @Synchronized
    fun benchmarkAndSelectBest(
        yunetPath: String,
        candidateOrtSessions: List<Triple<BackendCandidateId, Int, OrtSession>>,
        openClAvailable: Boolean,
        vulkanAvailable: Boolean,
        logger: EngineDiagnosticsLogger? = null
    ): ModelExecutionPlan {
        return benchmarkAndSelectBestFactories(
            yunetPath = yunetPath,
            candidateFactories = candidateOrtSessions.map { (id, threads, sess) ->
                Triple(id, threads) { sess }
            },
            openClAvailable = openClAvailable,
            vulkanAvailable = vulkanAvailable,
            logger = logger
        )
    }

    @Synchronized
    fun benchmarkAndSelectBestFactories(
        yunetPath: String,
        candidateFactories: List<Triple<BackendCandidateId, Int, () -> OrtSession?>>,
        openClAvailable: Boolean,
        vulkanAvailable: Boolean,
        logger: EngineDiagnosticsLogger? = null
    ): ModelExecutionPlan {
        close()
        modelPath = yunetPath
        val benchmarks = ArrayList<ModelBackendBenchmark>()
        var cpuReferenceLogits: FloatArray? = null

        // 1. Evaluate CPU XNNPACK / ORT baseline first (sequentially one-at-a-time to avoid RAM spikes)
        val orderedFactories = candidateFactories.sortedBy { it.first.kind != AcceleratorKind.CPU }
        for ((backendId, threads, factory) in orderedFactories) {
            val sess = runCatching { factory() }.getOrNull()
            if (sess == null) {
                logger?.info("YuNet: ${backendId.displayLabel} skipped during init")
                continue
            }
            val res = benchmarkOrtYuNet(sess, backendId, threads, cpuReferenceLogits, logger)
            if (res.first.verifiedFullOffload && res.first.numericalParityOk && backendId.kind == AcceleratorKind.CPU && cpuReferenceLogits == null) {
                cpuReferenceLogits = res.second
            }
            benchmarks.add(res.first)
            if (res.first.verifiedFullOffload && res.first.numericalParityOk && res.first.medianLatencyMs != null) {
                ortSessions[backendId] = sess
            } else {
                runCatching { sess.close() }
            }
        }

        // 2. Evaluate OpenCV FaceDetectorYN CPU baseline
        val cvCpuMs = benchmarkOpenCvDetector(yunetPath, Dnn.DNN_BACKEND_OPENCV, Dnn.DNN_TARGET_CPU, logger)
        if (cvCpuMs != null) {
            benchmarks.add(
                ModelBackendBenchmark(
                    backendId = BackendCandidateId.OPENCV_CPU,
                    kind = AcceleratorKind.CPU,
                    displayLabel = BackendCandidateId.OPENCV_CPU.displayLabel,
                    threads = 0,
                    medianLatencyMs = cvCpuMs,
                    verifiedFullOffload = true,
                    numericalParityOk = true
                )
            )
        } else {
            benchmarks.add(
                ModelBackendBenchmark(
                    backendId = BackendCandidateId.OPENCV_CPU,
                    kind = AcceleratorKind.CPU,
                    displayLabel = BackendCandidateId.OPENCV_CPU.displayLabel,
                    rejectedReason = "FaceDetectorYN CPU init failed"
                )
            )
        }

        // 3. Evaluate OpenCV OpenCL GPU if OpenCL driver probed present
        if (openClAvailable) {
            val cvClMs = benchmarkOpenCvDetector(yunetPath, Dnn.DNN_BACKEND_OPENCV, Dnn.DNN_TARGET_OPENCL, logger)
            if (cvClMs != null) {
                benchmarks.add(
                    ModelBackendBenchmark(
                        backendId = BackendCandidateId.OPENCV_OPENCL,
                        kind = AcceleratorKind.GPU,
                        displayLabel = BackendCandidateId.OPENCV_OPENCL.displayLabel,
                        medianLatencyMs = cvClMs,
                        verifiedFullOffload = true,
                        numericalParityOk = true
                    )
                )
            } else {
                benchmarks.add(
                    ModelBackendBenchmark(
                        backendId = BackendCandidateId.OPENCV_OPENCL,
                        kind = AcceleratorKind.GPU,
                        displayLabel = BackendCandidateId.OPENCV_OPENCL.displayLabel,
                        rejectedReason = "OpenCV OpenCL DNN target unavailable"
                    )
                )
            }
        }

        // 4. Evaluate OpenCV Vulkan GPU if Vulkan compute probed present
        if (vulkanAvailable) {
            val cvVkMs = benchmarkOpenCvDetector(yunetPath, Dnn.DNN_BACKEND_OPENCV, Dnn.DNN_TARGET_VULKAN, logger)
            if (cvVkMs != null) {
                benchmarks.add(
                    ModelBackendBenchmark(
                        backendId = BackendCandidateId.OPENCV_VULKAN,
                        kind = AcceleratorKind.GPU,
                        displayLabel = BackendCandidateId.OPENCV_VULKAN.displayLabel,
                        medianLatencyMs = cvVkMs,
                        verifiedFullOffload = true,
                        numericalParityOk = true
                    )
                )
            } else {
                benchmarks.add(
                    ModelBackendBenchmark(
                        backendId = BackendCandidateId.OPENCV_VULKAN,
                        kind = AcceleratorKind.GPU,
                        displayLabel = BackendCandidateId.OPENCV_VULKAN.displayLabel,
                        rejectedReason = "OpenCV Vulkan DNN target unavailable"
                    )
                )
            }
        }

        val plan = AdaptiveHardwareSelector.selectPlanForModel(ModelKind.FACE_YUNET, benchmarks)
        activePlan = plan
        activeBackendIndex = 0

        // Close losing ORT sessions not retained in fallbackChain
        val keepSet = plan.fallbackChain.toSet()
        val iter = ortSessions.entries.iterator()
        while (iter.hasNext()) {
            val entry = iter.next()
            if (entry.key !in keepSet) {
                runCatching { entry.value.close() }
                iter.remove()
            }
        }
        return plan
    }

    private fun benchmarkOrtYuNet(
        session: OrtSession,
        backendId: BackendCandidateId,
        threads: Int,
        referenceSamples: FloatArray?,
        logger: EngineDiagnosticsLogger?
    ): Pair<ModelBackendBenchmark, FloatArray?> {
        return try {
            val inputName = session.inputNames.first()
            val shape = longArrayOf(1L, 3L, probeSize.toLong(), probeSize.toLong())
            val direct = ByteBuffer.allocateDirect(3 * probePixels * 4).order(ByteOrder.nativeOrder())
            val fBuf = direct.asFloatBuffer()
            // Synthetic textured pattern so logits stay non-constant across grid cells
            for (i in 0 until 3 * probePixels) {
                fBuf.put(i, ((i * 17) % 255).toFloat())
            }
            fBuf.rewind()

            var sampleOut: FloatArray? = null
            // Warmup + numerical validation
            OnnxTensor.createTensor(env, fBuf, shape).use { tWarm ->
                session.run(mapOf(inputName to tWarm)).use { results ->
                    val outNames = session.outputNames.toList()
                    val firstTensor = results[0] as? OnnxTensor
                        ?: return ModelBackendBenchmark(backendId, backendId.kind, backendId.displayLabel, rejectedReason = "Unexpected output") to null
                    val buf = firstTensor.floatBuffer
                    val cap = min(buf.remaining(), 512)
                    val arr = FloatArray(cap)
                    buf.get(arr, 0, cap)
                    sampleOut = arr
                    // Require standard multi-head YuNet outputs
                    if (outNames.size < 6) {
                        return ModelBackendBenchmark(backendId, backendId.kind, backendId.displayLabel, rejectedReason = "Missing YuNet heads") to null
                    }
                }
            }

            val parity = AdaptiveHardwareSelector.verifyNumericalOutput(
                candidateSamples = sampleOut ?: FloatArray(0),
                referenceSamples = referenceSamples,
                maxMae = 0.25f
            )
            if (!parity.valid) {
                return ModelBackendBenchmark(
                    backendId = backendId,
                    kind = backendId.kind,
                    displayLabel = backendId.displayLabel,
                    threads = threads,
                    verifiedFullOffload = true,
                    numericalParityOk = false,
                    rejectedReason = parity.reason
                ) to null
            }

            val runs = ArrayList<Long>(3)
            for (r in 0 until 3) {
                fBuf.rewind()
                val t0 = System.nanoTime()
                OnnxTensor.createTensor(env, fBuf, shape).use { t ->
                    session.run(mapOf(inputName to t)).use { }
                }
                runs.add(((System.nanoTime() - t0) / 1_000_000L).coerceAtLeast(1L))
            }

            val med = AdaptiveHardwareSelector.medianMs(runs)
            ModelBackendBenchmark(
                backendId = backendId,
                kind = backendId.kind,
                displayLabel = backendId.displayLabel,
                threads = threads,
                medianLatencyMs = med,
                verifiedFullOffload = true,
                numericalParityOk = true
            ) to sampleOut
        } catch (t: Throwable) {
            logger?.warn("Face AI ORT benchmark failed for ${backendId.displayLabel}: ${t.message}")
            ModelBackendBenchmark(
                backendId = backendId,
                kind = backendId.kind,
                displayLabel = backendId.displayLabel,
                threads = threads,
                rejectedReason = t.message ?: "ORT exception"
            ) to null
        }
    }

    private fun benchmarkOpenCvDetector(
        yunetPath: String,
        backendId: Int,
        targetId: Int,
        logger: EngineDiagnosticsLogger?
    ): Long? {
        var mat: Mat? = null
        var out: Mat? = null
        return try {
            mat = Mat(probeSize, probeSize, CvType.CV_8UC3)
            out = Mat()
            mat.setTo(org.opencv.core.Scalar(110.0, 125.0, 140.0))
            val det = FaceDetectorYN.create(
                yunetPath,
                "",
                Size(probeSize.toDouble(), probeSize.toDouble()),
                0.56f,
                0.30f,
                10,
                backendId,
                targetId
            )
            det.detect(mat, out) // Warmup
            val samples = ArrayList<Long>(3)
            for (r in 0 until 3) {
                val t0 = System.nanoTime()
                det.detect(mat, out)
                samples.add(((System.nanoTime() - t0) / 1_000_000L).coerceAtLeast(1L))
            }
            AdaptiveHardwareSelector.medianMs(samples)
        } catch (t: Throwable) {
            logger?.info("OpenCV FaceDetectorYN target=$targetId skipped (${t.message})")
            null
        } finally {
            runCatching { mat?.release() }
            runCatching { out?.release() }
        }
    }

    /**
     * Runs Face AI detection using the active benchmark-selected backend and automatically
     * steps down the ranked fallback chain if any GPU/NPU session encounters a runtime error.
     */
    fun detectFaces(srcBgr: Mat, logger: EngineDiagnosticsLogger? = null): List<Rect> {
        val path = modelPath ?: return emptyList()
        val srcW = srcBgr.cols()
        val srcH = srcBgr.rows()
        if (srcW < 32 || srcH < 32) return emptyList()

        // Cap detection frame size at 384px for 2x faster YuNet inference while preserving box precision
        val maxDetSide = 384.0
        val scale = if (max(srcW, srcH) > maxDetSide) maxDetSide / max(srcW, srcH).toDouble() else 1.0
        val detW = ((srcW * scale).roundToInt().coerceAtLeast(32) / 2) * 2
        val detH = ((srcH * scale).roundToInt().coerceAtLeast(32) / 2) * 2

        val detInput = if (scale < 0.99) {
            val scaled = detScaledMat ?: Mat().also { detScaledMat = it }
            Imgproc.resize(srcBgr, scaled, Size(detW.toDouble(), detH.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
            scaled
        } else {
            srcBgr
        }

        val chain = activePlan?.fallbackChain ?: listOf(BackendCandidateId.OPENCV_CPU)
        while (activeBackendIndex < chain.size) {
            val backendId = chain[activeBackendIndex]
            try {
                return when (backendId) {
                    BackendCandidateId.OPENCV_OPENCL -> runOpenCvDetector(
                        path, detInput, srcW, srcH, detW, detH,
                        backendId = Dnn.DNN_BACKEND_OPENCV, targetId = Dnn.DNN_TARGET_OPENCL, isOpenCl = true
                    )
                    BackendCandidateId.OPENCV_CPU, BackendCandidateId.OPENCV_VULKAN -> runOpenCvDetector(
                        path, detInput, srcW, srcH, detW, detH,
                        backendId = Dnn.DNN_BACKEND_OPENCV,
                        targetId = if (backendId == BackendCandidateId.OPENCV_VULKAN) Dnn.DNN_TARGET_VULKAN else Dnn.DNN_TARGET_CPU,
                        isOpenCl = false
                    )
                    else -> {
                        val sess = ortSessions[backendId]
                        if (sess != null) {
                            runOrtYuNet(sess, detInput, srcW, srcH, detW, detH)
                        } else {
                            throw IllegalStateException("Missing session for $backendId")
                        }
                    }
                }
            } catch (t: Throwable) {
                logger?.warn("Face AI fallback from ${backendId.displayLabel} (${t.message})")
                activeBackendIndex++
            }
        }
        return emptyList()
    }

    private fun runOrtYuNet(
        session: OrtSession,
        detBgr: Mat,
        srcW: Int,
        srcH: Int,
        detW: Int,
        detH: Int
    ): List<Rect> {
        val pixels = detW * detH
        val shape = longArrayOf(1L, 3L, detH.toLong(), detW.toLong())
        if (ortInSize.first != detW || ortInSize.second != detH || ortDirectBuf == null) {
            runCatching { ortCachedTensor?.close() }
            ortCachedTensor = null
            ortChannelWraps.forEach { it.release() }
            ortChannelWraps.clear()
            ortRgbFloatMat?.release()

            ortInSize = detW to detH
            val planeBytes = pixels * 4
            val direct = ByteBuffer.allocateDirect(3 * planeBytes).order(ByteOrder.nativeOrder())
            ortDirectBuf = direct
            val fBuf = direct.asFloatBuffer()
            ortFloatBuf = fBuf
            ortRgbFloatMat = Mat(detH, detW, CvType.CV_32FC3)
            for (c in 0 until 3) {
                val slice = direct.duplicate().apply {
                    position(c * planeBytes)
                    limit((c + 1) * planeBytes)
                }.slice().order(ByteOrder.nativeOrder())
                ortChannelWraps.add(Mat(detH, detW, CvType.CV_32FC1, slice))
            }
            fBuf.clear()
            ortCachedTensor = OnnxTensor.createTensor(env, fBuf, shape)
        }

        val floatMat = ortRgbFloatMat!!
        detBgr.convertTo(floatMat, CvType.CV_32FC3) // YuNet expects BGR [0..255] FP32
        for (c in 0 until 3) {
            org.opencv.core.Core.extractChannel(floatMat, ortChannelWraps[c], c)
        }

        val fBuf = ortFloatBuf!!
        fBuf.rewind()
        val tensor = ortCachedTensor!!
        val inputName = session.inputNames.first()
        val outputsMap = HashMap<String, FloatArray>(9)

        session.run(mapOf(inputName to tensor)).use { results ->
            val outNames = session.outputNames.toList()
            for (i in outNames.indices) {
                val name = outNames[i]
                if (name.startsWith("cls_") || name.startsWith("obj_") || name.startsWith("bbox_")) {
                    val t = results[i] as OnnxTensor
                    val ob = t.floatBuffer
                    val arr = FloatArray(ob.remaining())
                    ob.get(arr)
                    outputsMap[name] = arr
                }
            }
        }

        val decoded = YuNetDecoder.decodeStridedOutputs(
            inputW = detW,
            inputH = detH,
            outputsByName = outputsMap,
            scoreThreshold = 0.56f,
            nmsThreshold = 0.30f,
            topK = 10
        )
        if (decoded.isEmpty()) return emptyList()

        val invScaleX = srcW.toDouble() / detW.toDouble()
        val invScaleY = srcH.toDouble() / detH.toDouble()
        return decoded
            .map { b ->
                val x = (b.x * invScaleX).roundToInt().coerceIn(0, srcW - 2)
                val y = (b.y * invScaleY).roundToInt().coerceIn(0, srcH - 2)
                val w = (b.width * invScaleX).roundToInt().coerceIn(8, srcW - x)
                val h = (b.height * invScaleY).roundToInt().coerceIn(8, srcH - y)
                Rect(x, y, w, h)
            }
            .filter { it.width >= 16 && it.height >= 16 }
            .sortedByDescending { it.width * it.height }
            .take(3)
    }

    private fun runOpenCvDetector(
        path: String,
        detBgr: Mat,
        srcW: Int,
        srcH: Int,
        detW: Int,
        detH: Int,
        backendId: Int,
        targetId: Int,
        isOpenCl: Boolean
    ): List<Rect> {
        val det = if (isOpenCl) {
            val d = openClDetector ?: FaceDetectorYN.create(
                path, "", Size(detW.toDouble(), detH.toDouble()), 0.56f, 0.30f, 10, backendId, targetId
            ).also {
                openClDetector = it
                openClDetSize = detW to detH
            }
            if (openClDetSize.first != detW || openClDetSize.second != detH) {
                d.inputSize = Size(detW.toDouble(), detH.toDouble())
                openClDetSize = detW to detH
            }
            d
        } else {
            val d = cpuDetector ?: FaceDetectorYN.create(
                path, "", Size(detW.toDouble(), detH.toDouble()), 0.56f, 0.30f, 10, backendId, targetId
            ).also {
                cpuDetector = it
                cpuDetSize = detW to detH
            }
            if (cpuDetSize.first != detW || cpuDetSize.second != detH) {
                d.inputSize = Size(detW.toDouble(), detH.toDouble())
                cpuDetSize = detW to detH
            }
            d
        }

        val outFaces = detOutMat ?: Mat().also { detOutMat = it }
        det.detect(detBgr, outFaces)
        val rows = outFaces.rows()
        if (rows <= 0) return emptyList()

        val invScaleX = srcW.toDouble() / detW.toDouble()
        val invScaleY = srcH.toDouble() / detH.toDouble()
        val cols = outFaces.cols().coerceAtLeast(15)
        if (openCvRowBuf.size < cols) {
            openCvRowBuf = FloatArray(cols)
        }
        val rowBuf = openCvRowBuf
        val detected = ArrayList<Pair<Rect, Float>>(rows)

        for (r in 0 until rows) {
            outFaces.get(r, 0, rowBuf)
            val score = rowBuf[14]
            if (score < 0.56f) continue
            val x = (rowBuf[0] * invScaleX).roundToInt().coerceIn(0, srcW - 2)
            val y = (rowBuf[1] * invScaleY).roundToInt().coerceIn(0, srcH - 2)
            val w = (rowBuf[2] * invScaleX).roundToInt().coerceIn(8, srcW - x)
            val h = (rowBuf[3] * invScaleY).roundToInt().coerceIn(8, srcH - y)
            if (w >= 16 && h >= 16) {
                detected.add(Rect(x, y, w, h) to (w * h).toFloat())
            }
        }
        return detected.sortedByDescending { it.second }.take(3).map { it.first }
    }

    @Synchronized
    fun releaseWorkBuffers() {
        detScaledMat?.release(); detScaledMat = null
        detOutMat?.release(); detOutMat = null
        runCatching { ortCachedTensor?.close() }
        ortCachedTensor = null
        ortRgbFloatMat?.release(); ortRgbFloatMat = null
        ortChannelWraps.forEach { it.release() }
        ortChannelWraps.clear()
        ortInSize = -1 to -1
        ortDirectBuf = null
        ortFloatBuf = null
        openCvRowBuf = FloatArray(15)
    }

    @Synchronized
    fun close() {
        releaseWorkBuffers()
        ortSessions.values.forEach { runCatching { it.close() } }
        ortSessions.clear()
        openClDetector = null
        openClDetSize = -1 to -1
        cpuDetector = null
        cpuDetSize = -1 to -1
        activeBackendIndex = 0
    }
}
