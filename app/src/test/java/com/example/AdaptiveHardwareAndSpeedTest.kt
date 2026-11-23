package com.example

import com.example.engine.AdaptiveHardwareSelector
import com.example.engine.EngineDiagnosticsLogger
import com.example.engine.FastYuvCopyMath
import com.example.engine.TemporalReusePolicy
import com.example.engine.YuvFastLayout
import com.example.engine.YuNetDecoder
import com.example.model.AcceleratorKind
import com.example.model.AdaptiveHardwareProfile
import com.example.model.BackendCandidateId
import com.example.model.CpuTopologyInfo
import com.example.model.HardwareProbeSnapshot
import com.example.model.ModelBackendBenchmark
import com.example.model.ModelKind
import com.example.model.ThermalState
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.nio.ByteBuffer
import kotlin.math.ceil

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [36])
class AdaptiveHardwareAndSpeedTest {

    @Test
    fun `selector picks NPU when benchmarked faster than GPU by more than 10 percent`() {
        val evaluated = listOf(
            ModelBackendBenchmark(
                backendId = BackendCandidateId.NNAPI_GPU,
                kind = AcceleratorKind.GPU,
                displayLabel = "NNAPI GPU",
                medianLatencyMs = 45L,
                verifiedFullOffload = true,
                numericalParityOk = true
            ),
            ModelBackendBenchmark(
                backendId = BackendCandidateId.QNN_HTP,
                kind = AcceleratorKind.NPU,
                displayLabel = "Qualcomm QNN HTP NPU",
                medianLatencyMs = 24L,
                verifiedFullOffload = true,
                numericalParityOk = true
            ),
            ModelBackendBenchmark(
                backendId = BackendCandidateId.ORT_XNNPACK_BIG,
                kind = AcceleratorKind.CPU,
                displayLabel = "CPU XNNPACK (Big)",
                threads = 4,
                medianLatencyMs = 58L,
                verifiedFullOffload = true,
                numericalParityOk = true
            )
        )

        val plan = AdaptiveHardwareSelector.selectPlanForModel(ModelKind.REAL_ESRGAN, evaluated)
        assertEquals(BackendCandidateId.QNN_HTP, plan.selectedBackend)
        assertEquals(AcceleratorKind.NPU, plan.selectedKind)
        assertEquals(24L, plan.measuredLatencyMs)
        assertEquals(BackendCandidateId.ORT_XNNPACK_ALL, plan.fallbackChain.last())
        assertTrue(plan.fallbackChain.contains(BackendCandidateId.NNAPI_GPU))
    }

    @Test
    fun `selector applies GPU over NPU over CPU tie break only within 10 percent margin`() {
        val evaluated = listOf(
            ModelBackendBenchmark(
                backendId = BackendCandidateId.NNAPI_GPU,
                kind = AcceleratorKind.GPU,
                displayLabel = "NNAPI GPU",
                medianLatencyMs = 32L,
                verifiedFullOffload = true,
                numericalParityOk = true
            ),
            ModelBackendBenchmark(
                backendId = BackendCandidateId.NNAPI_NPU,
                kind = AcceleratorKind.NPU,
                displayLabel = "NNAPI NPU",
                medianLatencyMs = 30L,
                verifiedFullOffload = true,
                numericalParityOk = true
            ),
            ModelBackendBenchmark(
                backendId = BackendCandidateId.ORT_XNNPACK_ALL,
                kind = AcceleratorKind.CPU,
                displayLabel = "CPU XNNPACK",
                threads = 4,
                medianLatencyMs = 44L,
                verifiedFullOffload = true,
                numericalParityOk = true
            )
        )

        // 32ms is within 10% of fastest 30ms (threshold 33ms) -> GPU wins tie-break!
        val plan = AdaptiveHardwareSelector.selectPlanForModel(ModelKind.DNCNN, evaluated)
        assertEquals(BackendCandidateId.NNAPI_GPU, plan.selectedBackend)
        assertEquals(AcceleratorKind.GPU, plan.selectedKind)
    }

    @Test
    fun `selector picks CPU XNNPACK when GPU or NPU drivers are slower or fail parity`() {
        val evaluated = listOf(
            ModelBackendBenchmark(
                backendId = BackendCandidateId.NNAPI_GPU,
                kind = AcceleratorKind.GPU,
                displayLabel = "NNAPI GPU",
                medianLatencyMs = 18L,
                verifiedFullOffload = true,
                numericalParityOk = false,
                rejectedReason = "MAE too high"
            ),
            ModelBackendBenchmark(
                backendId = BackendCandidateId.NNAPI_NPU,
                kind = AcceleratorKind.NPU,
                displayLabel = "NNAPI NPU",
                medianLatencyMs = 75L,
                verifiedFullOffload = true,
                numericalParityOk = true
            ),
            ModelBackendBenchmark(
                backendId = BackendCandidateId.ORT_XNNPACK_BIG,
                kind = AcceleratorKind.CPU,
                displayLabel = "CPU XNNPACK (Big)",
                threads = 4,
                medianLatencyMs = 31L,
                verifiedFullOffload = true,
                numericalParityOk = true
            )
        )

        val plan = AdaptiveHardwareSelector.selectPlanForModel(ModelKind.FACE_YUNET, evaluated)
        assertEquals(BackendCandidateId.ORT_XNNPACK_BIG, plan.selectedBackend)
        assertEquals(AcceleratorKind.CPU, plan.selectedKind)
        assertFalse(plan.fallbackChain.contains(BackendCandidateId.NNAPI_GPU))
        assertEquals(BackendCandidateId.OPENCV_CPU, plan.fallbackChain.last())
    }

    @Test
    fun `numerical verifier rejects NaN, constant degenerate outputs, and high MAE`() {
        val ref = floatArrayOf(0.1f, 0.4f, 0.7f, 0.9f)
        assertTrue(AdaptiveHardwareSelector.verifyNumericalOutput(floatArrayOf(0.11f, 0.39f, 0.71f, 0.89f), ref).valid)
        assertFalse(AdaptiveHardwareSelector.verifyNumericalOutput(floatArrayOf(0.1f, Float.NaN, 0.7f, 0.9f), ref).valid)
        assertFalse(AdaptiveHardwareSelector.verifyNumericalOutput(floatArrayOf(0.5f, 0.5f, 0.5f, 0.5f), ref).valid)
        assertFalse(AdaptiveHardwareSelector.verifyNumericalOutput(floatArrayOf(0.9f, 0.1f, 0.1f, 0.1f), ref).valid)
    }

    @Test
    fun `adaptive hardware profile cache serializes and invalidates on fingerprint or app version mismatch`() {
        val topology = CpuTopologyInfo(8, 4, 4, 2850, "4 Big + 4 Little (2850MHz)")
        val snapshot = HardwareProbeSnapshot(
            vulkanSupported = true,
            vulkanVersionLabel = "Vulkan 1.3",
            vulkanComputeSupported = true,
            openClSupported = true,
            openClSourceLabel = "Vendor OpenCL 64",
            openGlEsMajor = 3,
            nnapiSupported = true,
            nnapiGpuDetected = true,
            nnapiNpuDetected = true,
            nnapiDriversSummary = "GPU Delegate + NPU/DSP Accelerator",
            isQualcommSoc = true,
            qnnHtpAvailable = false,
            qnnGpuAvailable = false,
            qnnStatusLabel = "Snapdragon Vendor QNN Libs",
            cpuTopology = topology
        )

        val srPlan = AdaptiveHardwareSelector.selectPlanForModel(
            ModelKind.REAL_ESRGAN,
            listOf(ModelBackendBenchmark(BackendCandidateId.NNAPI_GPU, AcceleratorKind.GPU, "NNAPI GPU", 0, 29L, true, true))
        )
        val dnPlan = AdaptiveHardwareSelector.selectPlanForModel(
            ModelKind.DNCNN,
            listOf(ModelBackendBenchmark(BackendCandidateId.NNAPI_NPU, AcceleratorKind.NPU, "NNAPI NPU", 0, 14L, true, true))
        )
        val fcPlan = AdaptiveHardwareSelector.selectPlanForModel(
            ModelKind.FACE_YUNET,
            listOf(ModelBackendBenchmark(BackendCandidateId.OPENCV_CPU, AcceleratorKind.CPU, "OpenCV SIMD CPU", 0, 8L, true, true))
        )

        val profile = AdaptiveHardwareProfile(
            deviceFingerprint = "fp-1",
            appVersionCode = 10L,
            modelSignature = "sig-v1",
            probeSnapshot = snapshot,
            plans = mapOf(
                ModelKind.REAL_ESRGAN to srPlan,
                ModelKind.DNCNN to dnPlan,
                ModelKind.FACE_YUNET to fcPlan
            ),
            benchmarkedAtEpochMs = 123456L
        )
        assertEquals("SR:GPU • DN:NPU • Face:CPU", profile.compactSummaryPill())

        val json = AdaptiveHardwareSelector.serializeProfile(profile)
        val restored = AdaptiveHardwareSelector.deserializeProfileIfValid(json, "fp-1", 10L, "sig-v1")
        assertNotNull(restored)
        assertTrue(restored!!.fromCache)
        assertEquals(AcceleratorKind.GPU, restored.plans[ModelKind.REAL_ESRGAN]?.selectedKind)
        assertEquals(AcceleratorKind.NPU, restored.plans[ModelKind.DNCNN]?.selectedKind)
        assertEquals(AcceleratorKind.CPU, restored.plans[ModelKind.FACE_YUNET]?.selectedKind)

        // Invalidation checks
        assertNull(AdaptiveHardwareSelector.deserializeProfileIfValid(json, "fp-other", 10L, "sig-v1"))
        assertNull(AdaptiveHardwareSelector.deserializeProfileIfValid(json, "fp-1", 11L, "sig-v1"))
        assertNull(AdaptiveHardwareSelector.deserializeProfileIfValid(json, "fp-1", 10L, "sig-v2"))
    }

    @Test
    fun `smart XNNPACK candidates prioritize big cores and scale threads under thermal pressure`() {
        val topology = CpuTopologyInfo(8, 3, 5, 2900, "3 Big + 5 Little (2900MHz)")
        val cands = AdaptiveHardwareSelector.buildSmartXnnpackThreadCandidates(topology, maxTierThreads = 8)
        assertEquals(BackendCandidateId.ORT_XNNPACK_BIG to 3, cands.first())
        assertTrue(cands.any { it.second == 4 })

        assertEquals(6, AdaptiveHardwareSelector.thermalAdjustedThreads(6, ThermalState.NORMAL))
        assertEquals(5, AdaptiveHardwareSelector.thermalAdjustedThreads(6, ThermalState.WARM))
        assertEquals(3, AdaptiveHardwareSelector.thermalAdjustedThreads(6, ThermalState.THROTTLING))
        assertEquals(2, AdaptiveHardwareSelector.thermalAdjustedThreads(6, ThermalState.CRITICAL))
    }

    @Test
    fun `temporal reuse policy enforces max consecutive reuses, scene cut resets, and keyframe face schedule`() {
        val base = byteArrayOf(10, 20, 30, 40)
        val same = byteArrayOf(10, 21, 30, 40)
        val moved = byteArrayOf(80, 90, 110, 120)

        assertTrue(TemporalReusePolicy.computeMeanAbsDiff(base, same) < TemporalReusePolicy.TILE_STATIC_MAD_THRESHOLD)
        assertTrue(TemporalReusePolicy.computeMeanAbsDiff(base, moved) > TemporalReusePolicy.SCENE_CUT_MAD_THRESHOLD)

        assertTrue(TemporalReusePolicy.shouldReuseStaticTile(0.4f, 0, isSceneCut = false, hasCachedOutput = true))
        assertTrue(TemporalReusePolicy.shouldReuseStaticTile(0.4f, 2, isSceneCut = false, hasCachedOutput = true))
        assertFalse("Must force full AI refresh after max consecutive reuses", TemporalReusePolicy.shouldReuseStaticTile(0.4f, 3, isSceneCut = false, hasCachedOutput = true))
        assertFalse("Must not reuse on scene cut", TemporalReusePolicy.shouldReuseStaticTile(0.4f, 0, isSceneCut = true, hasCachedOutput = true))

        assertTrue(TemporalReusePolicy.shouldReuseDuplicateFrame(0.3f, 0, hasCachedFrame = true))
        assertFalse(TemporalReusePolicy.shouldReuseDuplicateFrame(0.3f, 3, hasCachedFrame = true))

        assertTrue(TemporalReusePolicy.shouldRunFullFaceDetection(0, 1.0f, true))
        assertFalse(TemporalReusePolicy.shouldRunFullFaceDetection(1, 1.0f, true))
        assertFalse(TemporalReusePolicy.shouldRunFullFaceDetection(6, 1.0f, true))
        assertTrue(TemporalReusePolicy.shouldRunFullFaceDetection(12, 1.0f, true))
        assertTrue("Scene cut must immediately refresh face boxes", TemporalReusePolicy.shouldRunFullFaceDetection(1, 12.0f, true))
    }

    @Test
    fun `YuNet strided decoder extracts face bounding box and performs NMS`() {
        val w = 320
        val h = 320
        val stride = 16
        val cols = ceil(w.toDouble() / stride).toInt() // 20
        val rows = ceil(h.toDouble() / stride).toInt() // 20
        val count = cols * rows

        val cls = FloatArray(count) { 0f }
        val obj = FloatArray(count) { 0f }
        val bbox = FloatArray(count * 4) { 0f }

        // Place strong detection at cell (r=5, c=6) and weaker overlapping cell (r=5, c=6 -> next neighbor c=7)
        val idx1 = 5 * cols + 6
        cls[idx1] = 0.92f
        obj[idx1] = 0.90f
        bbox[idx1 * 4 + 0] = 0f
        bbox[idx1 * 4 + 1] = 0f
        bbox[idx1 * 4 + 2] = 1.6f // exp(1.6)*16 ~= 79px
        bbox[idx1 * 4 + 3] = 1.8f

        val idx2 = 5 * cols + 7
        cls[idx2] = 0.70f
        obj[idx2] = 0.72f
        bbox[idx2 * 4 + 0] = -0.9f
        bbox[idx2 * 4 + 1] = 0f
        bbox[idx2 * 4 + 2] = 1.6f
        bbox[idx2 * 4 + 3] = 1.8f

        val decoded = YuNetDecoder.decodeStridedOutputs(
            inputW = w,
            inputH = h,
            outputsByName = mapOf("cls_16" to cls, "obj_16" to obj, "bbox_16" to bbox),
            scoreThreshold = 0.56f,
            nmsThreshold = 0.30f
        )
        assertEquals("Overlapping weaker box should be suppressed by NMS", 1, decoded.size)
        assertTrue(decoded[0].score > 0.85f)
        assertTrue(decoded[0].width in 60f..100f)
    }

    @Test
    fun `FastYuvCopyMath accurately identifies NV12 vs NV21 vs I420 and copies semi-planar UV rows`() {
        // Interleaved [U0=10, V0=20, U1=30, V1=40] -> NV12: uRemaining > vRemaining and uProbe1 == vProbe0
        assertEquals(
            YuvFastLayout.NV12,
            FastYuvCopyMath.detectLayout(
                uPixelStride = 2, vPixelStride = 2,
                uRemaining = 100, vRemaining = 99,
                uProbe0 = 10, uProbe1 = 20, uProbe2 = 30, uProbe3 = 40,
                vProbe0 = 20, vProbe1 = 30, vProbe2 = 40, vProbe3 = 50
            )
        )
        assertEquals(
            YuvFastLayout.NV21,
            FastYuvCopyMath.detectLayout(
                uPixelStride = 2, vPixelStride = 2,
                uRemaining = 99, vRemaining = 100,
                uProbe0 = 10, uProbe1 = 20, uProbe2 = 30, uProbe3 = 40,
                vProbe0 = 20, vProbe1 = 10, vProbe2 = 40, vProbe3 = 30
            )
        )
        assertEquals(
            YuvFastLayout.PLANAR_I420,
            FastYuvCopyMath.detectLayout(
                uPixelStride = 1, vPixelStride = 1,
                uRemaining = 50, vRemaining = 50,
                uProbe0 = 10, uProbe1 = 11, uProbe2 = 12, uProbe3 = 13,
                vProbe0 = 20, vProbe1 = 21, vProbe2 = 22, vProbe3 = 23
            )
        )

        // Strided semi-planar copy (rowStride = 6, uvWidth = 2 -> 4 active bytes per row)
        val srcInterleaved = ByteBuffer.wrap(
            byteArrayOf(
                10, 20, 30, 40, 99, 99,
                50, 60, 70, 80, 99, 99
            )
        )
        val dst = ByteArray(8)
        FastYuvCopyMath.copySemiPlanarUvPlane(
            interleavedBuf = srcInterleaved,
            cropLeftHalf = 0,
            cropTopHalf = 0,
            uvWidth = 2,
            uvHeight = 2,
            rowStride = 6,
            dst = dst,
            dstOffset = 0
        )
        assertArrayEquals(byteArrayOf(10, 20, 30, 40, 50, 60, 70, 80), dst)
    }

    @Test
    fun `diagnostics logger captures per-model acceleration, static tile reuse, duplicate reuse, and fallbacks`() {
        val logger = EngineDiagnosticsLogger()
        val report = logger.buildReport(
            totalMs = 4200L,
            executionProvider = "SR:GPU • DN:NPU • Face:CPU",
            benchmarkSummary = "SR=NNAPI GPU(28ms) | DN=QNN HTP NPU(14ms) | Face=OpenCV SIMD(9ms)",
            codecUsed = "H.265 (Hardware)",
            initialTileSize = 256,
            finalTileSize = 256,
            framesProcessed = 60,
            recoveredFrames = 0,
            analysisReusedFromCache = true,
            upscaleScaleLabel = "2.0x",
            perModelAccelerationSummary = "SR:GPU • DN:NPU • Face:CPU",
            staticTilesReused = 42,
            framesReusedDuplicate = 5,
            runtimeFallbacksCount = 1
        )
        assertEquals("SR:GPU • DN:NPU • Face:CPU", report.perModelAccelerationSummary)
        assertEquals(42, report.staticTilesReused)
        assertEquals(5, report.framesReusedDuplicate)
        assertEquals(1, report.runtimeFallbacksCount)
    }
}
