package com.example.engine

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtProvider
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import android.content.Context
import android.os.Build
import com.example.model.AcceleratorKind
import com.example.model.AdaptiveHardwareProfile
import com.example.model.BackendCandidateId
import com.example.model.DeviceCapabilities
import com.example.model.HardwareProbeSnapshot
import com.example.model.ModelBackendBenchmark
import com.example.model.ModelExecutionPlan
import com.example.model.ModelKind
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.EnumSet
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.min

/**
 * Hardware Capability Prober, Operator Verification Gate, Independent Per-Model Benchmarker,
 * and Persistent Hardware Profile Manager for ClearKit.
 *
 * Evaluates Real-ESRGAN, DnCNN, and Face AI (YuNet) independently across GPU, NPU, and CPU
 * candidates at realistic production workloads, never assuming GPU > NPU > CPU without
 * verified benchmark latency.
 */
class AdaptiveHardwareAccelerator(
    private val context: Context,
    private val env: OrtEnvironment
) {
    private val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    private val ortCacheDir: File by lazy {
        File(context.cacheDir, "clearkit_ort_optimized").apply { mkdirs() }
    }

    data class ActiveSessionsBundle(
        val profile: AdaptiveHardwareProfile,
        val srRankedSessions: List<Pair<BackendCandidateId, OrtSession>>,
        val dncnnRankedSessions: List<Pair<BackendCandidateId, OrtSession>>
    )

    fun buildOrRestoreExecutionBundle(
        srFile: File,
        dncnnFile: File,
        yunetFile: File,
        yunetRunner: YuNetOnnxRunner,
        capabilities: DeviceCapabilities,
        forceRebenchmark: Boolean = false,
        logger: EngineDiagnosticsLogger? = null
    ): ActiveSessionsBundle {
        val probe = capabilities.hardwareProbeSnapshot
            ?: DeviceCapabilityDetector(context).probeHardwareSnapshot()
        val appVer = resolveAppVersionCode()
        val modelSig = buildModelSignature(srFile, dncnnFile, yunetFile)
        val fingerprint = "${Build.FINGERPRINT}|${probe.cpuTopology.bigCoreCount}/${probe.cpuTopology.totalCores}"

        if (!forceRebenchmark) {
            val cachedRaw = prefs.getString(KEY_HW_PROFILE_JSON, null)
            val cached = AdaptiveHardwareSelector.deserializeProfileIfValid(
                rawJson = cachedRaw,
                expectedFingerprint = fingerprint,
                expectedVersionCode = appVer,
                expectedModelSignature = modelSig
            )
            if (cached != null) {
                val restored = restoreBundleFromProfile(
                    profile = cached,
                    srFile = srFile,
                    dncnnFile = dncnnFile,
                    yunetFile = yunetFile,
                    yunetRunner = yunetRunner,
                    probe = probe,
                    capabilities = capabilities,
                    logger = logger
                )
                if (restored != null) {
                    logger?.info("Restored cached adaptive hardware profile: ${cached.compactSummaryPill()}")
                    return restored
                }
            }
        }

        // Fresh calibration: probe & benchmark all 3 AI models independently
        val threadCandidates = AdaptiveHardwareSelector.buildSmartXnnpackThreadCandidates(
            topology = probe.cpuTopology,
            maxTierThreads = capabilities.inferenceThreads.coerceAtLeast(4)
        )
        val availableProviders = runCatching { OrtEnvironment.getAvailableProviders() }
            .getOrDefault(EnumSet.of(OrtProvider.CPU))

        // 1. Real-ESRGAN [1, 3, 256, 256]
        val srResult = if (srFile.exists() && srFile.length() > 0L) {
            benchmarkOnnxModelCandidates(
                modelKind = ModelKind.REAL_ESRGAN,
                modelFile = srFile,
                tileInputShape = longArrayOf(1L, 3L, 256L, 256L),
                minExpectedOutputElems = 3 * 256 * 256 * 16,
                probe = probe,
                availableProviders = availableProviders,
                threadCandidates = threadCandidates,
                logger = logger
            )
        } else {
            AdaptiveHardwareSelector.selectPlanForModel(ModelKind.REAL_ESRGAN, emptyList()) to emptyList()
        }

        // 2. DnCNN [1, 1, 256, 256]
        val dncnnResult = if (dncnnFile.exists() && dncnnFile.length() > 0L) {
            benchmarkOnnxModelCandidates(
                modelKind = ModelKind.DNCNN,
                modelFile = dncnnFile,
                tileInputShape = longArrayOf(1L, 1L, 256L, 256L),
                minExpectedOutputElems = 256 * 256,
                probe = probe,
                availableProviders = availableProviders,
                threadCandidates = threadCandidates,
                logger = logger
            )
        } else {
            AdaptiveHardwareSelector.selectPlanForModel(ModelKind.DNCNN, emptyList()) to emptyList()
        }

        // 3. Face AI (YuNet) [1, 3, 320, 320] across ORT (GPU/NPU/XNNPACK) & OpenCV (OpenCL/Vulkan/CPU)
        val facePlan = if (yunetFile.exists() && yunetFile.length() > 0L) {
            val faceOrtCandidates = buildCandidateSessions(
                modelKind = ModelKind.FACE_YUNET,
                modelFile = yunetFile,
                probe = probe,
                availableProviders = availableProviders,
                threadCandidates = threadCandidates,
                logger = logger
            )
            yunetRunner.benchmarkAndSelectBest(
                yunetPath = yunetFile.absolutePath,
                candidateOrtSessions = faceOrtCandidates,
                openClAvailable = probe.openClSupported,
                vulkanAvailable = probe.vulkanComputeSupported,
                logger = logger
            )
        } else {
            AdaptiveHardwareSelector.selectPlanForModel(ModelKind.FACE_YUNET, emptyList())
        }

        val freshProfile = AdaptiveHardwareProfile(
            deviceFingerprint = fingerprint,
            appVersionCode = appVer,
            modelSignature = modelSig,
            probeSnapshot = probe,
            plans = mapOf(
                ModelKind.REAL_ESRGAN to srResult.first,
                ModelKind.DNCNN to dncnnResult.first,
                ModelKind.FACE_YUNET to facePlan
            ),
            benchmarkedAtEpochMs = System.currentTimeMillis(),
            fromCache = false
        )

        prefs.edit().putString(KEY_HW_PROFILE_JSON, AdaptiveHardwareSelector.serializeProfile(freshProfile)).apply()
        logger?.info(
            "Adaptive Hardware Profile Benchmarked: ${freshProfile.compactSummaryPill()} • " +
                "SR=${srResult.first.selectedLabel}(${srResult.first.measuredLatencyMs}ms) • " +
                "DnCNN=${dncnnResult.first.selectedLabel}(${dncnnResult.first.measuredLatencyMs}ms) • " +
                "Face=${facePlan.selectedLabel}(${facePlan.measuredLatencyMs}ms)"
        )

        return ActiveSessionsBundle(
            profile = freshProfile,
            srRankedSessions = srResult.second,
            dncnnRankedSessions = dncnnResult.second
        )
    }

    private fun restoreBundleFromProfile(
        profile: AdaptiveHardwareProfile,
        srFile: File,
        dncnnFile: File,
        yunetFile: File,
        yunetRunner: YuNetOnnxRunner,
        probe: HardwareProbeSnapshot,
        capabilities: DeviceCapabilities,
        logger: EngineDiagnosticsLogger?
    ): ActiveSessionsBundle? {
        return try {
            val srPlan = profile.plans[ModelKind.REAL_ESRGAN] ?: return null
            val dnPlan = profile.plans[ModelKind.DNCNN] ?: return null
            val facePlan = profile.plans[ModelKind.FACE_YUNET] ?: return null

            val availableProviders = runCatching { OrtEnvironment.getAvailableProviders() }
                .getOrDefault(EnumSet.of(OrtProvider.CPU))

            val srSessions = recreateRankedSessions(srFile, ModelKind.REAL_ESRGAN, srPlan, probe, availableProviders)
            if (srFile.exists() && srSessions.isEmpty()) return null

            val dnSessions = recreateRankedSessions(dncnnFile, ModelKind.DNCNN, dnPlan, probe, availableProviders)
            if (dncnnFile.exists() && dnSessions.isEmpty()) {
                srSessions.forEach { runCatching { it.second.close() } }
                return null
            }

            if (yunetFile.exists()) {
                val restoredFaceCandidates = ArrayList<Triple<BackendCandidateId, Int, OrtSession>>()
                for (backendId in facePlan.fallbackChain) {
                    if (backendId == BackendCandidateId.OPENCV_CPU ||
                        backendId == BackendCandidateId.OPENCV_OPENCL ||
                        backendId == BackendCandidateId.OPENCV_VULKAN
                    ) {
                        continue
                    }
                    val threads = facePlan.evaluatedBackends.firstOrNull { it.backendId == backendId }?.threads
                        ?: capabilities.inferenceThreads
                    createSessionForBackend(yunetFile, ModelKind.FACE_YUNET, backendId, threads, availableProviders)?.let {
                        restoredFaceCandidates.add(Triple(backendId, threads, it))
                    }
                }
                // Re-verify tiny Face AI chain cleanly so active runner is armed
                yunetRunner.benchmarkAndSelectBest(
                    yunetPath = yunetFile.absolutePath,
                    candidateOrtSessions = restoredFaceCandidates,
                    openClAvailable = probe.openClSupported && facePlan.fallbackChain.contains(BackendCandidateId.OPENCV_OPENCL),
                    vulkanAvailable = probe.vulkanComputeSupported && facePlan.fallbackChain.contains(BackendCandidateId.OPENCV_VULKAN),
                    logger = logger
                )
            }

            ActiveSessionsBundle(
                profile = profile,
                srRankedSessions = srSessions,
                dncnnRankedSessions = dnSessions
            )
        } catch (_: Throwable) {
            null
        }
    }

    private fun recreateRankedSessions(
        modelFile: File,
        modelKind: ModelKind,
        plan: ModelExecutionPlan,
        probe: HardwareProbeSnapshot,
        availableProviders: EnumSet<OrtProvider>
    ): List<Pair<BackendCandidateId, OrtSession>> {
        if (!modelFile.exists() || modelFile.length() == 0L) return emptyList()
        val list = ArrayList<Pair<BackendCandidateId, OrtSession>>()
        for (backendId in plan.fallbackChain) {
            val threads = plan.evaluatedBackends.firstOrNull { it.backendId == backendId }?.threads
                ?: probe.cpuTopology.bigCoreCount.coerceIn(2, 6)
            val sess = createSessionForBackend(modelFile, modelKind, backendId, threads, availableProviders)
            if (sess != null) {
                list.add(backendId to sess)
            }
        }
        return list
    }

    private fun benchmarkOnnxModelCandidates(
        modelKind: ModelKind,
        modelFile: File,
        tileInputShape: LongArray,
        minExpectedOutputElems: Int,
        probe: HardwareProbeSnapshot,
        availableProviders: EnumSet<OrtProvider>,
        threadCandidates: List<Pair<BackendCandidateId, Int>>,
        logger: EngineDiagnosticsLogger?
    ): Pair<ModelExecutionPlan, List<Pair<BackendCandidateId, OrtSession>>> {
        val built = buildCandidateSessions(
            modelKind = modelKind,
            modelFile = modelFile,
            probe = probe,
            availableProviders = availableProviders,
            threadCandidates = threadCandidates,
            logger = logger
        )

        val benchmarks = ArrayList<ModelBackendBenchmark>(built.size)
        val verifiedSessions = LinkedHashMap<BackendCandidateId, Pair<ModelBackendBenchmark, OrtSession>>()
        var cpuRefSamples: FloatArray? = null

        // Run CPU candidates first to lock FP32 reference baseline for hardware parity validation
        val ordered = built.sortedBy { it.first.kind != AcceleratorKind.CPU }
        for ((backendId, threads, sess) in ordered) {
            val eval = evaluateOrtModelSession(
                session = sess,
                backendId = backendId,
                threads = threads,
                shape = tileInputShape,
                minOutputElems = minExpectedOutputElems,
                referenceSamples = cpuRefSamples,
                timeoutSec = 8L,
                logger = logger
            )
            if (eval.first.verifiedFullOffload && eval.first.numericalParityOk && backendId.kind == AcceleratorKind.CPU && cpuRefSamples == null) {
                cpuRefSamples = eval.second
            }
            benchmarks.add(eval.first)
            if (eval.first.verifiedFullOffload && eval.first.numericalParityOk && eval.first.medianLatencyMs != null) {
                val existing = verifiedSessions[backendId]
                if (existing == null || eval.first.medianLatencyMs!! < existing.first.medianLatencyMs!!) {
                    existing?.second?.let { runCatching { it.close() } }
                    verifiedSessions[backendId] = eval.first to sess
                } else {
                    runCatching { sess.close() }
                }
            } else {
                runCatching { sess.close() }
            }
        }

        val plan = AdaptiveHardwareSelector.selectPlanForModel(modelKind, benchmarks)
        val rankedSessions = ArrayList<Pair<BackendCandidateId, OrtSession>>()
        for (backendId in plan.fallbackChain) {
            val pair = verifiedSessions.remove(backendId)
            if (pair != null) {
                rankedSessions.add(backendId to pair.second)
            }
        }
        // Close any leftover non-selected sessions cleanly
        verifiedSessions.values.forEach { runCatching { it.second.close() } }
        return plan to rankedSessions
    }

    private fun buildCandidateSessions(
        modelKind: ModelKind,
        modelFile: File,
        probe: HardwareProbeSnapshot,
        availableProviders: EnumSet<OrtProvider>,
        threadCandidates: List<Pair<BackendCandidateId, Int>>,
        logger: EngineDiagnosticsLogger?
    ): List<Triple<BackendCandidateId, Int, OrtSession>> {
        val list = ArrayList<Triple<BackendCandidateId, Int, OrtSession>>()

        // CPU XNNPACK candidates (Big-core pinned + multi-thread)
        for ((cpuId, threads) in threadCandidates) {
            createSessionForBackend(modelFile, modelKind, cpuId, threads, availableProviders)?.let {
                list.add(Triple(cpuId, threads, it))
            }
        }
        if (list.isEmpty()) {
            createSessionForBackend(modelFile, modelKind, BackendCandidateId.ORT_CPU, 4, availableProviders)?.let {
                list.add(Triple(BackendCandidateId.ORT_CPU, 4, it))
            }
        }

        // Hardware GPU / NPU candidates
        val hwTargets = buildList {
            if (probe.qnnGpuAvailable) add(BackendCandidateId.QNN_GPU)
            if (probe.qnnHtpAvailable) add(BackendCandidateId.QNN_HTP)
            if (probe.nnapiSupported && availableProviders.contains(OrtProvider.NNAPI)) {
                if (probe.nnapiGpuDetected) add(BackendCandidateId.NNAPI_GPU)
                if (probe.nnapiNpuDetected) add(BackendCandidateId.NNAPI_NPU)
                add(BackendCandidateId.NNAPI_AUTO)
            }
        }
        val hwHelperThreads = probe.cpuTopology.bigCoreCount.coerceIn(2, 4)
        for (hwId in hwTargets) {
            val sess = createSessionForBackend(modelFile, modelKind, hwId, hwHelperThreads, availableProviders)
            if (sess != null) {
                list.add(Triple(hwId, hwHelperThreads, sess))
            } else {
                logger?.info("${modelKind.shortTag}: ${hwId.displayLabel} skipped during init")
            }
        }
        return list
    }

    private fun createSessionForBackend(
        modelFile: File,
        modelKind: ModelKind,
        backendId: BackendCandidateId,
        threads: Int,
        availableProviders: EnumSet<OrtProvider>
    ): OrtSession? {
        return try {
            val opts = OrtSession.SessionOptions().apply {
                setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                setMemoryPatternOptimization(true)
                setCPUArenaAllocator(true)
                setInterOpNumThreads(1)
                addConfigEntry("session.intra_op.allow_spinning", "0")
                addConfigEntry("session.inter_op.allow_spinning", "0")

                when (backendId) {
                    BackendCandidateId.ORT_XNNPACK_BIG,
                    BackendCandidateId.ORT_XNNPACK_ALL -> {
                        setIntraOpNumThreads(1)
                        addXnnpack(mapOf("intra_op_num_threads" to threads.coerceIn(2, 8).toString()))
                    }
                    BackendCandidateId.ORT_CPU -> {
                        setIntraOpNumThreads(threads.coerceIn(2, 8))
                    }
                    BackendCandidateId.NNAPI_GPU -> {
                        if (!availableProviders.contains(OrtProvider.NNAPI)) throw IllegalStateException("NNAPI missing")
                        setIntraOpNumThreads(threads.coerceIn(2, 4))
                        // CPU_DISABLED enforces strict hardware operator verification (prevents silent CPU fallback)
                        addNnapi(EnumSet.of(NNAPIFlags.USE_FP16, NNAPIFlags.USE_NCHW, NNAPIFlags.CPU_DISABLED))
                    }
                    BackendCandidateId.NNAPI_NPU -> {
                        if (!availableProviders.contains(OrtProvider.NNAPI)) throw IllegalStateException("NNAPI missing")
                        setIntraOpNumThreads(threads.coerceIn(2, 4))
                        addNnapi(EnumSet.of(NNAPIFlags.USE_FP16, NNAPIFlags.CPU_DISABLED))
                    }
                    BackendCandidateId.NNAPI_AUTO -> {
                        if (!availableProviders.contains(OrtProvider.NNAPI)) throw IllegalStateException("NNAPI missing")
                        setIntraOpNumThreads(threads.coerceIn(2, 4))
                        addNnapi(EnumSet.of(NNAPIFlags.CPU_DISABLED))
                    }
                    BackendCandidateId.QNN_HTP -> {
                        if (!invokeAddQnnIfPresent(
                                opts = this,
                                options = mapOf(
                                    "backend_path" to "libQnnHtp.so",
                                    "htp_performance_mode" to "burst"
                                )
                            )
                        ) {
                            throw IllegalStateException("QNN HTP unavailable")
                        }
                    }
                    BackendCandidateId.QNN_GPU -> {
                        if (!invokeAddQnnIfPresent(
                                opts = this,
                                options = mapOf("backend_path" to "libQnnGpu.so")
                            )
                        ) {
                            throw IllegalStateException("QNN GPU unavailable")
                        }
                    }
                    else -> throw IllegalStateException("Unsupported ORT candidate $backendId")
                }

                // Persist optimized compiled graph once for CPU/XNNPACK so subsequent cold-starts skip graph passes
                if (backendId == BackendCandidateId.ORT_XNNPACK_BIG || backendId == BackendCandidateId.ORT_XNNPACK_ALL) {
                    val cacheFile = File(ortCacheDir, "${modelKind.id.lowercase()}_${backendId.name.lowercase()}.ort")
                    // DnCNN uses external weights (.data); skip .ort rewrite for external-data models safely
                    if (modelKind != ModelKind.DNCNN) {
                        setOptimizedModelFilePath(cacheFile.absolutePath)
                    }
                }
            }
            env.createSession(modelFile.absolutePath, opts)
        } catch (_: Throwable) {
            null
        }
    }

    /**
     * Safely attaches Qualcomm QNN EP via reflection so standard multi-ABI `onnxruntime-android`
     * compiles and runs on x86_64 emulators & non-Snapdragon SoCs while activating QNN whenever present.
     */
    private fun invokeAddQnnIfPresent(
        opts: OrtSession.SessionOptions,
        options: Map<String, String>
    ): Boolean {
        return try {
            val method = opts.javaClass.methods.firstOrNull { m ->
                m.name == "addQnn" && m.parameterCount == 1
            } ?: return false
            method.invoke(opts, options)
            true
        } catch (_: Throwable) {
            false
        }
    }

    private fun evaluateOrtModelSession(
        session: OrtSession,
        backendId: BackendCandidateId,
        threads: Int,
        shape: LongArray,
        minOutputElems: Int,
        referenceSamples: FloatArray?,
        timeoutSec: Long,
        logger: EngineDiagnosticsLogger?
    ): Pair<ModelBackendBenchmark, FloatArray?> {
        val executor = Executors.newSingleThreadExecutor()
        return try {
            val future = executor.submit<Pair<ModelBackendBenchmark, FloatArray?>> {
                val inputName = session.inputNames.first()
                val totalFloats = shape.fold(1L) { acc, v -> acc * v }.toInt()
                val direct = ByteBuffer.allocateDirect(totalFloats * 4).order(ByteOrder.nativeOrder())
                val fBuf = direct.asFloatBuffer()
                for (i in 0 until totalFloats) {
                    fBuf.put(i, ((i * 29) % 255) / 255f)
                }
                fBuf.rewind()

                var extractedSample: FloatArray? = null

                // Warmup + Shape + Numerical Parity Verification
                OnnxTensor.createTensor(env, fBuf, shape).use { tWarm ->
                    session.run(mapOf(inputName to tWarm)).use { res ->
                        val outTensor = res[0] as? OnnxTensor
                            ?: return@submit ModelBackendBenchmark(
                                backendId, backendId.kind, backendId.displayLabel, threads,
                                rejectedReason = "Non-tensor output"
                            ) to null
                        val outBuf = outTensor.floatBuffer
                        if (outBuf.remaining() < minOutputElems) {
                            return@submit ModelBackendBenchmark(
                                backendId, backendId.kind, backendId.displayLabel, threads,
                                rejectedReason = "Output size ${outBuf.remaining()} < $minOutputElems"
                            ) to null
                        }
                        val sampleStep = (outBuf.remaining() / 512).coerceAtLeast(1)
                        val arr = FloatArray(min(512, outBuf.remaining()))
                        for (idx in arr.indices) {
                            arr[idx] = outBuf.get((idx * sampleStep).coerceAtMost(outBuf.remaining() - 1))
                        }
                        extractedSample = arr
                    }
                }

                val parity = AdaptiveHardwareSelector.verifyNumericalOutput(
                    candidateSamples = extractedSample ?: FloatArray(0),
                    referenceSamples = referenceSamples
                )
                if (!parity.valid) {
                    return@submit ModelBackendBenchmark(
                        backendId = backendId,
                        kind = backendId.kind,
                        displayLabel = backendId.displayLabel,
                        threads = threads,
                        verifiedFullOffload = true,
                        numericalParityOk = false,
                        rejectedReason = parity.reason
                    ) to null
                }

                // Multi-run benchmark for reliable sustained median latency
                val runs = ArrayList<Long>(3)
                for (r in 0 until 3) {
                    fBuf.rewind()
                    val t0 = System.nanoTime()
                    OnnxTensor.createTensor(env, fBuf, shape).use { tRun ->
                        session.run(mapOf(inputName to tRun)).use { }
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
                ) to extractedSample
            }
            future.get(timeoutSec, TimeUnit.SECONDS)
        } catch (t: Throwable) {
            logger?.info("${backendId.displayLabel} rejected during verification (${t.message})")
            ModelBackendBenchmark(
                backendId = backendId,
                kind = backendId.kind,
                displayLabel = backendId.displayLabel,
                threads = threads,
                rejectedReason = t.message ?: "Execution failed or timed out"
            ) to null
        } finally {
            executor.shutdownNow()
        }
    }

    private fun buildModelSignature(srFile: File, dncnnFile: File, yunetFile: File): String {
        return "${srFile.length()}_${dncnnFile.length()}_${yunetFile.length()}"
    }

    private fun resolveAppVersionCode(): Long {
        return try {
            val pInfo = context.packageManager.getPackageInfo(context.packageName, 0)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) pInfo.longVersionCode else pInfo.versionCode.toLong()
        } catch (_: Throwable) {
            1L
        }
    }

    companion object {
        private const val PREFS_NAME = "clearkit_adaptive_hw_cache"
        private const val KEY_HW_PROFILE_JSON = "hw_profile_v1"
    }
}
