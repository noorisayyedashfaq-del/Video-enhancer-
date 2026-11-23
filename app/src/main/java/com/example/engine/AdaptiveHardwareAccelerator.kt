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
import java.util.concurrent.atomic.AtomicBoolean
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

        // 1. Real-ESRGAN [1, 3, 192, 192] fast production probe shape
        val srResult = if (srFile.exists() && srFile.length() > 0L) {
            benchmarkOnnxModelCandidates(
                modelKind = ModelKind.REAL_ESRGAN,
                modelFile = srFile,
                tileInputShape = longArrayOf(1L, 3L, 192L, 192L),
                minExpectedOutputElems = 3 * 192 * 192 * 16,
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
            val faceSpecs = buildCandidateSpecs(
                probe = probe,
                availableProviders = availableProviders,
                threadCandidates = threadCandidates
            )
            val faceFactories = faceSpecs.map { (backendId, threads) ->
                Triple(backendId, threads) {
                    createSessionForBackend(yunetFile, ModelKind.FACE_YUNET, backendId, threads, availableProviders)
                }
            }
            yuNetRunnerSafeBenchmark(
                yunetRunner = yunetRunner,
                yunetPath = yunetFile.absolutePath,
                candidateFactories = faceFactories,
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
            if (srFile.exists() && srPlan.fallbackChain.isNotEmpty() && srSessions.isEmpty()) return null

            val dnSessions = recreateRankedSessions(dncnnFile, ModelKind.DNCNN, dnPlan, probe, availableProviders)
            if (dncnnFile.exists() && dnPlan.fallbackChain.isNotEmpty() && dnSessions.isEmpty()) {
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

    private fun yuNetRunnerSafeBenchmark(
        yunetRunner: YuNetOnnxRunner,
        yunetPath: String,
        candidateFactories: List<Triple<BackendCandidateId, Int, () -> OrtSession?>>,
        openClAvailable: Boolean,
        vulkanAvailable: Boolean,
        logger: EngineDiagnosticsLogger?
    ): ModelExecutionPlan {
        return try {
            yunetRunner.benchmarkAndSelectBestFactories(
                yunetPath = yunetPath,
                candidateFactories = candidateFactories,
                openClAvailable = openClAvailable,
                vulkanAvailable = vulkanAvailable,
                logger = logger
            )
        } catch (t: Throwable) {
            logger?.warn("YuNet benchmark fallback (${t.message})")
            AdaptiveHardwareSelector.selectPlanForModel(ModelKind.FACE_YUNET, emptyList())
        }
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
        val specs = buildCandidateSpecs(
            probe = probe,
            availableProviders = availableProviders,
            threadCandidates = threadCandidates
        )

        val benchmarks = ArrayList<ModelBackendBenchmark>(specs.size)
        val verifiedSessions = LinkedHashMap<BackendCandidateId, Pair<ModelBackendBenchmark, OrtSession>>()
        var cpuRefSamples: FloatArray? = null

        // Evaluate strictly one session at a time (CPU first to lock FP32 parity baseline)
        val ordered = specs.sortedBy { it.first.kind != AcceleratorKind.CPU }
        for ((backendId, threads) in ordered) {
            val sess = createSessionForBackend(modelFile, modelKind, backendId, threads, availableProviders)
            if (sess == null) {
                logger?.info("${modelKind.shortTag}: ${backendId.displayLabel} skipped during init")
                continue
            }
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
            if (eval.bench.verifiedFullOffload && eval.bench.numericalParityOk && backendId.kind == AcceleratorKind.CPU && cpuRefSamples == null) {
                cpuRefSamples = eval.samples
            }
            benchmarks.add(eval.bench)
            val medMs = eval.bench.medianLatencyMs
            if (eval.bench.verifiedFullOffload && eval.bench.numericalParityOk && medMs != null) {
                val existing = verifiedSessions[backendId]
                val existingMed = existing?.first?.medianLatencyMs
                if (existing == null || existingMed == null || medMs < existingMed) {
                    existing?.second?.let { runCatching { it.close() } }
                    verifiedSessions[backendId] = eval.bench to sess
                } else {
                    runCatching { sess.close() }
                }
            } else if (!eval.deferredCloseArmed) {
                // Only close immediately when native session.run() is no longer active
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

    private fun buildCandidateSpecs(
        probe: HardwareProbeSnapshot,
        availableProviders: EnumSet<OrtProvider>,
        threadCandidates: List<Pair<BackendCandidateId, Int>>
    ): List<Pair<BackendCandidateId, Int>> {
        val list = ArrayList<Pair<BackendCandidateId, Int>>()
        for ((cpuId, threads) in threadCandidates) {
            list.add(cpuId to threads)
        }
        if (list.isEmpty()) {
            list.add(BackendCandidateId.ORT_CPU to 4)
        }
        if (probe.qnnGpuAvailable) list.add(BackendCandidateId.QNN_GPU to probe.cpuTopology.bigCoreCount.coerceIn(2, 4))
        if (probe.qnnHtpAvailable) list.add(BackendCandidateId.QNN_HTP to probe.cpuTopology.bigCoreCount.coerceIn(2, 4))
        if (probe.nnapiSupported && availableProviders.contains(OrtProvider.NNAPI)) {
            val hwHelperThreads = probe.cpuTopology.bigCoreCount.coerceIn(2, 4)
            if (probe.nnapiGpuDetected) list.add(BackendCandidateId.NNAPI_GPU to hwHelperThreads)
            if (probe.nnapiNpuDetected) list.add(BackendCandidateId.NNAPI_NPU to hwHelperThreads)
            list.add(BackendCandidateId.NNAPI_AUTO to hwHelperThreads)
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
        val isXnnpackCpu = backendId == BackendCandidateId.ORT_XNNPACK_BIG ||
            backendId == BackendCandidateId.ORT_XNNPACK_ALL
        val cacheFile = if (isXnnpackCpu && modelKind != ModelKind.DNCNN) {
            File(ortCacheDir, "${modelKind.id.lowercase()}_${backendId.name.lowercase()}_t${threads}_v${resolveAppVersionCode()}.ort").also {
                if (it.exists() && it.length() == 0L) it.delete()
            }
        } else {
            null
        }

        fun buildAndCreate(ortCache: File?, nnapiUseNchw: Boolean = true): OrtSession {
            val opts = OrtSession.SessionOptions().apply {
                setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                setMemoryPatternOptimization(true)
                setCPUArenaAllocator(true)
                setInterOpNumThreads(1)
                addConfigEntry("session.intra_op.allow_spinning", "0")
                addConfigEntry("session.inter_op.allow_spinning", "0")
                addConfigEntry("session.set_denormal_as_zero", "1")
                addConfigEntry("session.disable_prepacking", "0")

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
                        // CPU_DISABLED enforces strict hardware operator verification (prevents silent CPU fallback).
                        // Retry without USE_NCHW if Adreno/Mali NNAPI driver prefers NHWC layout for Conv2D graphs.
                        val flags = if (nnapiUseNchw) {
                            EnumSet.of(NNAPIFlags.USE_FP16, NNAPIFlags.USE_NCHW, NNAPIFlags.CPU_DISABLED)
                        } else {
                            EnumSet.of(NNAPIFlags.USE_FP16, NNAPIFlags.CPU_DISABLED)
                        }
                        addNnapi(flags)
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

                if (ortCache != null) {
                    setOptimizedModelFilePath(ortCache.absolutePath)
                }
            }
            return try {
                env.createSession(modelFile.absolutePath, opts)
            } finally {
                runCatching { opts.close() }
            }
        }

        return try {
            buildAndCreate(cacheFile, nnapiUseNchw = true)
        } catch (_: Throwable) {
            if (backendId == BackendCandidateId.NNAPI_GPU) {
                runCatching { buildAndCreate(null, nnapiUseNchw = false) }.getOrNull()
            } else if (cacheFile != null) {
                runCatching { cacheFile.delete() }
                runCatching { buildAndCreate(null, nnapiUseNchw = true) }.getOrNull()
            } else {
                null
            }
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

    private data class SessionEvalOutcome(
        val bench: ModelBackendBenchmark,
        val samples: FloatArray?,
        val deferredCloseArmed: Boolean = false
    )

    private fun evaluateOrtModelSession(
        session: OrtSession,
        backendId: BackendCandidateId,
        threads: Int,
        shape: LongArray,
        minOutputElems: Int,
        referenceSamples: FloatArray?,
        timeoutSec: Long,
        logger: EngineDiagnosticsLogger?
    ): SessionEvalOutcome {
        val executor = Executors.newSingleThreadExecutor()
        val nativeActive = AtomicBoolean(true)
        val callerTimedOut = AtomicBoolean(false)
        return try {
            val future = executor.submit<Pair<ModelBackendBenchmark, FloatArray?>> {
                try {
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

                    // 2 timed runs after warmup for fast, deterministic latency selection
                    val runs = ArrayList<Long>(2)
                    for (r in 0 until 2) {
                        if (callerTimedOut.get()) break
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
                } finally {
                    nativeActive.set(false)
                    if (callerTimedOut.get()) {
                        // Safe deferred close after native session.run() unwinds, preventing SIGSEGV use-after-free
                        runCatching { session.close() }
                    }
                }
            }
            val result = future.get(timeoutSec, TimeUnit.SECONDS)
            SessionEvalOutcome(result.first, result.second, deferredCloseArmed = false)
        } catch (t: Throwable) {
            callerTimedOut.set(nativeActive.get())
            logger?.info("${backendId.displayLabel} rejected during verification (${t.message})")
            SessionEvalOutcome(
                bench = ModelBackendBenchmark(
                    backendId = backendId,
                    kind = backendId.kind,
                    displayLabel = backendId.displayLabel,
                    threads = threads,
                    rejectedReason = t.message ?: "Execution failed or timed out"
                ),
                samples = null,
                deferredCloseArmed = nativeActive.get()
            )
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
