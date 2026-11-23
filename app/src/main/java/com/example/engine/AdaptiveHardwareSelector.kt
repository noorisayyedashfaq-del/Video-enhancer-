package com.example.engine

import com.example.model.AcceleratorKind
import com.example.model.AdaptiveHardwareProfile
import com.example.model.BackendCandidateId
import com.example.model.CpuTopologyInfo
import com.example.model.ModelBackendBenchmark
import com.example.model.ModelExecutionPlan
import com.example.model.ModelKind
import com.example.model.ThermalState
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.roundToLong

/**
 * Pure benchmark-driven Adaptive Hardware Selector & Output Parity Verifier.
 *
 * Selection Rules:
 * 1. Never assume GPU > NPU > CPU: every candidate is verified for operator offload
 *    and numerical accuracy, then timed at representative production input shapes.
 * 2. Pick the fastest verified backend. When two passing candidates are within 10%
 *    latency parity, break ties using priority: 1. GPU -> 2. NPU -> 3. CPU.
 * 3. Always retain an ordered fallback chain ending in a verified CPU backend.
 */
object AdaptiveHardwareSelector {

    const val TIE_BREAK_MARGIN_RATIO = 1.10
    const val MAX_ACCEPTABLE_PARITY_MAE = 0.085f
    const val MIN_NON_CONSTANT_SPREAD = 1e-4f

    private val json = Json {
        ignoreUnknownKeys = true
        encodeDefaults = true
    }

    /**
     * Evaluates all benchmarked candidates for [modelKind] and returns an optimal [ModelExecutionPlan].
     */
    fun selectPlanForModel(
        modelKind: ModelKind,
        evaluated: List<ModelBackendBenchmark>,
        tieBreakMarginRatio: Double = TIE_BREAK_MARGIN_RATIO
    ): ModelExecutionPlan {
        val passing = evaluated.filter {
            it.verifiedFullOffload &&
                it.numericalParityOk &&
                it.medianLatencyMs != null &&
                it.medianLatencyMs > 0L
        }

        if (passing.isEmpty()) {
            val safeFallbackBackend = if (modelKind == ModelKind.FACE_YUNET) {
                BackendCandidateId.OPENCV_CPU
            } else {
                BackendCandidateId.ORT_XNNPACK_ALL
            }
            return ModelExecutionPlan(
                modelKind = modelKind,
                selectedBackend = safeFallbackBackend,
                selectedKind = AcceleratorKind.CPU,
                selectedLabel = safeFallbackBackend.displayLabel,
                selectedThreads = 4,
                measuredLatencyMs = evaluated.firstOrNull { it.medianLatencyMs != null }?.medianLatencyMs ?: 500L,
                fallbackChain = listOf(safeFallbackBackend),
                evaluatedBackends = evaluated
            )
        }

        val fastestMs = passing.minOf { it.medianLatencyMs!! }
        val tieThresholdMs = max(fastestMs + 1L, (fastestMs * tieBreakMarginRatio).roundToLong())

        // Among candidates within 10% of the fastest real latency, prefer priority 1. GPU -> 2. NPU -> 3. CPU,
        // then shortest latency. If an NPU or CPU backend beats GPU by >10%, the faster backend wins!
        val nearTiePool = passing.filter { it.medianLatencyMs!! <= tieThresholdMs }
        val winner = nearTiePool.sortedWith(
            compareBy<ModelBackendBenchmark> { it.kind.priorityRank }
                .thenBy { it.medianLatencyMs!! }
        ).first()

        // Ordered fallback chain: Winner -> other faster/passing hardware backends -> CPU safety tail
        val remaining = passing
            .filter { it.backendId != winner.backendId }
            .sortedWith(
                compareBy<ModelBackendBenchmark> { it.medianLatencyMs!! }
                    .thenBy { it.kind.priorityRank }
            )

        val chainIds = ArrayList<BackendCandidateId>(remaining.size + 2)
        chainIds.add(winner.backendId)
        for (cand in remaining) {
            if (cand.backendId !in chainIds) {
                chainIds.add(cand.backendId)
            }
        }
        val cpuTail = if (modelKind == ModelKind.FACE_YUNET) {
            BackendCandidateId.OPENCV_CPU
        } else {
            BackendCandidateId.ORT_XNNPACK_ALL
        }
        if (cpuTail !in chainIds) {
            chainIds.add(cpuTail)
        }

        return ModelExecutionPlan(
            modelKind = modelKind,
            selectedBackend = winner.backendId,
            selectedKind = winner.kind,
            selectedLabel = winner.displayLabel,
            selectedThreads = winner.threads,
            measuredLatencyMs = winner.medianLatencyMs!!,
            fallbackChain = chainIds,
            evaluatedBackends = evaluated
        )
    }

    /**
     * Verifies tensor numerical health (finite values, non-degenerate spread) and optionally
     * compares sample parity against an FP32 CPU reference array.
     */
    data class OutputVerificationResult(
        val valid: Boolean,
        val meanAbsoluteError: Float = 0f,
        val reason: String? = null
    )

    fun verifyNumericalOutput(
        candidateSamples: FloatArray,
        referenceSamples: FloatArray? = null,
        maxMae: Float = MAX_ACCEPTABLE_PARITY_MAE
    ): OutputVerificationResult {
        if (candidateSamples.isEmpty()) {
            return OutputVerificationResult(valid = false, reason = "Empty output tensor")
        }

        var minVal = Float.POSITIVE_INFINITY
        var maxVal = Float.NEGATIVE_INFINITY
        var absDiffSum = 0.0
        val compareLen = if (referenceSamples != null) {
            minOf(candidateSamples.size, referenceSamples.size)
        } else {
            0
        }

        for (i in candidateSamples.indices) {
            val v = candidateSamples[i]
            if (v.isNaN() || v.isInfinite()) {
                return OutputVerificationResult(valid = false, reason = "NaN/Inf output at idx=$i")
            }
            if (v < minVal) minVal = v
            if (v > maxVal) maxVal = v
            if (i < compareLen && referenceSamples != null) {
                absDiffSum += abs(v - referenceSamples[i]).toDouble()
            }
        }

        if ((maxVal - minVal) < MIN_NON_CONSTANT_SPREAD) {
            return OutputVerificationResult(
                valid = false,
                reason = "Degenerate constant output (spread=${maxVal - minVal})"
            )
        }

        if (compareLen > 0) {
            val mae = (absDiffSum / compareLen.toDouble()).toFloat()
            if (mae > maxMae) {
                return OutputVerificationResult(
                    valid = false,
                    meanAbsoluteError = mae,
                    reason = "Parity MAE=${"%.4f".format(mae)} exceeds $maxMae"
                )
            }
            return OutputVerificationResult(valid = true, meanAbsoluteError = mae)
        }

        return OutputVerificationResult(valid = true)
    }

    /**
     * Computes candidate XNNPACK thread counts prioritizing big/prime core counts first
     * so little-core barrier sync stalls don't slow inference.
     */
    fun buildSmartXnnpackThreadCandidates(
        topology: CpuTopologyInfo,
        maxTierThreads: Int
    ): List<Pair<BackendCandidateId, Int>> {
        val ceiling = minOf(topology.totalCores.coerceAtLeast(2), maxTierThreads.coerceAtLeast(4))
        val result = ArrayList<Pair<BackendCandidateId, Int>>()
        val bigSafe = topology.bigCoreCount.coerceIn(2, ceiling)
        if (bigSafe >= 2) {
            result.add(BackendCandidateId.ORT_XNNPACK_BIG to bigSafe)
        }
        for (t in listOf(4, 6, 8)) {
            if (t <= ceiling && result.none { it.second == t }) {
                result.add(BackendCandidateId.ORT_XNNPACK_ALL to t)
            }
        }
        if (result.isEmpty()) {
            result.add(BackendCandidateId.ORT_XNNPACK_ALL to ceiling.coerceIn(2, 4))
        }
        return result
    }

    /**
     * Adjusts runtime worker concurrency smoothly under thermal pressure without ever reducing AI strength.
     */
    fun thermalAdjustedThreads(baseThreads: Int, thermalState: ThermalState): Int {
        val safeBase = baseThreads.coerceIn(2, 8)
        return when (thermalState) {
            ThermalState.NORMAL -> safeBase
            ThermalState.WARM -> (safeBase - 1).coerceAtLeast(2)
            ThermalState.THROTTLING -> (safeBase / 2).coerceIn(2, 4)
            ThermalState.CRITICAL -> 2
        }
    }

    fun medianMs(samples: List<Long>): Long {
        if (samples.isEmpty()) return 0L
        val sorted = samples.sorted()
        return sorted[sorted.size / 2].coerceAtLeast(1L)
    }

    fun serializeProfile(profile: AdaptiveHardwareProfile): String {
        return json.encodeToString(profile)
    }

    fun deserializeProfileIfValid(
        rawJson: String?,
        expectedFingerprint: String,
        expectedVersionCode: Long,
        expectedModelSignature: String
    ): AdaptiveHardwareProfile? {
        if (rawJson.isNullOrBlank()) return null
        return try {
            val parsed = json.decodeFromString<AdaptiveHardwareProfile>(rawJson)
            if (parsed.deviceFingerprint != expectedFingerprint ||
                parsed.appVersionCode != expectedVersionCode ||
                parsed.modelSignature != expectedModelSignature ||
                parsed.plans.size < ModelKind.entries.size
            ) {
                null
            } else {
                parsed.copy(fromCache = true)
            }
        } catch (_: Throwable) {
            null
        }
    }
}
