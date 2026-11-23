package com.example.engine

import android.util.Log
import com.example.model.EngineDiagnosticsReport
import java.util.Collections
import java.util.concurrent.atomic.AtomicLong

/**
 * Structured engine telemetry & error logger for ClearKit.
 * Records exact timings, execution provider benchmark outcomes, per-model hardware allocation,
 * temporal reuse gains, peak memory usage, and runtime fallback events.
 */
class EngineDiagnosticsLogger(
    private val tag: String = "ClearKit_Engine"
) {
    private val warnings = Collections.synchronizedList(mutableListOf<String>())
    private val timeline = Collections.synchronizedList(mutableListOf<String>())

    var modelInitMs: Long = 0L
    var analysisMs: Long = 0L
    var extractionMs: Long = 0L
    val aiInferenceMs = AtomicLong(0L)
    val dncnnInferenceMs = AtomicLong(0L)
    val faceAiMs = AtomicLong(0L)
    val enhancementMs = AtomicLong(0L)
    val encodingMs = AtomicLong(0L)
    var muxingMs: Long = 0L
    var speedFallbackEngaged: Boolean = false
    var speedFallbackReason: String? = null
    var decoderBackend: String = "MediaCodec NV12/I420 Pooled"
    private var peakMemMb: Long = 0L

    fun info(message: String) {
        timeline.add("[INFO] $message")
        try {
            Log.i(tag, message)
        } catch (_: Throwable) {
        }
    }

    fun warn(message: String) {
        warnings.add(message)
        timeline.add("[WARN] $message")
        try {
            Log.w(tag, message)
        } catch (_: Throwable) {
        }
    }

    fun error(message: String, throwable: Throwable? = null) {
        val full = if (throwable != null) "$message: ${throwable.javaClass.simpleName}: ${throwable.message}" else message
        warnings.add(full)
        timeline.add("[ERROR] $full")
        try {
            if (throwable != null) {
                Log.e(tag, message, throwable)
            } else {
                Log.e(tag, message)
            }
        } catch (_: Throwable) {
        }
    }

    fun sampleHeapUsedMb(): Long {
        val rt = Runtime.getRuntime()
        val usedMb = ((rt.totalMemory() - rt.freeMemory()) / (1024L * 1024L)).coerceAtLeast(1L)
        if (usedMb > peakMemMb) {
            peakMemMb = usedMb
        }
        return usedMb
    }

    fun peakMemoryMb(): Long {
        sampleHeapUsedMb()
        return peakMemMb
    }

    fun buildReport(
        totalMs: Long,
        executionProvider: String,
        benchmarkSummary: String,
        codecUsed: String,
        initialTileSize: Int,
        finalTileSize: Int,
        framesProcessed: Int,
        recoveredFrames: Int,
        analysisReusedFromCache: Boolean,
        upscaleScaleLabel: String = "1.0x",
        perModelAccelerationSummary: String = executionProvider,
        staticTilesReused: Int = 0,
        framesReusedDuplicate: Int = 0,
        runtimeFallbacksCount: Int = 0,
        interpolatedFramesCount: Int = 0,
        stutterRepairsCount: Int = 0,
        colorEnhancedFrames: Int = framesProcessed,
        smoothnessModeLabel: String = "Flow + VFR Stabilized",
        dncnnFramesProcessed: Int = 0,
        dncnnAsyncPrefetchHits: Int = 0
    ): EngineDiagnosticsReport {
        return EngineDiagnosticsReport(
            modelInitMs = modelInitMs,
            analysisMs = analysisMs,
            extractionMs = extractionMs,
            aiInferenceMs = aiInferenceMs.get(),
            enhancementMs = enhancementMs.get(),
            encodingMs = encodingMs.get(),
            muxingMs = muxingMs,
            totalMs = totalMs,
            peakMemoryMb = peakMemoryMb(),
            executionProvider = executionProvider,
            benchmarkSummary = benchmarkSummary,
            codecUsed = codecUsed,
            initialTileSize = initialTileSize,
            finalTileSize = finalTileSize,
            framesProcessed = framesProcessed,
            recoveredFrames = recoveredFrames,
            analysisReusedFromCache = analysisReusedFromCache,
            speedFallbackEngaged = speedFallbackEngaged,
            speedFallbackReason = speedFallbackReason,
            decoderBackend = decoderBackend,
            dncnnInferenceMs = dncnnInferenceMs.get(),
            faceAiMs = faceAiMs.get(),
            upscaleScaleLabel = upscaleScaleLabel,
            perModelAccelerationSummary = perModelAccelerationSummary,
            staticTilesReused = staticTilesReused,
            framesReusedDuplicate = framesReusedDuplicate,
            runtimeFallbacksCount = runtimeFallbacksCount,
            interpolatedFramesCount = interpolatedFramesCount,
            stutterRepairsCount = stutterRepairsCount,
            colorEnhancedFrames = colorEnhancedFrames,
            smoothnessModeLabel = smoothnessModeLabel,
            dncnnFramesProcessed = dncnnFramesProcessed,
            dncnnAsyncPrefetchHits = dncnnAsyncPrefetchHits,
            warnings = warnings.toList()
        )
    }
}
