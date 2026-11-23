package com.example.model

import android.net.Uri
import kotlinx.serialization.Serializable
import java.io.File

enum class EnhancementMode(
    val id: String,
    val label: String,
    val title: String,
    val shortBadge: String,
    val subtitle: String
) {
    CLARITY_1X(
        id = "CLARITY_1X",
        label = "1x",
        title = "Clarity 1x",
        shortBadge = "Natural AI • 1x",
        subtitle = "Natural AI • Original Res"
    ),
    CLARITY_2X(
        id = "CLARITY_2X",
        label = "2x",
        title = "Clarity 2x",
        shortBadge = "HD AI • 2x",
        subtitle = "HD AI • 2x Super-Res"
    ),
    CLARITY_3X(
        id = "CLARITY_3X",
        label = "3x",
        title = "Clarity 3x",
        shortBadge = "Ultra AI • 3x",
        subtitle = "Ultra AI • 3x Super-Res"
    ),
    CLARITY_4X(
        id = "CLARITY_4X",
        label = "4x",
        title = "Clarity 4x",
        shortBadge = "Max AI • 4x",
        subtitle = "Maximum AI • 4x Super-Res"
    )
}

@Serializable
data class EnhancementProfile(
    val id: String,
    val displayName: String,
    val description: String,
    val targetLongEdge: Int,
    val targetShortEdge: Int,
    val preserveOriginalResolution: Boolean,
    val targetScaleFactor: Float = 1.0f,
    val maxOutputLongEdge: Int = 3840,
    val useAiSuperResolution: Boolean,
    val useMultiStageUpscaling: Boolean,
    val useTiledInference: Boolean,
    val aiTileSize: Int = 256,
    val targetTilesPerFrame: Int = 4,
    val temporalConsistency: Boolean,
    val brightnessPercent: Float,
    val contrastPercent: Float,
    val sharpnessPercent: Float,
    val denoisePercent: Float,
    val saturationPercent: Float,
    val brilliancePercent: Float = 0f,
    val clarityPercent: Float = 0f,
    val highlightsPercent: Float = 0f,
    val shadowsPercent: Float = 0f,
    val whitesPercent: Float = 0f,
    val blacksPercent: Float = 0f,
    val temperatureShift: Float = 0f,
    val hueShift: Float = 0f,
    val aiStrengthPercent: Float = 65f,
    val detailRecoveryPercent: Float = 65f,
    val noiseReductionPercent: Float = 45f,
    val deblurPercent: Float = 55f,
    val aiSharpeningPercent: Float = 40f,
    val artifactReductionPercent: Float = 45f,
    val dncnnStrengthPercent: Float = 48f,
    val faceEnhancePercent: Float = 58f,
    val gamma: Float,
    val whiteBalance: String,
    val colorCorrection: String,
    val deblocking: String,
    val dehazingPercent: Float,
    val edgeEnhancementPercent: Float,
    val flickerReduction: String,
    val preferredCodec: String,
    val targetBpp: Float
)

@Serializable
data class EnhancementProfilesRoot(
    val version: String,
    val modes: Map<String, EnhancementProfile>
)

@Serializable
data class DeviceProfileTier(
    val name: String,
    val minRamMb: Long,
    val maxRamMb: Long,
    val tileSize: Int,
    val tileOverlap: Int,
    val maxBatchFrames: Int,
    val maxInferenceThreads: Int,
    val preferHardwareAcceleration: Boolean,
    val minRequiredFreeStorageMb: Long,
    val throttleDelayMsOnWarm: Long,
    val throttleDelayMsOnHot: Long
)

@Serializable
data class DeviceProfilesRoot(
    val version: String,
    val tiers: List<DeviceProfileTier>
)

enum class ThermalState(val label: String) {
    NORMAL("Nominal"),
    WARM("Warm"),
    THROTTLING("Throttled"),
    CRITICAL("Overheating")
}

@Serializable
enum class AcceleratorKind(val label: String, val priorityRank: Int) {
    GPU("GPU", 0),
    NPU("NPU", 1),
    CPU("CPU", 2)
}

@Serializable
enum class ModelKind(val id: String, val displayName: String, val shortTag: String) {
    REAL_ESRGAN("REAL_ESRGAN", "Real-ESRGAN SR", "SR"),
    DNCNN("DNCNN", "DnCNN Denoise", "DN"),
    FACE_YUNET("FACE_YUNET", "Face AI (YuNet)", "Face")
}

@Serializable
enum class BackendCandidateId(
    val kind: AcceleratorKind,
    val displayLabel: String
) {
    QNN_GPU(AcceleratorKind.GPU, "Qualcomm QNN GPU"),
    NNAPI_GPU(AcceleratorKind.GPU, "NNAPI GPU"),
    OPENCV_OPENCL(AcceleratorKind.GPU, "OpenCL GPU"),
    OPENCV_VULKAN(AcceleratorKind.GPU, "Vulkan GPU"),
    GLES_COMPUTE(AcceleratorKind.GPU, "GLES 3.1 GPU"),
    QNN_HTP(AcceleratorKind.NPU, "Qualcomm QNN HTP NPU"),
    NNAPI_NPU(AcceleratorKind.NPU, "NNAPI NPU/Accelerator"),
    NNAPI_AUTO(AcceleratorKind.NPU, "NNAPI Hardware"),
    ORT_XNNPACK_BIG(AcceleratorKind.CPU, "CPU XNNPACK (Big Cores)"),
    ORT_XNNPACK_ALL(AcceleratorKind.CPU, "CPU XNNPACK"),
    ORT_CPU(AcceleratorKind.CPU, "CPU ORT"),
    OPENCV_CPU(AcceleratorKind.CPU, "OpenCV SIMD CPU")
}

@Serializable
data class CpuTopologyInfo(
    val totalCores: Int,
    val bigCoreCount: Int,
    val littleCoreCount: Int,
    val maxClockMhz: Int,
    val clusterSummary: String
)

@Serializable
data class HardwareProbeSnapshot(
    val vulkanSupported: Boolean,
    val vulkanVersionLabel: String,
    val vulkanComputeSupported: Boolean,
    val openClSupported: Boolean,
    val openClSourceLabel: String,
    val openGlEsMajor: Int,
    val nnapiSupported: Boolean,
    val nnapiGpuDetected: Boolean,
    val nnapiNpuDetected: Boolean,
    val nnapiDriversSummary: String,
    val isQualcommSoc: Boolean,
    val qnnHtpAvailable: Boolean,
    val qnnGpuAvailable: Boolean,
    val qnnStatusLabel: String,
    val cpuTopology: CpuTopologyInfo
)

@Serializable
data class ModelBackendBenchmark(
    val backendId: BackendCandidateId,
    val kind: AcceleratorKind,
    val displayLabel: String,
    val threads: Int = 0,
    val medianLatencyMs: Long? = null,
    val verifiedFullOffload: Boolean = false,
    val numericalParityOk: Boolean = false,
    val rejectedReason: String? = null
)

@Serializable
data class ModelExecutionPlan(
    val modelKind: ModelKind,
    val selectedBackend: BackendCandidateId,
    val selectedKind: AcceleratorKind,
    val selectedLabel: String,
    val selectedThreads: Int = 0,
    val measuredLatencyMs: Long,
    val fallbackChain: List<BackendCandidateId>,
    val evaluatedBackends: List<ModelBackendBenchmark>
)

@Serializable
data class AdaptiveHardwareProfile(
    val deviceFingerprint: String,
    val appVersionCode: Long,
    val modelSignature: String,
    val probeSnapshot: HardwareProbeSnapshot,
    val plans: Map<ModelKind, ModelExecutionPlan>,
    val benchmarkedAtEpochMs: Long,
    val fromCache: Boolean = false
) {
    fun compactSummaryPill(): String {
        val sr = plans[ModelKind.REAL_ESRGAN]?.selectedKind?.label ?: "CPU"
        val dn = plans[ModelKind.DNCNN]?.selectedKind?.label ?: "CPU"
        val fc = plans[ModelKind.FACE_YUNET]?.selectedKind?.label ?: "CPU"
        return "SR:$sr • DN:$dn • Face:$fc"
    }
}

data class DeviceCapabilities(
    val tierName: String,
    val totalRamMb: Long,
    val availableRamMb: Long,
    val cpuCores: Int,
    val primaryAbi: String,
    val supportedAbis: List<String>,
    val vulkanSupported: Boolean,
    val vulkanVersionLabel: String,
    val nnapiSupported: Boolean,
    val openGlEsMajor: Int,
    val hardwareH264Supported: Boolean,
    val hardwareH265Supported: Boolean,
    val maxEncodeLongEdge: Int = 3840,
    val availableStorageMb: Long,
    val thermalState: ThermalState,
    val temperatureCelsius: Float,
    val adaptiveTileSize: Int,
    val tileOverlap: Int,
    val batchSize: Int,
    val inferenceThreads: Int,
    val minRequiredFreeStorageMb: Long,
    val throttleDelayMsOnWarm: Long,
    val throttleDelayMsOnHot: Long,
    val processingCapacityScore: Int,
    val vulkanComputeSupported: Boolean = false,
    val openClSupported: Boolean = false,
    val qnnSupported: Boolean = false,
    val cpuTopology: CpuTopologyInfo? = null,
    val hardwareProbeSnapshot: HardwareProbeSnapshot? = null
)

data class VideoMetadata(
    val uri: Uri,
    val displayName: String,
    val width: Int,
    val height: Int,
    val rotationDegrees: Int,
    val displayWidth: Int,
    val displayHeight: Int,
    val fps: Double,
    val durationMs: Long,
    val estimatedFrameCount: Int,
    val bitrateBps: Long,
    val videoCodec: String,
    val hasAudio: Boolean,
    val audioCodec: String?,
    val audioSampleRate: Int?,
    val audioChannels: Int?,
    val fileSizeBytes: Long
) {
    val isPortrait: Boolean get() = displayHeight > displayWidth
    val aspectRatio: Double get() = if (displayHeight > 0) displayWidth.toDouble() / displayHeight else 16.0 / 9.0
    val resolutionLabel: String get() = "${displayWidth}x${displayHeight}"
}

data class OutputDimensions(
    val width: Int,
    val height: Int,
    val scaleFactorX: Float,
    val scaleFactorY: Float,
    val needsAiSuperResolution: Boolean,
    val needsMultiStageUpscaling: Boolean,
    val preserveOriginalResolution: Boolean,
    val targetBitrateBps: Int,
    val useH265: Boolean,
    val aiTileSize: Int = 0,
    val tilesPerFrame: Int = 0,
    val targetScaleFactor: Float = 1f
) {
    val label: String get() = "${width}x${height}"
}

data class VideoQualityAnalysis(
    val avgBrightness: Float,          // 0..255
    val darkAreaRatio: Float,          // 0..1
    val overexposedAreaRatio: Float,   // 0..1
    val contrastStdDev: Float,         // 0..128
    val dynamicRange: Float,           // p95 - p5
    val noiseLevel: Float,             // Immerkær noise sigma estimate
    val blurVariance: Float,           // Laplacian variance
    val saturationMean: Float,         // HSV S mean 0..255
    val whiteBalanceShiftR: Float,     // G/R deviation
    val whiteBalanceShiftB: Float,     // G/B deviation
    val compressionArtifactScore: Float, // Blockiness ratio >= 1.0
    val edgeDensity: Float,            // Sobel high-gradient ratio 0..1
    val conditionTags: List<String>
)

data class AdaptiveAdjustmentParams(
    val profileId: String,
    val brightnessShift: Float,       // additive luma shift (-25..+25)
    val exposureHighlightCompress: Float, // 0..0.45 highlight roll-off
    val contrastGain: Float,          // 1.0..1.35
    val claheClipLimit: Double,       // 0.0..1.6
    val gamma: Float,                 // 0.80..1.25
    val saturationGain: Float,        // 0.90..1.35
    val vibranceGain: Float = 0.25f,  // 0.0..0.75 selective boost for faded colors
    val wbGainR: Float,               // 0.85..1.18
    val wbGainG: Float,               // ~1.0
    val wbGainB: Float,               // 0.85..1.18
    val denoiseStrength: Float,       // 0..1.0
    val useAiDenoise: Boolean,
    val deblockingStrength: Float,    // 0..1.0
    val dehazingStrength: Float,      // 0..0.35
    val sharpnessAmount: Float,       // 0..0.9 (used for lightweight fallback & clarity scaling)
    val edgeEnhancementAmount: Float, // 0..0.6
    val flickerReductionAlpha: Float, // 0..0.35
    val faceEnhanceStrength: Float,   // 0..0.55
    val brillianceStrength: Float = 0f,
    val clarityStrength: Float = 0f,
    val shadowsLift: Float = 0f,      // 0..0.25 subtle shadow detail lift
    val whitesGain: Float = 0f,       // 0..0.20 crisp upper-midtone lift without clipping
    val blacksCrush: Float = 0f,      // 0..0.15 clean near-black anchor
    val aiStrength: Float = 0.65f,    // 0.25..0.95 Real-ESRGAN ONNX blend factor
    val detailRecovery: Float = 0.65f,
    val deblurStrength: Float = 0.55f,
    val aiSharpening: Float = 0.40f,
    val dncnnStrength: Float = 0.48f, // 0.0..0.95 DnCNN neural luminance denoise blend
    val skinToneProtection: Boolean = true,
    val enableSmoothness: Boolean = true,
    val motionSmoothStrength: Float = 0.45f,
    val allowFrameInterpolation: Boolean = true,
    val adjustmentSummary: String
)

enum class ProcessingStage(val title: String) {
    IDLE("Ready"),
    IMPORTING("Reading Video Metadata"),
    ANALYZING("Analyzing Video"),
    LOADING_MODEL("Loading AI Model"),
    EXTRACTING("Extracting Frames"),
    ENHANCING("AI Enhancement"),
    UPSCALING("Upscaling"),
    ENCODING("Video Encoding"),
    FINALIZING("Finalizing"),
    COMPLETED("Enhancement Complete"),
    CANCELLED("Cancelled"),
    ERROR("Processing Failed")
}

data class ProcessingProgress(
    val stage: ProcessingStage = ProcessingStage.IDLE,
    val processedFrames: Int = 0,
    val totalFrames: Int = 0,
    val overallProgress: Float = 0f,
    val etaSeconds: Long? = null,
    val statusDetail: String = "",
    val thermalState: ThermalState = ThermalState.NORMAL,
    val activeAcceleration: String = "",
    val processingSpeedLabel: String = "",
    val memoryUsageMb: Long = 0L,
    val tileProgressLabel: String = "",
    val speedFallbackActive: Boolean = false
)

data class EngineDiagnosticsReport(
    val modelInitMs: Long,
    val analysisMs: Long,
    val extractionMs: Long,
    val aiInferenceMs: Long,
    val enhancementMs: Long,
    val encodingMs: Long,
    val muxingMs: Long,
    val totalMs: Long,
    val peakMemoryMb: Long,
    val executionProvider: String,
    val benchmarkSummary: String,
    val codecUsed: String,
    val initialTileSize: Int,
    val finalTileSize: Int,
    val framesProcessed: Int,
    val recoveredFrames: Int,
    val analysisReusedFromCache: Boolean,
    val speedFallbackEngaged: Boolean = false,
    val speedFallbackReason: String? = null,
    val decoderBackend: String = "MediaCodec Zero-Copy",
    val dncnnInferenceMs: Long = 0L,
    val faceAiMs: Long = 0L,
    val upscaleScaleLabel: String = "1.0x",
    val perModelAccelerationSummary: String = "",
    val staticTilesReused: Int = 0,
    val framesReusedDuplicate: Int = 0,
    val runtimeFallbacksCount: Int = 0,
    val interpolatedFramesCount: Int = 0,
    val stutterRepairsCount: Int = 0,
    val colorEnhancedFrames: Int = 0,
    val smoothnessModeLabel: String = "Flow + VFR Stabilized",
    val dncnnFramesProcessed: Int = 0,
    val dncnnAsyncPrefetchHits: Int = 0,
    val warnings: List<String> = emptyList()
)

data class EnhancementResult(
    val outputFile: File,
    val mode: EnhancementMode,
    val originalMetadata: VideoMetadata,
    val outputDimensions: OutputDimensions,
    val codecUsed: String,
    val processingDurationMs: Long,
    val qualityAnalysis: VideoQualityAnalysis,
    val appliedParams: AdaptiveAdjustmentParams,
    val facesEnhancedCount: Int,
    val usedAiSuperResolution: Boolean,
    val usedAiDenoise: Boolean,
    val savedGalleryUri: Uri? = null,
    val diagnostics: EngineDiagnosticsReport? = null
)

data class EngineHealthStatus(
    val openCvReady: Boolean,
    val ffmpegReady: Boolean,
    val onnxRuntimeReady: Boolean,
    val realEsrganModelReady: Boolean,
    val dncnnModelReady: Boolean,
    val yunetModelReady: Boolean,
    val activeExecutionProvider: String,
    val hardwareProfile: AdaptiveHardwareProfile? = null,
    val diagnosticMessage: String? = null
)

data class MainUiState(
    val engineHealth: EngineHealthStatus? = null,
    val deviceCapabilities: DeviceCapabilities? = null,
    val profiles: Map<EnhancementMode, EnhancementProfile> = emptyMap(),
    val selectedVideo: VideoMetadata? = null,
    val selectedMode: EnhancementMode = EnhancementMode.CLARITY_2X,
    val computedOutputDimensions: Map<EnhancementMode, OutputDimensions> = emptyMap(),
    val isProcessing: Boolean = false,
    val isBenchmarkingHardware: Boolean = false,
    val progress: ProcessingProgress = ProcessingProgress(),
    val enhancementResult: EnhancementResult? = null,
    val isSavingToGallery: Boolean = false,
    val saveSuccessMessage: String? = null,
    val errorMessage: String? = null
)
