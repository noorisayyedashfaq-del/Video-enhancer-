package com.example.model

import android.net.Uri
import kotlinx.serialization.Serializable
import java.io.File

enum class EnhancementMode(
    val id: String,
    val label: String,
    val shortBadge: String,
    val subtitle: String
) {
    HD(
        id = "HD",
        label = "HD",
        shortBadge = "No AI Tiles",
        subtitle = "Fast OpenCV • Original Res"
    ),
    HD_PLUS(
        id = "HD_PLUS",
        label = "HD+",
        shortBadge = "512px • 6 tiles",
        subtitle = "AI 6 Tiles • Original Res"
    ),
    QHD_2K(
        id = "QHD_2K",
        label = "2K",
        shortBadge = "512px • 12 tiles",
        subtitle = "AI 12 Tiles • Original Res"
    ),
    UHD_4K(
        id = "UHD_4K",
        label = "4K",
        shortBadge = "512px • 25 tiles",
        subtitle = "AI 25 Tiles • Original Res"
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
    val useAiSuperResolution: Boolean,
    val useMultiStageUpscaling: Boolean,
    val useTiledInference: Boolean,
    val aiTileSize: Int = 0,
    val targetTilesPerFrame: Int = 0,
    val temporalConsistency: Boolean,
    val brightnessPercent: Float,
    val contrastPercent: Float,
    val sharpnessPercent: Float,
    val denoisePercent: Float,
    val saturationPercent: Float,
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
    val processingCapacityScore: Int
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
    val tilesPerFrame: Int = 0
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
    val claheClipLimit: Double,       // 0.0..2.2
    val gamma: Float,                 // 0.80..1.25
    val saturationGain: Float,        // 0.90..1.45
    val wbGainR: Float,               // 0.85..1.18
    val wbGainG: Float,               // ~1.0
    val wbGainB: Float,               // 0.85..1.18
    val denoiseStrength: Float,       // 0..1.0
    val useAiDenoise: Boolean,
    val deblockingStrength: Float,    // 0..1.0
    val dehazingStrength: Float,      // 0..0.35
    val sharpnessAmount: Float,       // 0..0.9
    val edgeEnhancementAmount: Float, // 0..0.6
    val flickerReductionAlpha: Float, // 0..0.35
    val faceEnhanceStrength: Float,   // 0..0.55
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
    val diagnosticMessage: String? = null
)

data class MainUiState(
    val engineHealth: EngineHealthStatus? = null,
    val deviceCapabilities: DeviceCapabilities? = null,
    val profiles: Map<EnhancementMode, EnhancementProfile> = emptyMap(),
    val selectedVideo: VideoMetadata? = null,
    val selectedMode: EnhancementMode = EnhancementMode.HD_PLUS,
    val computedOutputDimensions: Map<EnhancementMode, OutputDimensions> = emptyMap(),
    val isProcessing: Boolean = false,
    val progress: ProcessingProgress = ProcessingProgress(),
    val enhancementResult: EnhancementResult? = null,
    val isSavingToGallery: Boolean = false,
    val saveSuccessMessage: String? = null,
    val errorMessage: String? = null
)
