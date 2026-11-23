package com.example.service

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.util.Log
import androidx.core.content.ContextCompat
import com.example.engine.DeviceCapabilityDetector
import com.example.engine.EnhancementProfileRepository
import com.example.engine.GallerySaver
import com.example.engine.OnnxAiInferenceEngine
import com.example.engine.OpenCvFrameProcessor
import com.example.engine.ResolutionCalculator
import com.example.engine.SmartAutoAdjustmentEngine
import com.example.engine.VideoMediaPipeline
import com.example.model.EngineHealthStatus
import com.example.model.EnhancementMode
import com.example.model.EnhancementResult
import com.example.model.MainUiState
import com.example.model.ProcessingProgress
import com.example.model.ProcessingStage
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineExceptionHandler
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Process-scoped coordinator preserving selected video, enhancement preset,
 * active pipeline execution, and live UI progress across Activity lifecycle events,
 * app minimization, screen lock, and notification resumes.
 */
class EnhancementSessionCoordinator private constructor(
    private val appContext: Context
) {
    private val exceptionHandler = CoroutineExceptionHandler { _, throwable ->
        Log.e(TAG, "Uncaught exception in session coordinator coroutine", throwable)
    }
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO + exceptionHandler)
    private val prefs = appContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    val capabilityDetector = DeviceCapabilityDetector(appContext)
    val profileRepository = EnhancementProfileRepository(appContext)
    val onnxEngine = OnnxAiInferenceEngine(appContext)
    val frameProcessor = OpenCvFrameProcessor(appContext, onnxEngine)
    val smartAdjuster = SmartAutoAdjustmentEngine()
    val pipeline = VideoMediaPipeline(
        context = appContext,
        capabilityDetector = capabilityDetector,
        onnxEngine = onnxEngine,
        frameProcessor = frameProcessor,
        smartAdjuster = smartAdjuster
    )

    private val _uiState = MutableStateFlow(MainUiState())
    val uiState: StateFlow<MainUiState> = _uiState.asStateFlow()

    private val jobRunningLock = AtomicBoolean(false)
    private val enginesInitialized = AtomicBoolean(false)

    val isJobLockedRunning: Boolean get() = jobRunningLock.get()

    init {
        restorePersistedModeAndNotice()
        ensureEnginesInitializedAsync()
    }

    fun ensureEnginesInitializedAsync() {
        if (!enginesInitialized.compareAndSet(false, true)) return
        scope.launch {
            try {
                val profiles = runCatching { profileRepository.loadProfiles() }
                    .getOrDefault(emptyMap())
                val caps = runCatching { capabilityDetector.detect() }.getOrNull()

                _uiState.update {
                    it.copy(
                        deviceCapabilities = caps ?: it.deviceCapabilities,
                        profiles = profiles.ifEmpty { it.profiles }
                    )
                }

                if (caps != null) {
                    val health = try {
                        pipeline.initializeAndCheckHealth(caps)
                    } catch (ce: CancellationException) {
                        throw ce
                    } catch (t: Throwable) {
                        Log.e(TAG, "Pipeline health init failed safely", t)
                        EngineHealthStatus(
                            openCvReady = frameProcessor.isOpenCvReady,
                            ffmpegReady = false,
                            onnxRuntimeReady = false,
                            realEsrganModelReady = onnxEngine.isRealEsrganReady,
                            dncnnModelReady = onnxEngine.isDncnnReady,
                            yunetModelReady = frameProcessor.isFaceDetectorReady,
                            activeExecutionProvider = onnxEngine.providerLabel,
                            hardwareProfile = onnxEngine.adaptiveProfile,
                            diagnosticMessage = t.message ?: "Engine startup warning"
                        )
                    }
                    _uiState.update {
                        it.copy(
                            deviceCapabilities = caps,
                            profiles = profiles.ifEmpty { it.profiles },
                            engineHealth = health
                        )
                    }
                }
                restorePersistedVideoIfUnset()
            } catch (ce: CancellationException) {
                throw ce
            } catch (t: Throwable) {
                Log.e(TAG, "Cold-start async initialization error caught safely", t)
            }
        }
    }

    private fun restorePersistedModeAndNotice() {
        try {
            val savedModeId = prefs.getString(KEY_SELECTED_MODE, null)
            val migratedId = when (savedModeId) {
                "HD" -> EnhancementMode.CLARITY_1X.id
                "HD_PLUS" -> EnhancementMode.CLARITY_2X.id
                "QHD_2K" -> EnhancementMode.CLARITY_3X.id
                "UHD_4K" -> EnhancementMode.CLARITY_4X.id
                else -> savedModeId
            }
            val restoredMode = EnhancementMode.entries.firstOrNull { it.id == migratedId } ?: EnhancementMode.CLARITY_2X
            val interruptedNotice = prefs.getString(KEY_INTERRUPTED_NOTICE, null)
            _uiState.update {
                it.copy(
                    selectedMode = restoredMode,
                    errorMessage = interruptedNotice
                )
            }
        } catch (t: Throwable) {
            Log.w(TAG, "Failed restoring persisted mode cleanly", t)
        }
    }

    private suspend fun restorePersistedVideoIfUnset() {
        try {
            if (_uiState.value.selectedVideo != null || _uiState.value.isProcessing) return
            val savedUriStr = prefs.getString(KEY_SELECTED_VIDEO_URI, null) ?: return
            val uri = runCatching { Uri.parse(savedUriStr) }.getOrNull() ?: return
            val result = pipeline.importAndProbeVideo(uri)
            result.onSuccess { metadata ->
                val caps = _uiState.value.deviceCapabilities ?: capabilityDetector.detect()
                val profiles = _uiState.value.profiles.ifEmpty { profileRepository.loadProfiles() }
                val dimsMap = EnhancementMode.entries.associateWith { mode ->
                    ResolutionCalculator.calculate(metadata, mode, resolveProfile(profiles, mode), caps)
                }
                _uiState.update { state ->
                    if (state.selectedVideo != null) state else state.copy(
                        selectedVideo = metadata,
                        computedOutputDimensions = dimsMap
                    )
                }
            }
        } catch (ce: CancellationException) {
            throw ce
        } catch (t: Throwable) {
            Log.w(TAG, "Could not restore previous video URI cleanly", t)
        }
    }

    fun selectVideo(uri: Uri) {
        if (_uiState.value.isProcessing || jobRunningLock.get()) return
        scope.launch {
            try {
                _uiState.update {
                    it.copy(
                        errorMessage = null,
                        saveSuccessMessage = null,
                        enhancementResult = null,
                        progress = ProcessingProgress(stage = ProcessingStage.IMPORTING, statusDetail = "Reading video metadata...")
                    )
                }
                clearInterruptedNotice()
                val caps = capabilityDetector.detect()
                val result = pipeline.importAndProbeVideo(uri)
                result.fold(
                    onSuccess = { metadata ->
                        prefs.edit().putString(KEY_SELECTED_VIDEO_URI, uri.toString()).apply()
                        val profiles = _uiState.value.profiles.ifEmpty { profileRepository.loadProfiles() }
                        val dimsMap = EnhancementMode.entries.associateWith { mode ->
                            ResolutionCalculator.calculate(metadata, mode, resolveProfile(profiles, mode), caps)
                        }
                        _uiState.update {
                            it.copy(
                                deviceCapabilities = caps,
                                profiles = profiles,
                                selectedVideo = metadata,
                                computedOutputDimensions = dimsMap,
                                progress = ProcessingProgress(stage = ProcessingStage.IDLE),
                                errorMessage = null
                            )
                        }
                    },
                    onFailure = { err ->
                        _uiState.update {
                            it.copy(
                                progress = ProcessingProgress(stage = ProcessingStage.ERROR),
                                errorMessage = err.message ?: "Unable to open selected video."
                            )
                        }
                    }
                )
            } catch (ce: CancellationException) {
                throw ce
            } catch (t: Throwable) {
                Log.e(TAG, "selectVideo failed safely", t)
                _uiState.update {
                    it.copy(
                        progress = ProcessingProgress(stage = ProcessingStage.ERROR),
                        errorMessage = t.message ?: "Unable to open selected video."
                    )
                }
            }
        }
    }

    fun selectMode(mode: EnhancementMode) {
        if (_uiState.value.isProcessing || jobRunningLock.get()) return
        prefs.edit().putString(KEY_SELECTED_MODE, mode.id).apply()
        _uiState.update { it.copy(selectedMode = mode, errorMessage = null) }
    }

    /**
     * Launches the foreground service safely. Prevents duplicate concurrent jobs.
     */
    fun startEnhancementViaForegroundService(context: Context = appContext): Boolean {
        val current = _uiState.value
        if (current.selectedVideo == null) {
            _uiState.update { it.copy(errorMessage = "Please select a video first.") }
            return false
        }
        if (current.isProcessing || jobRunningLock.get()) {
            return false
        }

        clearInterruptedNotice()
        val intent = Intent(context, VideoEnhancementForegroundService::class.java).apply {
            action = VideoEnhancementForegroundService.ACTION_START_ENHANCEMENT
        }
        ContextCompat.startForegroundService(context, intent)
        return true
    }

    fun requestCancelEnhancement(context: Context = appContext) {
        pipeline.cancel()
        val intent = Intent(context, VideoEnhancementForegroundService::class.java).apply {
            action = VideoEnhancementForegroundService.ACTION_CANCEL_ENHANCEMENT
        }
        try {
            context.startService(intent)
        } catch (_: Throwable) {
            // If service isn't currently running, still ensure state reflects cancellation cleanly
            if (!jobRunningLock.get()) {
                _uiState.update {
                    it.copy(
                        isProcessing = false,
                        progress = ProcessingProgress(stage = ProcessingStage.CANCELLED, statusDetail = "Cancelled by user")
                    )
                }
            }
        }
    }

    fun tryAcquireJobLock(): Boolean = jobRunningLock.compareAndSet(false, true)

    fun releaseJobLock() {
        jobRunningLock.set(false)
    }

    /**
     * Executed by [VideoEnhancementForegroundService] in background-protected foreground execution.
     */
    suspend fun executeCurrentJob(
        onProgressUpdate: (ProcessingProgress) -> Unit
    ): Result<EnhancementResult> {
        val current = _uiState.value
        val metadata = current.selectedVideo
            ?: return Result.failure(IllegalStateException("No video selected."))

        val mode = current.selectedMode
        val profiles = current.profiles.ifEmpty { profileRepository.loadProfiles() }
        val profile = resolveProfile(profiles, mode)
        val caps = capabilityDetector.detect()

        val initialProgress = ProcessingProgress(
            stage = ProcessingStage.IMPORTING,
            overallProgress = 0.02f,
            statusDetail = "Initializing ${mode.title} background pipeline...",
            thermalState = caps.thermalState,
            activeAcceleration = onnxEngine.providerLabel
        )
        persistRunningSnapshot(mode, initialProgress)

        _uiState.update {
            it.copy(
                deviceCapabilities = caps,
                profiles = profiles,
                isProcessing = true,
                enhancementResult = null,
                saveSuccessMessage = null,
                errorMessage = null,
                progress = initialProgress
            )
        }
        onProgressUpdate(initialProgress)

        return try {
            val res = pipeline.processVideo(
                metadata = metadata,
                mode = mode,
                profile = profile,
                capabilities = caps,
                onProgress = { progress ->
                    _uiState.update { state -> state.copy(progress = progress) }
                    persistRunningSnapshot(mode, progress)
                    onProgressUpdate(progress)
                }
            )
            res.fold(
                onSuccess = { enhancementResult ->
                    clearRunningSnapshot()
                    _uiState.update {
                        it.copy(
                            isProcessing = false,
                            enhancementResult = enhancementResult,
                            errorMessage = null
                        )
                    }
                    Result.success(enhancementResult)
                },
                onFailure = { err ->
                    clearRunningSnapshot()
                    val isCancelled = err is CancellationException || err is InterruptedException
                    val msg = if (isCancelled) "Processing cancelled." else (err.message ?: "Video processing failed.")
                    val stage = if (isCancelled) ProcessingStage.CANCELLED else ProcessingStage.ERROR
                    _uiState.update {
                        it.copy(
                            isProcessing = false,
                            progress = ProcessingProgress(stage = stage, statusDetail = msg),
                            errorMessage = if (stage == ProcessingStage.ERROR) msg else null
                        )
                    }
                    Result.failure(err)
                }
            )
        } catch (cancelled: CancellationException) {
            clearRunningSnapshot()
            _uiState.update {
                it.copy(
                    isProcessing = false,
                    progress = ProcessingProgress(stage = ProcessingStage.CANCELLED, statusDetail = "Processing cancelled.")
                )
            }
            Result.failure(cancelled)
        } catch (t: Throwable) {
            clearRunningSnapshot()
            _uiState.update {
                it.copy(
                    isProcessing = false,
                    progress = ProcessingProgress(stage = ProcessingStage.ERROR, statusDetail = t.message ?: "Error"),
                    errorMessage = t.message ?: "Unexpected processing error."
                )
            }
            Result.failure(t)
        }
    }

    fun markServiceTimeoutInterruption(reason: String) {
        pipeline.cancel()
        prefs.edit()
            .putBoolean(KEY_WAS_ACTIVE, false)
            .putString(KEY_INTERRUPTED_NOTICE, reason)
            .apply()
        _uiState.update {
            it.copy(
                isProcessing = false,
                progress = ProcessingProgress(stage = ProcessingStage.ERROR, statusDetail = reason),
                errorMessage = reason
            )
        }
    }

    fun saveToGallery() {
        val result = _uiState.value.enhancementResult ?: return
        if (_uiState.value.isSavingToGallery) return

        scope.launch {
            try {
                _uiState.update { it.copy(isSavingToGallery = true, errorMessage = null, saveSuccessMessage = null) }
                val name = "${result.originalMetadata.displayName.substringBeforeLast('.')}_${result.mode.label}"
                val saveRes = GallerySaver.saveToMoviesGallery(appContext, result.outputFile, name)
                saveRes.fold(
                    onSuccess = { savedUri ->
                        _uiState.update {
                            it.copy(
                                isSavingToGallery = false,
                                enhancementResult = result.copy(savedGalleryUri = savedUri),
                                saveSuccessMessage = "Saved to Movies/ClearKit (${result.outputDimensions.label})"
                            )
                        }
                    },
                    onFailure = { err ->
                        _uiState.update {
                            it.copy(
                                isSavingToGallery = false,
                                errorMessage = err.message ?: "Failed to save video to gallery."
                            )
                        }
                    }
                )
            } catch (ce: CancellationException) {
                throw ce
            } catch (t: Throwable) {
                _uiState.update {
                    it.copy(
                        isSavingToGallery = false,
                        errorMessage = t.message ?: "Failed to save video to gallery."
                    )
                }
            }
        }
    }

    fun saveToDocumentUri(destUri: Uri) {
        val result = _uiState.value.enhancementResult ?: return
        scope.launch {
            try {
                _uiState.update { it.copy(isSavingToGallery = true, errorMessage = null, saveSuccessMessage = null) }
                val copyRes = GallerySaver.copyToUri(appContext, result.outputFile, destUri)
                copyRes.fold(
                    onSuccess = {
                        _uiState.update {
                            it.copy(
                                isSavingToGallery = false,
                                enhancementResult = result.copy(savedGalleryUri = destUri),
                                saveSuccessMessage = "Enhanced video exported successfully."
                            )
                        }
                    },
                    onFailure = { err ->
                        _uiState.update {
                            it.copy(
                                isSavingToGallery = false,
                                errorMessage = err.message ?: "Failed to export video."
                            )
                        }
                    }
                )
            } catch (ce: CancellationException) {
                throw ce
            } catch (t: Throwable) {
                _uiState.update {
                    it.copy(
                        isSavingToGallery = false,
                        errorMessage = t.message ?: "Failed to export video."
                    )
                }
            }
        }
    }

    fun clearError() {
        clearInterruptedNotice()
        _uiState.update { it.copy(errorMessage = null) }
    }

    fun rebenchmarkHardware() {
        if (_uiState.value.isProcessing || jobRunningLock.get() || _uiState.value.isBenchmarkingHardware) return
        scope.launch {
            _uiState.update { it.copy(isBenchmarkingHardware = true, errorMessage = null) }
            try {
                val caps = capabilityDetector.detect()
                val health = pipeline.initializeAndCheckHealth(caps, forceRebenchmark = true)
                _uiState.update {
                    it.copy(
                        deviceCapabilities = caps,
                        engineHealth = health,
                        isBenchmarkingHardware = false
                    )
                }
            } catch (ce: CancellationException) {
                _uiState.update { it.copy(isBenchmarkingHardware = false) }
                throw ce
            } catch (t: Throwable) {
                Log.e(TAG, "Hardware benchmark failed safely", t)
                _uiState.update {
                    it.copy(
                        isBenchmarkingHardware = false,
                        errorMessage = t.message ?: "Hardware benchmark failed."
                    )
                }
            }
        }
    }

    fun onTrimMemory() {
        if (!jobRunningLock.get()) {
            runCatching { onnxEngine.releaseWorkBuffers() }
            runCatching { frameProcessor.resetTemporalState() }
        }
    }

    private fun persistRunningSnapshot(mode: EnhancementMode, progress: ProcessingProgress) {
        prefs.edit()
            .putBoolean(KEY_WAS_ACTIVE, true)
            .putString(KEY_SELECTED_MODE, mode.id)
            .putFloat(KEY_LAST_PROGRESS, progress.overallProgress)
            .putString(KEY_LAST_STAGE, progress.stage.title)
            .apply()
    }

    private fun clearRunningSnapshot() {
        prefs.edit()
            .putBoolean(KEY_WAS_ACTIVE, false)
            .remove(KEY_LAST_PROGRESS)
            .remove(KEY_LAST_STAGE)
            .apply()
    }

    private fun clearInterruptedNotice() {
        prefs.edit().remove(KEY_INTERRUPTED_NOTICE).apply()
    }

    private fun resolveProfile(
        profiles: Map<EnhancementMode, com.example.model.EnhancementProfile>,
        mode: EnhancementMode
    ): com.example.model.EnhancementProfile {
        return profiles[mode] ?: profileRepository.loadProfiles().getValue(mode)
    }

    companion object {
        private const val TAG = "ClearKitCoordinator"
        private const val PREFS_NAME = "luminasr_session_state"
        private const val KEY_SELECTED_VIDEO_URI = "selected_video_uri"
        private const val KEY_SELECTED_MODE = "selected_mode"
        private const val KEY_WAS_ACTIVE = "was_active"
        private const val KEY_LAST_PROGRESS = "last_progress"
        private const val KEY_LAST_STAGE = "last_stage"
        private const val KEY_INTERRUPTED_NOTICE = "interrupted_notice"

        @Volatile
        private var INSTANCE: EnhancementSessionCoordinator? = null

        fun getInstance(context: Context): EnhancementSessionCoordinator {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: EnhancementSessionCoordinator(context.applicationContext).also {
                    INSTANCE = it
                }
            }
        }
    }
}
