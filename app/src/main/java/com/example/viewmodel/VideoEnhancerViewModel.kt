package com.example.viewmodel

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.engine.DeviceCapabilityDetector
import com.example.engine.EnhancementProfileRepository
import com.example.engine.GallerySaver
import com.example.engine.OnnxAiInferenceEngine
import com.example.engine.OpenCvFrameProcessor
import com.example.engine.ResolutionCalculator
import com.example.engine.SmartAutoAdjustmentEngine
import com.example.engine.VideoMediaPipeline
import com.example.model.EnhancementMode
import com.example.model.MainUiState
import com.example.model.ProcessingProgress
import com.example.model.ProcessingStage
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

class VideoEnhancerViewModel(application: Application) : AndroidViewModel(application) {

    private val capabilityDetector = DeviceCapabilityDetector(application)
    private val profileRepository = EnhancementProfileRepository(application)
    private val onnxEngine = OnnxAiInferenceEngine(application)
    private val frameProcessor = OpenCvFrameProcessor(application, onnxEngine)
    private val smartAdjuster = SmartAutoAdjustmentEngine()
    private val pipeline = VideoMediaPipeline(
        context = application,
        capabilityDetector = capabilityDetector,
        onnxEngine = onnxEngine,
        frameProcessor = frameProcessor,
        smartAdjuster = smartAdjuster
    )

    private val _uiState = MutableStateFlow(MainUiState())
    val uiState: StateFlow<MainUiState> = _uiState.asStateFlow()

    private var activeJob: Job? = null

    init {
        initializeEngines()
    }

    private fun initializeEngines() {
        viewModelScope.launch {
            val caps = capabilityDetector.detect()
            val profiles = profileRepository.loadProfiles()
            val health = pipeline.initializeAndCheckHealth(caps)
            _uiState.update {
                it.copy(
                    deviceCapabilities = caps,
                    profiles = profiles,
                    engineHealth = health
                )
            }
        }
    }

    fun onVideoSelected(uri: Uri) {
        if (_uiState.value.isProcessing) return
        viewModelScope.launch {
            _uiState.update {
                it.copy(
                    errorMessage = null,
                    saveSuccessMessage = null,
                    enhancementResult = null,
                    progress = ProcessingProgress(stage = ProcessingStage.IMPORTING, statusDetail = "Reading video metadata...")
                )
            }
            val caps = capabilityDetector.detect()
            val result = pipeline.importAndProbeVideo(uri)
            result.fold(
                onSuccess = { metadata ->
                    val profiles = _uiState.value.profiles.ifEmpty { profileRepository.loadProfiles() }
                    val dimsMap = EnhancementMode.entries.associateWith { mode ->
                        val profile = profiles.getValue(mode)
                        ResolutionCalculator.calculate(metadata, mode, profile, caps)
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
        }
    }

    fun onSelectMode(mode: EnhancementMode) {
        if (_uiState.value.isProcessing) return
        _uiState.update { it.copy(selectedMode = mode, errorMessage = null) }
    }

    fun onStartEnhancement() {
        val current = _uiState.value
        val metadata = current.selectedVideo ?: run {
            _uiState.update { it.copy(errorMessage = "Please select a video first.") }
            return
        }
        if (current.isProcessing) return

        val mode = current.selectedMode
        val profiles = current.profiles.ifEmpty { profileRepository.loadProfiles() }
        val profile = profiles.getValue(mode)
        val caps = capabilityDetector.detect()

        activeJob?.cancel()
        activeJob = viewModelScope.launch {
            _uiState.update {
                it.copy(
                    deviceCapabilities = caps,
                    isProcessing = true,
                    enhancementResult = null,
                    saveSuccessMessage = null,
                    errorMessage = null,
                    progress = ProcessingProgress(
                        stage = ProcessingStage.IMPORTING,
                        overallProgress = 0.02f,
                        statusDetail = "Initializing ${mode.label} pipeline...",
                        thermalState = caps.thermalState,
                        activeAcceleration = onnxEngine.providerLabel
                    )
                )
            }

            try {
                val res = pipeline.processVideo(
                    metadata = metadata,
                    mode = mode,
                    profile = profile,
                    capabilities = caps,
                    onProgress = { progress ->
                        _uiState.update { state -> state.copy(progress = progress) }
                    }
                )
                res.fold(
                    onSuccess = { enhancementResult ->
                        _uiState.update {
                            it.copy(
                                isProcessing = false,
                                enhancementResult = enhancementResult,
                                errorMessage = null
                            )
                        }
                    },
                    onFailure = { err ->
                        val msg = if (err is CancellationException || err is InterruptedException) {
                            "Processing cancelled."
                        } else {
                            err.message ?: "Video processing failed."
                        }
                        val stage = if (err is CancellationException || err is InterruptedException) {
                            ProcessingStage.CANCELLED
                        } else {
                            ProcessingStage.ERROR
                        }
                        _uiState.update {
                            it.copy(
                                isProcessing = false,
                                progress = ProcessingProgress(stage = stage, statusDetail = msg),
                                errorMessage = if (stage == ProcessingStage.ERROR) msg else null
                            )
                        }
                    }
                )
            } catch (_: CancellationException) {
                _uiState.update {
                    it.copy(
                        isProcessing = false,
                        progress = ProcessingProgress(stage = ProcessingStage.CANCELLED, statusDetail = "Processing cancelled.")
                    )
                }
            } catch (t: Throwable) {
                _uiState.update {
                    it.copy(
                        isProcessing = false,
                        progress = ProcessingProgress(stage = ProcessingStage.ERROR, statusDetail = t.message ?: "Error"),
                        errorMessage = t.message ?: "Unexpected processing error."
                    )
                }
            }
        }
    }

    fun onCancelEnhancement() {
        pipeline.cancel()
        activeJob?.cancel()
        _uiState.update {
            it.copy(
                isProcessing = false,
                progress = ProcessingProgress(stage = ProcessingStage.CANCELLED, statusDetail = "Cancelled by user")
            )
        }
    }

    fun onSaveToGallery() {
        val result = _uiState.value.enhancementResult ?: return
        if (_uiState.value.isSavingToGallery) return

        viewModelScope.launch {
            _uiState.update { it.copy(isSavingToGallery = true, errorMessage = null, saveSuccessMessage = null) }
            val name = "${result.originalMetadata.displayName.substringBeforeLast('.')}_${result.mode.label}"
            val saveRes = GallerySaver.saveToMoviesGallery(getApplication(), result.outputFile, name)
            saveRes.fold(
                onSuccess = { savedUri ->
                    _uiState.update {
                        it.copy(
                            isSavingToGallery = false,
                            enhancementResult = result.copy(savedGalleryUri = savedUri),
                            saveSuccessMessage = "Saved to Movies/LuminaSR (${result.outputDimensions.label})"
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
        }
    }

    fun onSaveToDocumentUri(destUri: Uri) {
        val result = _uiState.value.enhancementResult ?: return
        viewModelScope.launch {
            _uiState.update { it.copy(isSavingToGallery = true, errorMessage = null, saveSuccessMessage = null) }
            val copyRes = GallerySaver.copyToUri(getApplication(), result.outputFile, destUri)
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
        }
    }

    fun clearError() {
        _uiState.update { it.copy(errorMessage = null) }
    }

    override fun onCleared() {
        super.onCleared()
        pipeline.cancel()
        onnxEngine.close()
    }
}
