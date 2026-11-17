package com.example.viewmodel

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import com.example.model.EnhancementMode
import com.example.model.MainUiState
import com.example.service.EnhancementSessionCoordinator
import kotlinx.coroutines.flow.StateFlow

/**
 * Lifecycle-aware UI facade backed by the process-scoped [EnhancementSessionCoordinator]
 * and [com.example.service.VideoEnhancementForegroundService].
 * Minimizing, switching apps, turning off the screen, or recreating Activity never interrupts
 * active video enhancement or resets state to the home screen.
 */
class VideoEnhancerViewModel(application: Application) : AndroidViewModel(application) {

    private val coordinator = EnhancementSessionCoordinator.getInstance(application)
    val uiState: StateFlow<MainUiState> = coordinator.uiState

    fun onVideoSelected(uri: Uri) {
        coordinator.selectVideo(uri)
    }

    fun onSelectMode(mode: EnhancementMode) {
        coordinator.selectMode(mode)
    }

    fun onStartEnhancement() {
        coordinator.startEnhancementViaForegroundService(getApplication())
    }

    fun onCancelEnhancement() {
        coordinator.requestCancelEnhancement(getApplication())
    }

    fun onSaveToGallery() {
        coordinator.saveToGallery()
    }

    fun onSaveToDocumentUri(destUri: Uri) {
        coordinator.saveToDocumentUri(destUri)
    }

    fun clearError() {
        coordinator.clearError()
    }
}
