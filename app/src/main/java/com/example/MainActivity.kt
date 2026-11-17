package com.example

import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.service.EnhancementSessionCoordinator
import com.example.ui.VideoEnhancerScreen
import com.example.ui.theme.LuminaSRTheme
import com.example.viewmodel.VideoEnhancerViewModel

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        EnhancementSessionCoordinator.getInstance(applicationContext)
        setContent {
            LuminaSRTheme {
                val viewModel: VideoEnhancerViewModel = viewModel()
                val uiState by viewModel.uiState.collectAsStateWithLifecycle()

                VideoEnhancerScreen(
                    uiState = uiState,
                    onVideoSelected = viewModel::onVideoSelected,
                    onSelectMode = viewModel::onSelectMode,
                    onStartEnhancement = viewModel::onStartEnhancement,
                    onCancelEnhancement = viewModel::onCancelEnhancement,
                    onSaveToGallery = viewModel::onSaveToGallery,
                    onSaveToDocumentUri = viewModel::onSaveToDocumentUri,
                    onDismissError = viewModel::clearError
                )
            }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
    }
}
