package com.example

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.PlayCircleOutline
import androidx.compose.material.icons.filled.WifiOff
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.lifecycleScope
import com.example.ads.AdGateDecision
import com.example.ads.AdGatePolicy
import com.example.ads.AdManager
import com.example.model.EnhancementMode
import com.example.service.EnhancementSessionCoordinator
import com.example.ui.VideoEnhancerScreen
import com.example.ui.components.ClearKitSplashScreen
import com.example.ui.theme.CyanMaster
import com.example.ui.theme.MutedSteel
import com.example.ui.theme.StatusWarning
import com.example.ui.theme.StudioBorder
import com.example.ui.theme.StudioObsidian
import com.example.ui.theme.StudioSlate
import com.example.viewmodel.VideoEnhancerViewModel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {

    private val viewModel: VideoEnhancerViewModel by viewModels()

    private var showInternetRequiredDialog by mutableStateOf(false)
    private var showAdPrepareNotice by mutableStateOf(false)
    private var isAdTransitionPending by mutableStateOf(false)
    private var showSplashScreen by mutableStateOf(false)

    companion object {
        private var hasShownSplashThisSession: Boolean = false
    }

    private val notificationPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { _ ->
        viewModel.onStartEnhancement()
    }

    private val createDocumentLauncher = registerForActivityResult(
        ActivityResultContracts.CreateDocument("video/mp4")
    ) { uri ->
        uri?.let(viewModel::onSaveToDocumentUri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        val coordinator = EnhancementSessionCoordinator.getInstance(applicationContext)
        AdManager.initialize(applicationContext)

        if (!hasShownSplashThisSession && !coordinator.uiState.value.isProcessing) {
            showSplashScreen = true
            hasShownSplashThisSession = true
        }

        setContent {
            ClearKitRootContent()
        }
    }

    @Composable
    private fun ClearKitRootContent() {
        com.example.ui.theme.LuminaSRTheme {
            val uiState by viewModel.uiState.collectAsStateWithLifecycle()

            LaunchedEffect(showSplashScreen) {
                if (showSplashScreen) {
                    delay(1000L)
                    showSplashScreen = false
                }
            }

            VideoEnhancerScreen(
                uiState = uiState,
                onVideoSelected = viewModel::onVideoSelected,
                onStartWithAd = ::startWithAd,
                onCancelEnhancement = viewModel::onCancelEnhancement,
                onDismissError = viewModel::clearError
            )

            ClearKitSplashScreen(visible = showSplashScreen)

            if (showInternetRequiredDialog) {
                InternetRequiredDialog(
                    onTurnOnInternet = {
                        showInternetRequiredDialog = false
                        AdManager.openWifiSettings(this@MainActivity)
                    },
                    onDismiss = {
                        showInternetRequiredDialog = false
                    }
                )
            }

            if (showAdPrepareNotice) {
                AdPrepareNoticeDialog()
            }
        }
    }

    override fun onResume() {
        super.onResume()
        AdManager.loadAdIfNeeded(applicationContext)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
    }

    /**
     * Single common entry point used by all 5 action buttons:
     * HD, HD+, 2K, 4K, and Download / Enhance Video.
     *
     * Guarantees processing/download starts ONLY after an Interstitial Ad is watched and closed.
     */
    fun startWithAd(quality: String) {
        if (isAdTransitionPending) return

        when (
            AdGatePolicy.evaluate(
                isInternetAvailable = AdManager.isInternetAvailable(this),
                isAdLoaded = AdManager.isAdReady()
            )
        ) {
            AdGateDecision.SHOW_INTERNET_DIALOG -> {
                showInternetRequiredDialog = true
            }

            AdGateDecision.SHOW_AD_LOADING_TOAST -> {
                AdManager.loadAdIfNeeded(applicationContext)
                Toast.makeText(
                    this,
                    getString(R.string.ad_loading_toast),
                    Toast.LENGTH_SHORT
                ).show()
            }

            AdGateDecision.SHOW_AD_THEN_EXECUTE -> {
                isAdTransitionPending = true
                showAdPrepareNotice = true
                val noticeText =
                    "${getString(R.string.ad_notice_line_1)}\n${getString(R.string.ad_notice_line_2)}"
                Toast.makeText(this, noticeText, Toast.LENGTH_LONG).show()

                lifecycleScope.launch {
                    delay(950L)
                    showAdPrepareNotice = false
                    if (isFinishing || isDestroyed || !lifecycle.currentState.isAtLeast(Lifecycle.State.RESUMED)) {
                        isAdTransitionPending = false
                        return@launch
                    }

                    AdManager.showInterstitialAd(
                        activity = this@MainActivity,
                        onAdClosed = {
                            isAdTransitionPending = false
                            executeQualityTask(quality)
                        },
                        onAdFailedToShow = {
                            isAdTransitionPending = false
                            Toast.makeText(
                                this@MainActivity,
                                getString(R.string.ad_loading_toast),
                                Toast.LENGTH_SHORT
                            ).show()
                        }
                    )
                }
            }
        }
    }

    private fun executeQualityTask(quality: String) {
        when (quality.uppercase()) {
            "HD" -> selectQualityAndOptionallyStart(EnhancementMode.HD)
            "HD+" -> selectQualityAndOptionallyStart(EnhancementMode.HD_PLUS)
            "2K" -> selectQualityAndOptionallyStart(EnhancementMode.QHD_2K)
            "4K" -> selectQualityAndOptionallyStart(EnhancementMode.UHD_4K)
            "ENHANCE" -> startEnhancementWithNotificationCheck()
            "DOWNLOAD" -> saveEnhancedVideo()
            "EXPORT" -> exportEnhancedVideoAs()
        }
    }

    private fun selectQualityAndOptionallyStart(mode: EnhancementMode) {
        viewModel.onSelectMode(mode)
        val currentState = viewModel.uiState.value
        if (currentState.selectedVideo != null && !currentState.isProcessing) {
            startEnhancementWithNotificationCheck()
        } else if (currentState.selectedVideo == null) {
            Toast.makeText(
                this,
                "${mode.label} selected. Choose a video to start processing.",
                Toast.LENGTH_SHORT
            ).show()
        }
    }

    private fun startEnhancementWithNotificationCheck() {
        if (viewModel.uiState.value.selectedVideo == null || viewModel.uiState.value.isProcessing) {
            return
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) !=
            PackageManager.PERMISSION_GRANTED
        ) {
            notificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
        } else {
            viewModel.onStartEnhancement()
        }
    }

    private fun saveEnhancedVideo() {
        val result = viewModel.uiState.value.enhancementResult ?: return
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            viewModel.onSaveToGallery()
        } else {
            val suggested = "ClearKit_${result.mode.label}_${System.currentTimeMillis()}.mp4"
            createDocumentLauncher.launch(suggested)
        }
    }

    private fun exportEnhancedVideoAs() {
        val result = viewModel.uiState.value.enhancementResult ?: return
        val suggested = "ClearKit_${result.mode.label}_${System.currentTimeMillis()}.mp4"
        createDocumentLauncher.launch(suggested)
    }
}

@Composable
private fun InternetRequiredDialog(
    onTurnOnInternet: () -> Unit,
    onDismiss: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        containerColor = StudioSlate,
        icon = {
            Icon(
                imageVector = Icons.Default.WifiOff,
                contentDescription = "No internet connection",
                tint = StatusWarning,
                modifier = Modifier.size(28.dp)
            )
        },
        title = {
            Text(
                text = stringResource(R.string.internet_required_title),
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold
            )
        },
        text = {
            Text(
                text = stringResource(R.string.internet_required_message),
                style = MaterialTheme.typography.bodyMedium,
                color = MutedSteel
            )
        },
        confirmButton = {
            Button(
                onClick = onTurnOnInternet,
                colors = ButtonDefaults.buttonColors(
                    containerColor = CyanMaster,
                    contentColor = StudioObsidian
                ),
                modifier = Modifier.testTag("turn_on_internet_button")
            ) {
                Text(
                    text = stringResource(R.string.action_turn_on_internet),
                    fontWeight = FontWeight.SemiBold
                )
            }
        },
        dismissButton = {
            TextButton(
                onClick = onDismiss,
                modifier = Modifier.testTag("dismiss_internet_dialog_button")
            ) {
                Text(
                    text = stringResource(R.string.action_cancel_dialog),
                    color = MutedSteel
                )
            }
        }
    )
}

@Composable
private fun AdPrepareNoticeDialog() {
    Dialog(
        onDismissRequest = {},
        properties = DialogProperties(
            dismissOnBackPress = false,
            dismissOnClickOutside = false
        )
    ) {
        Card(
            colors = CardDefaults.cardColors(containerColor = StudioSlate),
            border = BorderStroke(1.dp, StudioBorder),
            shape = RoundedCornerShape(16.dp),
            modifier = Modifier
                .fillMaxWidth()
                .testTag("ad_prepare_notice_dialog")
        ) {
            Column(
                modifier = Modifier.padding(20.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        imageVector = Icons.Default.PlayCircleOutline,
                        contentDescription = "Preparing Ad",
                        tint = CyanMaster,
                        modifier = Modifier.size(24.dp)
                    )
                    Spacer(modifier = Modifier.width(10.dp))
                    Text(
                        text = stringResource(R.string.ad_notice_line_1),
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                }
                Text(
                    text = stringResource(R.string.ad_notice_line_2),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MutedSteel
                )
                Spacer(modifier = Modifier.height(4.dp))
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    CircularProgressIndicator(
                        color = CyanMaster,
                        strokeWidth = 2.5.dp,
                        modifier = Modifier.size(18.dp)
                    )
                    Text(
                        text = "Opening Ad...",
                        style = MaterialTheme.typography.labelMedium,
                        color = CyanMaster
                    )
                }
            }
        }
    }
}
