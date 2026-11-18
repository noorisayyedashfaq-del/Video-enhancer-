package com.example.ui

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.ErrorOutline
import androidx.compose.material.icons.filled.Hd
import androidx.compose.material.icons.filled.HighQuality
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.Movie
import androidx.compose.material.icons.filled.OfflineBolt
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.SaveAlt
import androidx.compose.material.icons.filled.VideoLibrary
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.minimumInteractiveComponentSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.example.R
import com.example.model.DeviceCapabilities
import com.example.model.EngineHealthStatus
import com.example.model.EnhancementMode
import com.example.model.EnhancementResult
import com.example.model.MainUiState
import com.example.model.OutputDimensions
import com.example.model.ProcessingProgress
import com.example.model.ProcessingStage
import com.example.model.ThermalState
import com.example.model.VideoMetadata
import com.example.ui.components.OriginalVideoPreviewPlayer
import com.example.ui.components.SynchronizedVideoComparison
import com.example.ui.theme.AmberFourK
import com.example.ui.theme.AmberFourKContainer
import com.example.ui.theme.CyanMaster
import com.example.ui.theme.CyanMasterContainer
import com.example.ui.theme.MutedSteel
import com.example.ui.theme.StatusError
import com.example.ui.theme.StatusSuccess
import com.example.ui.theme.StatusWarning
import com.example.ui.theme.StudioBorder
import com.example.ui.theme.StudioObsidian
import com.example.ui.theme.StudioSlate
import com.example.ui.theme.StudioSurfaceElevated
import com.example.ui.theme.SubtleSteel

@Composable
fun VideoEnhancerScreen(
    uiState: MainUiState,
    onVideoSelected: (Uri) -> Unit,
    onStartWithAd: (String) -> Unit,
    onCancelEnhancement: () -> Unit,
    onDismissError: () -> Unit
) {
    // Zero-permission Photo Picker for video selection (Google Play compliant)
    val photoPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia()
    ) { uri ->
        uri?.let(onVideoSelected)
    }

    Scaffold(
        containerColor = StudioObsidian
    ) { innerPadding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .background(
                    Brush.verticalGradient(
                        listOf(StudioObsidian, StudioSlate.copy(alpha = 0.7f), StudioObsidian)
                    )
                ),
            contentAlignment = Alignment.TopCenter
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .widthIn(max = 720.dp)
                    .verticalScroll(rememberScrollState())
                    .padding(horizontal = 16.dp, vertical = 14.dp),
                verticalArrangement = Arrangement.spacedBy(14.dp)
            ) {
                StudioHeader(
                    health = uiState.engineHealth,
                    capabilities = uiState.deviceCapabilities
                )

                // Error Banner
                AnimatedVisibility(visible = uiState.errorMessage != null) {
                    uiState.errorMessage?.let { msg ->
                        ErrorBanner(message = msg, onDismiss = onDismissError)
                    }
                }

                // 1. Select Video Card
                SelectVideoSection(
                    selectedVideo = uiState.selectedVideo,
                    isProcessing = uiState.isProcessing,
                    onPickVideo = {
                        photoPickerLauncher.launch(
                            PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.VideoOnly)
                        )
                    }
                )

                // 2. Original Video Preview OR 6. Synchronized Before/After Comparison
                ViewportSection(
                    selectedVideo = uiState.selectedVideo,
                    enhancementResult = uiState.enhancementResult
                )

                // 3. Four Enhancement Buttons: HD, HD+, 2K, 4K (all wired to startWithAd)
                EnhancementModesSection(
                    selectedMode = uiState.selectedMode,
                    computedDimensions = uiState.computedOutputDimensions,
                    isProcessing = uiState.isProcessing,
                    onSelectMode = { mode -> onStartWithAd(mode.label) }
                )

                // 4. Start Enhancement / Cancel (wired to startWithAd)
                StartEnhancementSection(
                    canStart = uiState.selectedVideo != null && !uiState.isProcessing,
                    isProcessing = uiState.isProcessing,
                    selectedMode = uiState.selectedMode,
                    outputDims = uiState.computedOutputDimensions[uiState.selectedMode],
                    onStart = { onStartWithAd("ENHANCE") },
                    onCancel = onCancelEnhancement
                )

                // 5. Processing Progress
                AnimatedVisibility(
                    visible = uiState.isProcessing ||
                        (uiState.progress.stage != ProcessingStage.IDLE && uiState.enhancementResult == null)
                ) {
                    ProcessingProgressSection(progress = uiState.progress)
                }

                // Quality Analysis Summary & 7. Save Enhanced Video (wired to startWithAd)
                AnimatedVisibility(visible = uiState.enhancementResult != null) {
                    uiState.enhancementResult?.let { result ->
                        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                            EnhancementReportCard(result = result)
                            SaveEnhancedVideoSection(
                                result = result,
                                isSaving = uiState.isSavingToGallery,
                                saveMessage = uiState.saveSuccessMessage,
                                onSaveToGallery = { onStartWithAd("DOWNLOAD") },
                                onExportAs = { onStartWithAd("EXPORT") }
                            )
                        }
                    }
                }

                Spacer(modifier = Modifier.height(12.dp))
            }
        }
    }
}

@Composable
private fun StudioHeader(
    health: EngineHealthStatus?,
    capabilities: DeviceCapabilities?
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Image(
                painter = painterResource(id = R.drawable.img_clearkit_logo),
                contentDescription = "ClearKit logo",
                modifier = Modifier
                    .size(40.dp)
                    .clip(RoundedCornerShape(10.dp))
                    .border(1.dp, StudioBorder, RoundedCornerShape(10.dp)),
                contentScale = ContentScale.Crop
            )
            Column {
                Text(
                    text = stringResource(id = R.string.app_name),
                    style = MaterialTheme.typography.headlineSmall,
                    color = MaterialTheme.colorScheme.onBackground
                )
                Text(
                    text = stringResource(id = R.string.studio_subtitle),
                    style = MaterialTheme.typography.bodySmall,
                    color = MutedSteel
                )
            }
        }

        Surface(
            color = StudioSurfaceElevated,
            shape = RoundedCornerShape(10.dp),
            border = BorderStroke(1.dp, StudioBorder)
        ) {
            Row(
                modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                Icon(
                    imageVector = Icons.Default.OfflineBolt,
                    contentDescription = "100% offline engine",
                    tint = CyanMaster,
                    modifier = Modifier.size(16.dp)
                )
                Text(
                    text = if (health?.realEsrganModelReady == true) "100% OFFLINE AI" else "INIT...",
                    style = MaterialTheme.typography.labelSmall,
                    color = CyanMaster
                )
                if (capabilities != null) {
                    Text(
                        text = "• ${capabilities.totalRamMb / 1024}GB",
                        style = MaterialTheme.typography.labelSmall,
                        color = MutedSteel
                    )
                }
            }
        }
    }
}

@Composable
private fun SelectVideoSection(
    selectedVideo: VideoMetadata?,
    isProcessing: Boolean,
    onPickVideo: () -> Unit
) {
    Card(
        colors = CardDefaults.cardColors(containerColor = StudioSlate),
        border = BorderStroke(1.dp, StudioBorder),
        shape = RoundedCornerShape(16.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(14.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Row(
                modifier = Modifier.weight(1f),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Surface(
                    color = CyanMasterContainer,
                    shape = RoundedCornerShape(12.dp),
                    modifier = Modifier.size(44.dp)
                ) {
                    Box(contentAlignment = Alignment.Center) {
                        Icon(
                            imageVector = Icons.Default.VideoLibrary,
                            contentDescription = "Video file",
                            tint = CyanMaster
                        )
                    }
                }
                Column {
                    Text(
                        text = selectedVideo?.displayName ?: "No Video Selected",
                        style = MaterialTheme.typography.titleMedium,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                    Text(
                        text = if (selectedVideo != null) {
                            "${selectedVideo.resolutionLabel} • ${String.format("%.1f", selectedVideo.fps)} fps • " +
                                "${selectedVideo.durationMs / 1000}s • ${if (selectedVideo.hasAudio) "Audio" else "Muted"}"
                        } else {
                            "MP4, MOV, MKV, WebM • Zero cloud upload"
                        },
                        style = MaterialTheme.typography.bodySmall,
                        color = MutedSteel,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                }
            }

            Spacer(modifier = Modifier.width(10.dp))

            Button(
                onClick = onPickVideo,
                enabled = !isProcessing,
                colors = ButtonDefaults.buttonColors(
                    containerColor = if (selectedVideo == null) CyanMaster else StudioSurfaceElevated,
                    contentColor = if (selectedVideo == null) StudioObsidian else CyanMaster
                ),
                contentPadding = PaddingValues(horizontal = 16.dp, vertical = 12.dp),
                modifier = Modifier
                    .minimumInteractiveComponentSize()
                    .testTag("select_video_button")
            ) {
                Icon(
                    imageVector = Icons.Default.Movie,
                    contentDescription = null,
                    modifier = Modifier.size(18.dp)
                )
                Spacer(modifier = Modifier.width(6.dp))
                Text(
                    text = if (selectedVideo == null) {
                        stringResource(R.string.action_select_video)
                    } else {
                        stringResource(R.string.action_change_video)
                    },
                    style = MaterialTheme.typography.labelLarge
                )
            }
        }
    }
}

@Composable
private fun ViewportSection(
    selectedVideo: VideoMetadata?,
    enhancementResult: EnhancementResult?
) {
    Card(
        colors = CardDefaults.cardColors(containerColor = StudioSlate),
        border = BorderStroke(1.dp, StudioBorder),
        shape = RoundedCornerShape(16.dp)
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    text = if (enhancementResult != null) {
                        stringResource(R.string.label_before_after)
                    } else {
                        stringResource(R.string.label_original_preview)
                    },
                    style = MaterialTheme.typography.titleMedium
                )
                if (enhancementResult != null) {
                    Text(
                        text = "${enhancementResult.originalMetadata.resolutionLabel} → ${enhancementResult.outputDimensions.label}",
                        style = MaterialTheme.typography.labelMedium,
                        color = CyanMaster
                    )
                }
            }

            Spacer(modifier = Modifier.height(10.dp))

            when {
                enhancementResult != null -> {
                    SynchronizedVideoComparison(
                        originalUri = enhancementResult.originalMetadata.uri,
                        enhancedUri = Uri.fromFile(enhancementResult.outputFile),
                        aspectRatio = enhancementResult.originalMetadata.aspectRatio.toFloat()
                    )
                }
                selectedVideo != null -> {
                    OriginalVideoPreviewPlayer(
                        videoUri = selectedVideo.uri,
                        aspectRatio = selectedVideo.aspectRatio.toFloat()
                    )
                }
                else -> {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .aspectRatio(16f / 9f)
                            .clip(RoundedCornerShape(12.dp))
                            .border(1.dp, StudioBorder, RoundedCornerShape(12.dp))
                    ) {
                        Image(
                            painter = painterResource(id = R.drawable.img_studio_banner),
                            contentDescription = "Offline AI video mastering preview banner",
                            contentScale = ContentScale.Crop,
                            modifier = Modifier.fillMaxSize()
                        )
                        Box(
                            modifier = Modifier
                                .fillMaxSize()
                                .background(
                                    Brush.verticalGradient(
                                        listOf(
                                            StudioObsidian.copy(alpha = 0.35f),
                                            StudioObsidian.copy(alpha = 0.88f)
                                        )
                                    )
                                ),
                            contentAlignment = Alignment.BottomStart
                        ) {
                            Text(
                                text = stringResource(R.string.empty_video_prompt),
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onBackground,
                                modifier = Modifier.padding(16.dp)
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun EnhancementModesSection(
    selectedMode: EnhancementMode,
    computedDimensions: Map<EnhancementMode, OutputDimensions>,
    isProcessing: Boolean,
    onSelectMode: (EnhancementMode) -> Unit
) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(
            text = stringResource(R.string.label_enhancement_mode),
            style = MaterialTheme.typography.titleMedium
        )
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            EnhancementMode.entries.forEach { mode ->
                val isSelected = mode == selectedMode
                val isUltra = mode == EnhancementMode.UHD_4K || mode == EnhancementMode.QHD_2K
                val accentColor = if (mode == EnhancementMode.UHD_4K) AmberFourK else CyanMaster
                val containerColor = when {
                    isSelected && mode == EnhancementMode.UHD_4K -> AmberFourKContainer
                    isSelected -> CyanMasterContainer
                    else -> StudioSlate
                }
                val dims = computedDimensions[mode]

                Surface(
                    color = containerColor,
                    shape = RoundedCornerShape(14.dp),
                    border = BorderStroke(
                        width = if (isSelected) 1.5.dp else 1.dp,
                        color = if (isSelected) accentColor else StudioBorder
                    ),
                    modifier = Modifier
                        .weight(1f)
                        .minimumInteractiveComponentSize()
                        .clip(RoundedCornerShape(14.dp))
                        .clickable(enabled = !isProcessing) { onSelectMode(mode) }
                        .testTag("mode_button_${mode.id.lowercase()}")
                ) {
                    Column(
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 12.dp),
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(4.dp)
                    ) {
                        Icon(
                            imageVector = if (isUltra) Icons.Default.HighQuality else Icons.Default.Hd,
                            contentDescription = "${mode.label} enhancement mode",
                            tint = if (isSelected) accentColor else MutedSteel,
                            modifier = Modifier.size(20.dp)
                        )
                        Text(
                            text = mode.label,
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold,
                            color = if (isSelected) accentColor else MaterialTheme.colorScheme.onSurface
                        )
                        Text(
                            text = mode.shortBadge,
                            style = MaterialTheme.typography.labelSmall,
                            color = if (isSelected) MaterialTheme.colorScheme.onSurface else SubtleSteel,
                            maxLines = 1
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun StartEnhancementSection(
    canStart: Boolean,
    isProcessing: Boolean,
    selectedMode: EnhancementMode,
    outputDims: OutputDimensions?,
    onStart: () -> Unit,
    onCancel: () -> Unit
) {
    if (isProcessing) {
        OutlinedButton(
            onClick = onCancel,
            border = BorderStroke(1.dp, StatusError),
            colors = ButtonDefaults.outlinedButtonColors(contentColor = StatusError),
            shape = RoundedCornerShape(14.dp),
            modifier = Modifier
                .fillMaxWidth()
                .height(54.dp)
                .testTag("cancel_enhancement_button")
        ) {
            Icon(imageVector = Icons.Default.Close, contentDescription = null)
            Spacer(modifier = Modifier.width(8.dp))
            Text(
                text = stringResource(R.string.action_cancel_processing),
                style = MaterialTheme.typography.labelLarge
            )
        }
    } else {
        Button(
            onClick = onStart,
            enabled = canStart,
            colors = ButtonDefaults.buttonColors(
                containerColor = CyanMaster,
                contentColor = StudioObsidian,
                disabledContainerColor = StudioSurfaceElevated,
                disabledContentColor = SubtleSteel
            ),
            shape = RoundedCornerShape(14.dp),
            modifier = Modifier
                .fillMaxWidth()
                .height(54.dp)
                .testTag("start_enhancement_button")
        ) {
            Icon(imageVector = Icons.Default.AutoAwesome, contentDescription = null)
            Spacer(modifier = Modifier.width(8.dp))
            val targetInfo = outputDims?.let {
                if (it.tilesPerFrame > 0) " (${it.aiTileSize}px × ${it.tilesPerFrame} • ${it.label})"
                else " (No AI Tiles • ${it.label})"
            } ?: ""
            Text(
                text = "${stringResource(R.string.action_start_enhancement)} • ${selectedMode.label}$targetInfo",
                style = MaterialTheme.typography.labelLarge
            )
        }
    }
}

@Composable
private fun ProcessingProgressSection(progress: ProcessingProgress) {
    Card(
        colors = CardDefaults.cardColors(containerColor = StudioSlate),
        border = BorderStroke(1.dp, CyanMaster.copy(alpha = 0.45f)),
        shape = RoundedCornerShape(16.dp),
        modifier = Modifier.testTag("processing_progress_card")
    ) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    text = progress.stage.title,
                    style = MaterialTheme.typography.titleMedium,
                    color = CyanMaster
                )
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    if (progress.processingSpeedLabel.isNotBlank()) {
                        Text(
                            text = progress.processingSpeedLabel,
                            style = MaterialTheme.typography.labelSmall,
                            color = AmberFourK
                        )
                    }
                    Text(
                        text = "${(progress.overallProgress * 100).toInt()}%",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                }
            }

            LinearProgressIndicator(
                progress = { progress.overallProgress.coerceIn(0f, 1f) },
                color = CyanMaster,
                trackColor = StudioBorder,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(8.dp)
                    .clip(RoundedCornerShape(4.dp))
            )

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                val detailText = buildString {
                    append(progress.statusDetail.ifBlank { "Working offline..." })
                    if (progress.tileProgressLabel.isNotBlank()) {
                        append(" • ${progress.tileProgressLabel}")
                    }
                }
                Text(
                    text = detailText,
                    style = MaterialTheme.typography.bodySmall,
                    color = MutedSteel,
                    modifier = Modifier.weight(1f),
                    maxLines = 2
                )
                progress.etaSeconds?.let { eta ->
                    if (eta > 0) {
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = "ETA ${eta}s",
                            style = MaterialTheme.typography.labelSmall,
                            color = AmberFourK
                        )
                    }
                }
            }

            if (progress.activeAcceleration.isNotBlank() || progress.memoryUsageMb > 0L) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.Memory,
                        contentDescription = null,
                        tint = SubtleSteel,
                        modifier = Modifier.size(14.dp)
                    )
                    val thermalColor = when (progress.thermalState) {
                        ThermalState.NORMAL -> SubtleSteel
                        ThermalState.WARM -> StatusWarning
                        ThermalState.THROTTLING, ThermalState.CRITICAL -> StatusError
                    }
                    val telemetryLine = buildString {
                        if (progress.activeAcceleration.isNotBlank()) {
                            append(progress.activeAcceleration)
                            append(" • ")
                        }
                        if (progress.memoryUsageMb > 0L) {
                            append("${progress.memoryUsageMb}MB RAM • ")
                        }
                        append("Background Service Active")
                        if (progress.speedFallbackActive) {
                            append(" • Fast Mode")
                        }
                        append(" • ${progress.thermalState.label}")
                    }
                    Text(
                        text = telemetryLine,
                        style = MaterialTheme.typography.labelSmall,
                        color = thermalColor,
                        maxLines = 2,
                        overflow = TextOverflow.Ellipsis
                    )
                }
            }
        }
    }
}

@Composable
private fun EnhancementReportCard(result: EnhancementResult) {
    Card(
        colors = CardDefaults.cardColors(containerColor = StudioSlate),
        border = BorderStroke(1.dp, StudioBorder),
        shape = RoundedCornerShape(16.dp)
    ) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Smart Auto Mastering Report",
                    style = MaterialTheme.typography.titleSmall,
                    color = CyanMaster
                )
                Text(
                    text = "${result.codecUsed} • ${result.processingDurationMs / 1000}s",
                    style = MaterialTheme.typography.labelSmall,
                    color = MutedSteel
                )
            }
            Text(
                text = "Detected: ${result.qualityAnalysis.conditionTags.joinToString(", ")}",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurface
            )
            val appliedLine = buildString {
                append("Applied: ${result.appliedParams.adjustmentSummary}")
                if (result.outputDimensions.tilesPerFrame > 0) {
                    append(" • AI Preset ${result.outputDimensions.aiTileSize}px × ${result.outputDimensions.tilesPerFrame} tiles")
                } else {
                    append(" • Fast OpenCV (No AI Tiles)")
                }
            }
            Text(
                text = appliedLine,
                style = MaterialTheme.typography.bodySmall,
                color = MutedSteel
            )
            result.diagnostics?.let { diag ->
                val tileLabel = when {
                    diag.speedFallbackEngaged -> "GPU/SIMD Speed Fallback"
                    diag.finalTileSize > 0 -> "${diag.finalTileSize}px × ${result.outputDimensions.tilesPerFrame}"
                    else -> "No AI tiles"
                }
                Text(
                    text = "Engine: ${diag.executionProvider} • ${diag.decoderBackend} • $tileLabel • " +
                        "Analysis ${diag.analysisMs}ms${if (diag.analysisReusedFromCache) " (cached)" else ""} • " +
                        "AI ${diag.aiInferenceMs / 1000}s • Encode ${diag.encodingMs / 1000}s • Peak ${diag.peakMemoryMb}MB",
                    style = MaterialTheme.typography.labelSmall,
                    color = SubtleSteel
                )
                if (diag.warnings.isNotEmpty()) {
                    Text(
                        text = "Notices: ${diag.warnings.joinToString(" | ")}",
                        style = MaterialTheme.typography.labelSmall,
                        color = StatusWarning
                    )
                }
            }
        }
    }
}

@Composable
private fun SaveEnhancedVideoSection(
    result: EnhancementResult,
    isSaving: Boolean,
    saveMessage: String?,
    onSaveToGallery: () -> Unit,
    onExportAs: () -> Unit
) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        AnimatedVisibility(visible = saveMessage != null) {
            Surface(
                color = CyanMasterContainer,
                shape = RoundedCornerShape(12.dp),
                border = BorderStroke(1.dp, StatusSuccess)
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(12.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.CheckCircle,
                        contentDescription = null,
                        tint = StatusSuccess
                    )
                    Text(
                        text = saveMessage.orEmpty(),
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                }
            }
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Button(
                onClick = onSaveToGallery,
                enabled = !isSaving,
                colors = ButtonDefaults.buttonColors(
                    containerColor = AmberFourK,
                    contentColor = StudioObsidian
                ),
                shape = RoundedCornerShape(14.dp),
                modifier = Modifier
                    .weight(1f)
                    .height(52.dp)
                    .testTag("save_enhanced_video_button")
            ) {
                if (isSaving) {
                    CircularProgressIndicator(
                        color = StudioObsidian,
                        strokeWidth = 2.dp,
                        modifier = Modifier.size(18.dp)
                    )
                } else {
                    Icon(imageVector = Icons.Default.Download, contentDescription = null)
                }
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = stringResource(R.string.action_save_video),
                    style = MaterialTheme.typography.labelLarge
                )
            }

            OutlinedButton(
                onClick = onExportAs,
                enabled = !isSaving,
                border = BorderStroke(1.dp, StudioBorder),
                colors = ButtonDefaults.outlinedButtonColors(contentColor = MaterialTheme.colorScheme.onSurface),
                shape = RoundedCornerShape(14.dp),
                modifier = Modifier
                    .height(52.dp)
                    .testTag("export_as_button")
            ) {
                Icon(
                    imageVector = Icons.Default.SaveAlt,
                    contentDescription = "Save to custom location"
                )
            }
        }
    }
}

@Composable
private fun ErrorBanner(message: String, onDismiss: () -> Unit) {
    Surface(
        color = StatusError.copy(alpha = 0.14f),
        border = BorderStroke(1.dp, StatusError),
        shape = RoundedCornerShape(12.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 12.dp, vertical = 10.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Row(
                modifier = Modifier.weight(1f),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Icon(
                    imageVector = Icons.Default.ErrorOutline,
                    contentDescription = "Error",
                    tint = StatusError
                )
                Text(
                    text = message,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurface
                )
            }
            TextButton(onClick = onDismiss) {
                Text("Dismiss", color = StatusError)
            }
        }
    }
}
