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
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.SaveAlt
import androidx.compose.material.icons.filled.Speed
import androidx.compose.material.icons.filled.VideoLibrary
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
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
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
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
import androidx.compose.ui.window.Dialog
import com.example.R
import com.example.model.AcceleratorKind
import com.example.model.DeviceCapabilities
import com.example.model.EngineHealthStatus
import com.example.model.EnhancementMode
import com.example.model.EnhancementResult
import com.example.model.MainUiState
import com.example.model.ModelKind
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
    onDismissError: () -> Unit,
    onRebenchmarkHardware: () -> Unit = {}
) {
    // Zero-permission Photo Picker for video selection (Google Play compliant)
    val photoPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia()
    ) { uri ->
        uri?.let(onVideoSelected)
    }

    var showHardwareDialog by remember { mutableStateOf(false) }

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
                    capabilities = uiState.deviceCapabilities,
                    isBenchmarking = uiState.isBenchmarkingHardware,
                    onHardwarePillClick = { showHardwareDialog = true }
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

                // 3. Four Enhancement Buttons: HD, HD+, FHD, 2K (all wired to startWithAd)
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

    if (showHardwareDialog) {
        AdaptiveHardwareDiagnosticsDialog(
            health = uiState.engineHealth,
            capabilities = uiState.deviceCapabilities,
            isProcessing = uiState.isProcessing,
            isBenchmarking = uiState.isBenchmarkingHardware,
            onRebenchmark = onRebenchmarkHardware,
            onDismiss = { showHardwareDialog = false }
        )
    }
}

@Composable
private fun StudioHeader(
    health: EngineHealthStatus?,
    capabilities: DeviceCapabilities?,
    isBenchmarking: Boolean,
    onHardwarePillClick: () -> Unit
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp),
            modifier = Modifier.weight(1f)
        ) {
            Image(
                painter = painterResource(id = R.drawable.video_enhancer_logo),
                contentDescription = "Video Enhancer logo",
                modifier = Modifier
                    .size(40.dp)
                    .clip(RoundedCornerShape(10.dp))
                    .border(1.dp, StudioBorder, RoundedCornerShape(10.dp)),
                contentScale = ContentScale.Fit
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

        val pillLabel = when {
            isBenchmarking -> "BENCHMARK..."
            health?.hardwareProfile != null -> health.hardwareProfile.compactSummaryPill()
            health?.realEsrganModelReady == true -> "OFFLINE AI"
            else -> "INIT..."
        }

        Surface(
            color = StudioSurfaceElevated,
            shape = RoundedCornerShape(10.dp),
            border = BorderStroke(1.dp, CyanMaster.copy(alpha = 0.45f)),
            modifier = Modifier
                .minimumInteractiveComponentSize()
                .clip(RoundedCornerShape(10.dp))
                .clickable(onClick = onHardwarePillClick)
                .testTag("hardware_diagnostics_button")
        ) {
            Row(
                modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                if (isBenchmarking) {
                    CircularProgressIndicator(
                        color = CyanMaster,
                        strokeWidth = 2.dp,
                        modifier = Modifier.size(14.dp)
                    )
                } else {
                    Icon(
                        imageVector = Icons.Default.OfflineBolt,
                        contentDescription = "Hardware acceleration status",
                        tint = CyanMaster,
                        modifier = Modifier.size(16.dp)
                    )
                }
                Text(
                    text = pillLabel,
                    style = MaterialTheme.typography.labelSmall,
                    color = CyanMaster,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
                if (!isBenchmarking && capabilities != null && health?.hardwareProfile == null) {
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
                            "${selectedVideo.resolutionLabel} • ${String.format(java.util.Locale.US, "%.1f", selectedVideo.fps)} fps • " +
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
    val rows = EnhancementMode.entries.chunked(2)
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(
            text = stringResource(R.string.label_enhancement_mode),
            style = MaterialTheme.typography.titleMedium
        )
        rows.forEach { rowModes ->
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                rowModes.forEach { mode ->
                    val isSelected = mode == selectedMode
                    val isMax = mode == EnhancementMode.TWO_K
                    val isUltra = mode == EnhancementMode.FHD || isMax
                    val accentColor = if (isMax) AmberFourK else CyanMaster
                    val containerColor = when {
                        isSelected && isMax -> AmberFourKContainer
                        isSelected -> CyanMasterContainer
                        else -> StudioSlate
                    }
                    val dims = computedDimensions[mode]
                    val tileLabel = dims?.let { "${it.aiTileSize}px • ${it.tilesPerFrame}t" } ?: mode.shortBadge

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
                            modifier = Modifier.padding(horizontal = 12.dp, vertical = 12.dp),
                            verticalArrangement = Arrangement.spacedBy(4.dp)
                        ) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Row(
                                    verticalAlignment = Alignment.CenterVertically,
                                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                                ) {
                                    Icon(
                                        imageVector = if (isUltra) Icons.Default.HighQuality else Icons.Default.Hd,
                                        contentDescription = "${mode.title} enhancement mode",
                                        tint = if (isSelected) accentColor else MutedSteel,
                                        modifier = Modifier.size(18.dp)
                                    )
                                    Text(
                                        text = mode.title,
                                        style = MaterialTheme.typography.titleSmall,
                                        fontWeight = FontWeight.Bold,
                                        color = if (isSelected) accentColor else MaterialTheme.colorScheme.onSurface,
                                        maxLines = 1
                                    )
                                }
                                Text(
                                    text = tileLabel,
                                    style = MaterialTheme.typography.labelSmall,
                                    color = if (isSelected) accentColor else SubtleSteel,
                                    maxLines = 1
                                )
                            }
                            Text(
                                text = mode.subtitle,
                                style = MaterialTheme.typography.bodySmall,
                                color = if (isSelected) MaterialTheme.colorScheme.onSurface else MutedSteel,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis
                            )
                        }
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
                " (${it.aiTileSize}px × ${it.tilesPerFrame} • ${it.label})"
            } ?: ""
            Text(
                text = "${stringResource(R.string.action_start_enhancement)} • ${selectedMode.title}$targetInfo",
                style = MaterialTheme.typography.labelLarge,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
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
                append("AI Models: Real-ESRGAN ${if (result.usedAiSuperResolution) "Active" else "Fallback"}")
                append(" • DnCNN ${if (result.usedAiDenoise) "Active" else "Off"}")
                if (result.facesEnhancedCount > 0) {
                    append(" • Face AI (${result.facesEnhancedCount} detections)")
                }
                append(" • ${result.originalMetadata.resolutionLabel} → ${result.outputDimensions.label}")
            }
            Text(
                text = appliedLine,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurface
            )
            Text(
                text = "Applied: ${result.appliedParams.adjustmentSummary} • ${result.mode.title} (${result.outputDimensions.aiTileSize}px × ${result.outputDimensions.tilesPerFrame} tiles)",
                style = MaterialTheme.typography.bodySmall,
                color = MutedSteel
            )
            result.diagnostics?.let { diag ->
                val tileLabel = when {
                    diag.speedFallbackEngaged -> "GPU/SIMD Speed Fallback"
                    diag.finalTileSize > 0 -> "${diag.finalTileSize}px × ${result.outputDimensions.tilesPerFrame} (${diag.upscaleScaleLabel})"
                    else -> "No AI tiles"
                }
                val perModelText = diag.perModelAccelerationSummary.ifBlank { diag.executionProvider }
                Text(
                    text = "Engine: $perModelText • $tileLabel • " +
                        "SR ${diag.aiInferenceMs / 1000}s • DnCNN ${diag.dncnnInferenceMs / 1000}s • " +
                        "Face ${diag.faceAiMs / 1000}s • Encode ${diag.encodingMs / 1000}s • Peak ${diag.peakMemoryMb}MB",
                    style = MaterialTheme.typography.labelSmall,
                    color = SubtleSteel
                )
                if (diag.staticTilesReused > 0 || diag.framesReusedDuplicate > 0 || diag.runtimeFallbacksCount > 0) {
                    val speedStats = buildString {
                        append("Speed Opts: ${diag.staticTilesReused} static tiles reused")
                        if (diag.framesReusedDuplicate > 0) {
                            append(" • ${diag.framesReusedDuplicate} duplicate frames reused")
                        }
                        if (diag.runtimeFallbacksCount > 0) {
                            append(" • ${diag.runtimeFallbacksCount} runtime fallback(s)")
                        }
                    }
                    Text(
                        text = speedStats,
                        style = MaterialTheme.typography.labelSmall,
                        color = CyanMaster
                    )
                }
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

@Composable
private fun AdaptiveHardwareDiagnosticsDialog(
    health: EngineHealthStatus?,
    capabilities: DeviceCapabilities?,
    isProcessing: Boolean,
    isBenchmarking: Boolean,
    onRebenchmark: () -> Unit,
    onDismiss: () -> Unit
) {
    Dialog(onDismissRequest = onDismiss) {
        Card(
            colors = CardDefaults.cardColors(containerColor = StudioSlate),
            border = BorderStroke(1.dp, CyanMaster.copy(alpha = 0.45f)),
            shape = RoundedCornerShape(18.dp),
            modifier = Modifier
                .fillMaxWidth()
                .testTag("hardware_diagnostics_dialog")
        ) {
            Column(
                modifier = Modifier
                    .verticalScroll(rememberScrollState())
                    .padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Speed,
                            contentDescription = "Adaptive Hardware Acceleration",
                            tint = CyanMaster,
                            modifier = Modifier.size(22.dp)
                        )
                        Text(
                            text = stringResource(R.string.hardware_diagnostics_title),
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.onSurface
                        )
                    }
                }

                val profile = health?.hardwareProfile
                val probe = profile?.probeSnapshot ?: capabilities?.hardwareProbeSnapshot

                if (probe != null) {
                    Surface(
                        color = StudioSurfaceElevated,
                        shape = RoundedCornerShape(12.dp),
                        border = BorderStroke(1.dp, StudioBorder)
                    ) {
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(10.dp),
                            verticalArrangement = Arrangement.spacedBy(4.dp)
                        ) {
                            Text(
                                text = "Hardware Capabilities & Drivers",
                                style = MaterialTheme.typography.labelMedium,
                                color = CyanMaster
                            )
                            Text(
                                text = "Vulkan: ${if (probe.vulkanSupported) "Yes (${probe.vulkanVersionLabel}, compute=${probe.vulkanComputeSupported})" else "Unavailable"}",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurface
                            )
                            Text(
                                text = "OpenCL: ${probe.openClSourceLabel} • GLES ${probe.openGlEsMajor}.x",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurface
                            )
                            Text(
                                text = "NNAPI: ${probe.nnapiDriversSummary} • QNN: ${probe.qnnStatusLabel}",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurface
                            )
                            Text(
                                text = "CPU Topology: ${probe.cpuTopology.clusterSummary}",
                                style = MaterialTheme.typography.bodySmall,
                                color = MutedSteel
                            )
                        }
                    }
                }

                HorizontalDivider(color = StudioBorder)

                Text(
                    text = "Per-Model Benchmark Selection (GPU → NPU → CPU)",
                    style = MaterialTheme.typography.labelMedium,
                    color = AmberFourK
                )

                if (profile == null) {
                    Text(
                        text = if (isBenchmarking) {
                            stringResource(R.string.benchmarking_hardware_status)
                        } else {
                            "Benchmark profile not yet populated. Tap Re-Run Benchmark below."
                        },
                        style = MaterialTheme.typography.bodySmall,
                        color = MutedSteel
                    )
                } else {
                    listOf(
                        ModelKind.REAL_ESRGAN,
                        ModelKind.DNCNN,
                        ModelKind.FACE_YUNET
                    ).forEach { modelKind ->
                        val plan = profile.plans[modelKind]
                        Surface(
                            color = StudioSurfaceElevated,
                            shape = RoundedCornerShape(12.dp),
                            border = BorderStroke(1.dp, StudioBorder)
                        ) {
                            Column(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(10.dp),
                                verticalArrangement = Arrangement.spacedBy(4.dp)
                            ) {
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween,
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Text(
                                        text = modelKind.displayName,
                                        style = MaterialTheme.typography.labelLarge,
                                        color = MaterialTheme.colorScheme.onSurface
                                    )
                                    val badgeColor = when (plan?.selectedKind) {
                                        AcceleratorKind.GPU -> CyanMaster
                                        AcceleratorKind.NPU -> AmberFourK
                                        else -> MutedSteel
                                    }
                                    Text(
                                        text = plan?.let { "${it.selectedKind.label} • ${it.measuredLatencyMs}ms" } ?: "CPU",
                                        style = MaterialTheme.typography.labelSmall,
                                        fontWeight = FontWeight.Bold,
                                        color = badgeColor
                                    )
                                }
                                if (plan != null) {
                                    Text(
                                        text = "Active: ${plan.selectedLabel}",
                                        style = MaterialTheme.typography.bodySmall,
                                        color = CyanMaster
                                    )
                                    val evalSummary = plan.evaluatedBackends.joinToString(" | ") { b ->
                                        if (b.medianLatencyMs != null && b.verifiedFullOffload && b.numericalParityOk) {
                                            "${b.kind.label}:${b.medianLatencyMs}ms"
                                        } else {
                                            "${b.displayLabel}:${b.rejectedReason ?: "skip"}"
                                        }
                                    }
                                    Text(
                                        text = evalSummary,
                                        style = MaterialTheme.typography.labelSmall,
                                        color = SubtleSteel,
                                        maxLines = 2,
                                        overflow = TextOverflow.Ellipsis
                                    )
                                }
                            }
                        }
                    }
                }

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Button(
                        onClick = onRebenchmark,
                        enabled = !isProcessing && !isBenchmarking,
                        colors = ButtonDefaults.buttonColors(
                            containerColor = CyanMaster,
                            contentColor = StudioObsidian
                        ),
                        modifier = Modifier
                            .weight(1f)
                            .minimumInteractiveComponentSize()
                            .testTag("rebenchmark_hardware_button")
                    ) {
                        if (isBenchmarking) {
                            CircularProgressIndicator(
                                color = StudioObsidian,
                                strokeWidth = 2.dp,
                                modifier = Modifier.size(16.dp)
                            )
                        } else {
                            Icon(
                                imageVector = Icons.Default.Refresh,
                                contentDescription = null,
                                modifier = Modifier.size(18.dp)
                            )
                        }
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            text = if (isBenchmarking) {
                                stringResource(R.string.benchmarking_hardware_status)
                            } else {
                                stringResource(R.string.action_rebenchmark_hardware)
                            },
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis
                        )
                    }
                    OutlinedButton(
                        onClick = onDismiss,
                        border = BorderStroke(1.dp, StudioBorder),
                        modifier = Modifier
                            .minimumInteractiveComponentSize()
                            .testTag("close_hardware_dialog_button")
                    ) {
                        Text(
                            text = stringResource(R.string.action_close),
                            color = MaterialTheme.colorScheme.onSurface
                        )
                    }
                }
            }
        }
    }
}

