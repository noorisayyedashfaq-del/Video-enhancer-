package com.example.ui.components

import android.net.Uri
import android.view.TextureView
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CompareArrows
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Replay
import androidx.compose.material3.FilledIconButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Outline
import androidx.compose.ui.graphics.Shape
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.unit.Density
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.LayoutDirection
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.media3.common.MediaItem
import androidx.media3.common.Player
import androidx.media3.exoplayer.ExoPlayer
import com.example.ui.theme.AmberFourK
import com.example.ui.theme.CyanMaster
import com.example.ui.theme.MutedSteel
import com.example.ui.theme.StudioBorder
import com.example.ui.theme.StudioObsidian
import com.example.ui.theme.StudioSlate
import kotlinx.coroutines.delay
import kotlin.math.abs
import kotlin.math.roundToInt

private class HorizontalFractionClipShape(
    private val startFraction: Float,
    private val endFraction: Float
) : Shape {
    override fun createOutline(size: Size, layoutDirection: LayoutDirection, density: Density): Outline {
        val left = (size.width * startFraction).coerceIn(0f, size.width)
        val right = (size.width * endFraction).coerceIn(left, size.width)
        return Outline.Rectangle(Rect(left, 0f, right, size.height))
    }
}

@Composable
fun OriginalVideoPreviewPlayer(
    videoUri: Uri,
    aspectRatio: Float,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val player = remember(videoUri) {
        ExoPlayer.Builder(context).build().apply {
            setMediaItem(MediaItem.fromUri(videoUri))
            repeatMode = Player.REPEAT_MODE_ALL
            prepare()
        }
    }

    var isPlaying by remember { mutableStateOf(false) }
    var positionMs by remember { mutableLongStateOf(0L) }
    var durationMs by remember { mutableLongStateOf(1L) }

    DisposableEffect(player) {
        val listener = object : Player.Listener {
            override fun onIsPlayingChanged(playing: Boolean) {
                isPlaying = playing
            }
            override fun onPlaybackStateChanged(state: Int) {
                if (player.duration > 0) durationMs = player.duration
            }
        }
        player.addListener(listener)
        onDispose {
            player.removeListener(listener)
            player.release()
        }
    }

    LaunchedEffect(player, isPlaying) {
        while (true) {
            positionMs = player.currentPosition.coerceAtLeast(0L)
            if (player.duration > 0) durationMs = player.duration
            delay(80)
        }
    }

    Column(modifier = modifier) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .aspectRatio(aspectRatio.coerceIn(0.45f, 2.2f))
                .clip(RoundedCornerShape(14.dp))
                .background(StudioObsidian)
                .border(1.dp, StudioBorder, RoundedCornerShape(14.dp))
        ) {
            AndroidView(
                factory = { ctx ->
                    TextureView(ctx).also { tv -> player.setVideoTextureView(tv) }
                },
                update = { tv ->
                    player.setVideoTextureView(tv)
                },
                modifier = Modifier.fillMaxSize()
            )
            Surface(
                color = StudioObsidian.copy(alpha = 0.78f),
                shape = RoundedCornerShape(8.dp),
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .padding(10.dp)
            ) {
                Text(
                    text = "ORIGINAL SOURCE",
                    style = MaterialTheme.typography.labelSmall,
                    color = MutedSteel,
                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                )
            }
        }

        Spacer(modifier = Modifier.height(8.dp))

        PlaybackTransportBar(
            isPlaying = isPlaying,
            positionMs = positionMs,
            durationMs = durationMs,
            onTogglePlay = {
                if (player.isPlaying) player.pause() else player.play()
            },
            onSeek = { targetMs ->
                player.seekTo(targetMs)
                positionMs = targetMs
            },
            onRestart = {
                player.seekTo(0L)
                positionMs = 0L
            },
            playTestTag = "preview_play_pause_button"
        )
    }
}

@Composable
fun SynchronizedVideoComparison(
    originalUri: Uri,
    enhancedUri: Uri,
    aspectRatio: Float,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    var splitFraction by remember { mutableFloatStateOf(0.5f) }

    val originalPlayer = remember(originalUri) {
        ExoPlayer.Builder(context).build().apply {
            setMediaItem(MediaItem.fromUri(originalUri))
            volume = 0f // Audio comes cleanly from the enhanced synchronized stream
            repeatMode = Player.REPEAT_MODE_ALL
            prepare()
        }
    }

    val enhancedPlayer = remember(enhancedUri) {
        ExoPlayer.Builder(context).build().apply {
            setMediaItem(MediaItem.fromUri(enhancedUri))
            volume = 1f
            repeatMode = Player.REPEAT_MODE_ALL
            prepare()
        }
    }

    var isPlaying by remember { mutableStateOf(false) }
    var positionMs by remember { mutableLongStateOf(0L) }
    var durationMs by remember { mutableLongStateOf(1L) }

    DisposableEffect(originalPlayer, enhancedPlayer) {
        val listener = object : Player.Listener {
            override fun onIsPlayingChanged(playing: Boolean) {
                isPlaying = playing
            }
            override fun onPlaybackStateChanged(state: Int) {
                if (enhancedPlayer.duration > 0) durationMs = enhancedPlayer.duration
            }
        }
        enhancedPlayer.addListener(listener)
        onDispose {
            enhancedPlayer.removeListener(listener)
            originalPlayer.release()
            enhancedPlayer.release()
        }
    }

    // Continuous frame synchronization loop between Original and Enhanced players
    LaunchedEffect(originalPlayer, enhancedPlayer) {
        while (true) {
            val enhPos = enhancedPlayer.currentPosition.coerceAtLeast(0L)
            val origPos = originalPlayer.currentPosition.coerceAtLeast(0L)
            positionMs = enhPos
            if (enhancedPlayer.duration > 0) durationMs = enhancedPlayer.duration

            if (abs(enhPos - origPos) > 45L) {
                originalPlayer.seekTo(enhPos)
            }
            if (enhancedPlayer.isPlaying != originalPlayer.isPlaying) {
                if (enhancedPlayer.isPlaying) originalPlayer.play() else originalPlayer.pause()
            }
            delay(40)
        }
    }

    Column(modifier = modifier) {
        BoxWithConstraints(
            modifier = Modifier
                .fillMaxWidth()
                .aspectRatio(aspectRatio.coerceIn(0.45f, 2.2f))
                .clip(RoundedCornerShape(14.dp))
                .background(StudioObsidian)
                .border(1.dp, StudioBorder, RoundedCornerShape(14.dp))
                .pointerInput(Unit) {
                    detectHorizontalDragGestures { change, dragAmount ->
                        change.consume()
                        val widthPx = size.width.toFloat().coerceAtLeast(1f)
                        splitFraction = (splitFraction + dragAmount / widthPx).coerceIn(0.05f, 0.95f)
                    }
                }
                .testTag("comparison_viewport")
        ) {
            val widthPx = with(LocalDensity.current) { maxWidth.toPx() }
            val dividerX = (widthPx * splitFraction).roundToInt()

            // Left layer: Original source video clipped [0 .. splitFraction]
            AndroidView(
                factory = { ctx ->
                    TextureView(ctx).also { tv -> originalPlayer.setVideoTextureView(tv) }
                },
                update = { tv -> originalPlayer.setVideoTextureView(tv) },
                modifier = Modifier
                    .fillMaxSize()
                    .clip(HorizontalFractionClipShape(0f, splitFraction))
            )

            // Right layer: Enhanced output video clipped [splitFraction .. 1]
            AndroidView(
                factory = { ctx ->
                    TextureView(ctx).also { tv -> enhancedPlayer.setVideoTextureView(tv) }
                },
                update = { tv -> enhancedPlayer.setVideoTextureView(tv) },
                modifier = Modifier
                    .fillMaxSize()
                    .clip(HorizontalFractionClipShape(splitFraction, 1f))
            )

            // Before / After Pill Badges
            Surface(
                color = StudioObsidian.copy(alpha = 0.80f),
                shape = RoundedCornerShape(8.dp),
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .padding(10.dp)
            ) {
                Text(
                    text = "BEFORE",
                    style = MaterialTheme.typography.labelSmall,
                    color = MutedSteel,
                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                )
            }

            Surface(
                color = StudioObsidian.copy(alpha = 0.82f),
                shape = RoundedCornerShape(8.dp),
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(10.dp)
            ) {
                Text(
                    text = "AFTER • AI",
                    style = MaterialTheme.typography.labelSmall,
                    color = CyanMaster,
                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                )
            }

            // Draggable vertical divider line + 48dp accessible grab handle
            Box(
                modifier = Modifier
                    .offset { IntOffset(dividerX - 24.dp.roundToPx(), 0) }
                    .width(48.dp)
                    .fillMaxHeight()
                    .testTag("comparison_slider_handle"),
                contentAlignment = Alignment.Center
            ) {
                Box(
                    modifier = Modifier
                        .width(2.dp)
                        .fillMaxHeight()
                        .background(CyanMaster)
                )
                Surface(
                    shape = CircleShape,
                    color = StudioSlate,
                    shadowElevation = 6.dp,
                    modifier = Modifier
                        .size(36.dp)
                        .border(2.dp, CyanMaster, CircleShape)
                ) {
                    Box(contentAlignment = Alignment.Center) {
                        Icon(
                            imageVector = Icons.Default.CompareArrows,
                            contentDescription = "Drag to compare original and enhanced video",
                            tint = CyanMaster,
                            modifier = Modifier.size(20.dp)
                        )
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(8.dp))

        PlaybackTransportBar(
            isPlaying = isPlaying,
            positionMs = positionMs,
            durationMs = durationMs,
            onTogglePlay = {
                if (enhancedPlayer.isPlaying) {
                    enhancedPlayer.pause()
                    originalPlayer.pause()
                } else {
                    originalPlayer.seekTo(enhancedPlayer.currentPosition)
                    originalPlayer.play()
                    enhancedPlayer.play()
                }
            },
            onSeek = { targetMs ->
                originalPlayer.seekTo(targetMs)
                enhancedPlayer.seekTo(targetMs)
                positionMs = targetMs
            },
            onRestart = {
                originalPlayer.seekTo(0L)
                enhancedPlayer.seekTo(0L)
                positionMs = 0L
            },
            playTestTag = "comparison_play_pause_button"
        )
    }
}

@Composable
private fun PlaybackTransportBar(
    isPlaying: Boolean,
    positionMs: Long,
    durationMs: Long,
    onTogglePlay: () -> Unit,
    onSeek: (Long) -> Unit,
    onRestart: () -> Unit,
    playTestTag: String
) {
    val safeDuration = durationMs.coerceAtLeast(1L)
    val fraction = (positionMs.toFloat() / safeDuration.toFloat()).coerceIn(0f, 1f)

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        FilledIconButton(
            onClick = onTogglePlay,
            colors = IconButtonDefaults.filledIconButtonColors(
                containerColor = CyanMaster,
                contentColor = StudioObsidian
            ),
            modifier = Modifier
                .size(48.dp)
                .testTag(playTestTag)
        ) {
            Icon(
                imageVector = if (isPlaying) Icons.Default.Pause else Icons.Default.PlayArrow,
                contentDescription = if (isPlaying) "Pause video" else "Play video"
            )
        }

        Slider(
            value = fraction,
            onValueChange = { f -> onSeek((f * safeDuration).toLong()) },
            colors = SliderDefaults.colors(
                thumbColor = CyanMaster,
                activeTrackColor = CyanMaster,
                inactiveTrackColor = StudioBorder
            ),
            modifier = Modifier
                .weight(1f)
                .testTag("${playTestTag}_scrubber")
        )

        Text(
            text = "${formatTime(positionMs)} / ${formatTime(safeDuration)}",
            style = MaterialTheme.typography.labelSmall,
            color = MutedSteel
        )

        FilledIconButton(
            onClick = onRestart,
            colors = IconButtonDefaults.filledIconButtonColors(
                containerColor = StudioSlate,
                contentColor = AmberFourK
            ),
            modifier = Modifier.size(48.dp)
        ) {
            Icon(
                imageVector = Icons.Default.Replay,
                contentDescription = "Restart playback"
            )
        }
    }
}

private fun formatTime(ms: Long): String {
    val totalSec = (ms / 1000L).coerceAtLeast(0L)
    val min = totalSec / 60L
    val sec = totalSec % 60L
    return String.format("%d:%02d", min, sec)
}
