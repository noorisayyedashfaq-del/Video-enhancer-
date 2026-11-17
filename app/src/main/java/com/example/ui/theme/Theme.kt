package com.example.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable

private val LuminaDarkColorScheme = darkColorScheme(
    primary = CyanMaster,
    onPrimary = OnCyanMaster,
    primaryContainer = CyanMasterContainer,
    onPrimaryContainer = CyanMaster,
    secondary = AmberFourK,
    onSecondary = StudioObsidian,
    secondaryContainer = AmberFourKContainer,
    onSecondaryContainer = AmberFourK,
    background = StudioObsidian,
    onBackground = CoolWhite,
    surface = StudioSlate,
    onSurface = CoolWhite,
    surfaceVariant = StudioSurfaceElevated,
    onSurfaceVariant = MutedSteel,
    outline = StudioBorder,
    error = StatusError,
    onError = CoolWhite
)

@Composable
fun LuminaSRTheme(
    content: @Composable () -> Unit
) {
    MaterialTheme(
        colorScheme = LuminaDarkColorScheme,
        typography = Typography,
        content = content
    )
}
