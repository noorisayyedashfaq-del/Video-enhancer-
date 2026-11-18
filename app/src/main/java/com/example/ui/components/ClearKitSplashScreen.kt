package com.example.ui.components

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.min
import com.example.R

/**
 * Branded dark-navy ClearKit launch splash overlay with responsive scaling
 * across compact phones, foldables, and tablets.
 */
@Composable
fun ClearKitSplashScreen(
    visible: Boolean
) {
    AnimatedVisibility(
        visible = visible,
        exit = fadeOut(animationSpec = tween(durationMillis = 360))
    ) {
        BoxWithConstraints(
            modifier = Modifier
                .fillMaxSize()
                .background(
                    Brush.verticalGradient(
                        colors = listOf(
                            Color(0xFF02050E),
                            Color(0xFF061228),
                            Color(0xFF030712)
                        )
                    )
                )
                .testTag("clearkit_splash_screen"),
            contentAlignment = Alignment.Center
        ) {
            val minSide = min(maxWidth, maxHeight)
            val logoSize = (minSide * 0.44f).coerceIn(136.dp, 220.dp)
            val glowSize = logoSize * 1.45f

            // Subtle blue radial glow accent
            Box(
                modifier = Modifier
                    .size(glowSize)
                    .background(
                        Brush.radialGradient(
                            colors = listOf(
                                Color(0x4400A3FF),
                                Color(0x180059D4),
                                Color.Transparent
                            ),
                            center = Offset.Unspecified
                        ),
                        shape = CircleShape
                    )
            )

            Column(
                modifier = Modifier.padding(horizontal = 32.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center
            ) {
                Image(
                    painter = painterResource(id = R.drawable.img_clearkit_logo),
                    contentDescription = "ClearKit splash logo",
                    contentScale = ContentScale.Fit,
                    modifier = Modifier
                        .size(logoSize)
                        .clip(RoundedCornerShape(28.dp))
                )

                Spacer(modifier = Modifier.height(22.dp))

                Text(
                    text = stringResource(id = R.string.app_name),
                    style = MaterialTheme.typography.headlineMedium,
                    fontWeight = FontWeight.Bold,
                    color = Color.White
                )

                Spacer(modifier = Modifier.height(4.dp))

                Text(
                    text = stringResource(id = R.string.studio_subtitle),
                    style = MaterialTheme.typography.bodyMedium,
                    color = Color(0xFF8FB8FF)
                )

                Spacer(modifier = Modifier.height(28.dp))

                LinearProgressIndicator(
                    color = Color(0xFF2FA8FF),
                    trackColor = Color(0xFF102140),
                    modifier = Modifier
                        .width(140.dp)
                        .height(3.dp)
                        .clip(RoundedCornerShape(2.dp))
                )
            }
        }
    }
}
