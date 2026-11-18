package com.example.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.ComponentCallbacks2
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import com.example.MainActivity
import com.example.R
import com.example.model.EnhancementMode
import com.example.model.ProcessingProgress
import com.example.model.ProcessingStage
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlin.math.roundToInt

/**
 * Android Foreground Service (type = mediaProcessing on API 35+) keeping video enhancement
 * reliably active when minimized, app-switched, or when the screen is turned off.
 */
class VideoEnhancementForegroundService : Service() {

    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private lateinit var coordinator: EnhancementSessionCoordinator
    private lateinit var notificationManager: NotificationManager
    private var wakeLock: PowerManager.WakeLock? = null
    private var activeServiceJob: Job? = null
    private var lastNotifiedPct: Int = -1
    private var lastNotifiedStage: ProcessingStage? = null
    private var lastNotifyMs: Long = 0L

    override fun onCreate() {
        super.onCreate()
        coordinator = EnhancementSessionCoordinator.getInstance(applicationContext)
        notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        ensureNotificationChannels()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_CANCEL_ENHANCEMENT -> {
                handleCancelAction()
                return START_NOT_STICKY
            }
            ACTION_START_ENHANCEMENT, null -> {
                handleStartEnhancement()
            }
        }
        return START_NOT_STICKY
    }

    private fun handleStartEnhancement() {
        val currentState = coordinator.uiState.value
        val mode = currentState.selectedMode

        // Prevent duplicate processing jobs
        if (!coordinator.tryAcquireJobLock()) {
            val currentNotification = buildProgressNotification(
                progress = currentState.progress,
                mode = mode
            )
            promoteToForeground(currentNotification)
            return
        }

        val initialProgress = ProcessingProgress(
            stage = ProcessingStage.IMPORTING,
            overallProgress = 0.02f,
            statusDetail = "Preparing ${mode.label} offline processing..."
        )
        promoteToForeground(buildProgressNotification(initialProgress, mode))
        acquirePartialWakeLock()

        activeServiceJob = serviceScope.launch {
            try {
                val result = coordinator.executeCurrentJob { progress ->
                    maybeUpdateProgressNotification(progress, mode)
                }
                result.fold(
                    onSuccess = { res ->
                        postCompletionNotification(
                            title = "ClearKit • Enhancement Complete",
                            body = "${res.mode.label} (${res.outputDimensions.label}) ready in ${res.processingDurationMs / 1000}s • Tap to preview & save"
                        )
                    },
                    onFailure = { err ->
                        val isCancelled = err is CancellationException || err is InterruptedException
                        if (!isCancelled) {
                            postCompletionNotification(
                                title = "ClearKit • Processing Failed",
                                body = err.message ?: "Video enhancement encountered an error."
                            )
                        }
                    }
                )
            } finally {
                coordinator.releaseJobLock()
                releaseWakeLockSafely()
                stopForegroundCompat()
                stopSelf()
            }
        }
    }

    private fun handleCancelAction() {
        coordinator.pipeline.cancel()
        activeServiceJob?.cancel()
        coordinator.releaseJobLock()
        releaseWakeLockSafely()
        stopForegroundCompat()
        stopSelf()
    }

    override fun onTimeout(startId: Int, fgsType: Int) {
        coordinator.markServiceTimeoutInterruption(
            "Background media processing time limit reached by system. Reopen ClearKit to retry."
        )
        activeServiceJob?.cancel()
        coordinator.releaseJobLock()
        releaseWakeLockSafely()
        stopForegroundCompat()
        stopSelf()
    }

    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        if (level >= ComponentCallbacks2.TRIM_MEMORY_RUNNING_LOW) {
            coordinator.onTrimMemory()
        }
    }

    override fun onDestroy() {
        coordinator.releaseJobLock()
        releaseWakeLockSafely()
        serviceScope.cancel()
        super.onDestroy()
    }

    private fun promoteToForeground(notification: Notification) {
        val fgsType = if (Build.VERSION.SDK_INT >= 35) {
            ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROCESSING
        } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
        } else {
            0
        }
        try {
            ServiceCompat.startForeground(
                this,
                NOTIFICATION_ID_PROGRESS,
                notification,
                fgsType
            )
        } catch (_: Throwable) {
            startForeground(NOTIFICATION_ID_PROGRESS, notification)
        }
    }

    private fun stopForegroundCompat() {
        try {
            ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
        } catch (_: Throwable) {
        }
    }

    private fun maybeUpdateProgressNotification(progress: ProcessingProgress, mode: EnhancementMode) {
        val pct = (progress.overallProgress * 100f).roundToInt().coerceIn(0, 100)
        val now = System.currentTimeMillis()
        val stageChanged = progress.stage != lastNotifiedStage
        if (!stageChanged && pct == lastNotifiedPct && now - lastNotifyMs < 750L) {
            return
        }
        lastNotifiedPct = pct
        lastNotifiedStage = progress.stage
        lastNotifyMs = now
        try {
            notificationManager.notify(
                NOTIFICATION_ID_PROGRESS,
                buildProgressNotification(progress, mode)
            )
        } catch (_: Throwable) {
        }
    }

    fun buildProgressNotification(
        progress: ProcessingProgress,
        mode: EnhancementMode
    ): Notification {
        val pct = (progress.overallProgress * 100f).roundToInt().coerceIn(0, 100)
        val resumeIntent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
        }
        val contentPending = PendingIntent.getActivity(
            this,
            REQ_RESUME_APP,
            resumeIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val cancelIntent = Intent(this, VideoEnhancementForegroundService::class.java).apply {
            action = ACTION_CANCEL_ENHANCEMENT
        }
        val cancelPending = PendingIntent.getService(
            this,
            REQ_CANCEL_JOB,
            cancelIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val stageLine = buildString {
            append("${mode.label} • ${progress.stage.title} • $pct%")
            if (progress.processingSpeedLabel.isNotBlank()) {
                append(" (${progress.processingSpeedLabel})")
            }
        }
        val subLine = progress.statusDetail.ifBlank { "${progress.stage.title}..." }

        return NotificationCompat.Builder(this, CHANNEL_ID_PROGRESS)
            .setSmallIcon(R.mipmap.ic_launcher)
            .setContentTitle(getString(R.string.notification_title_processing))
            .setContentText(stageLine)
            .setSubText(subLine)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setSilent(true)
            .setCategory(NotificationCompat.CATEGORY_PROGRESS)
            .setForegroundServiceBehavior(NotificationCompat.FOREGROUND_SERVICE_IMMEDIATE)
            .setProgress(100, pct, progress.stage == ProcessingStage.IMPORTING && pct <= 2)
            .setContentIntent(contentPending)
            .addAction(
                android.R.drawable.ic_menu_close_clear_cancel,
                getString(R.string.notification_action_cancel),
                cancelPending
            )
            .build()
    }

    private fun postCompletionNotification(title: String, body: String) {
        val resumeIntent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
        }
        val contentPending = PendingIntent.getActivity(
            this,
            REQ_RESUME_APP,
            resumeIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        val notification = NotificationCompat.Builder(this, CHANNEL_ID_COMPLETE)
            .setSmallIcon(R.mipmap.ic_launcher)
            .setContentTitle(title)
            .setContentText(body)
            .setStyle(NotificationCompat.BigTextStyle().bigText(body))
            .setAutoCancel(true)
            .setContentIntent(contentPending)
            .build()
        try {
            notificationManager.notify(NOTIFICATION_ID_COMPLETE, notification)
        } catch (_: Throwable) {
        }
    }

    private fun ensureNotificationChannels() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val progressChannel = NotificationChannel(
                CHANNEL_ID_PROGRESS,
                getString(R.string.notification_channel_processing),
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Displays ongoing offline AI video enhancement progress"
                setShowBadge(false)
            }
            val completeChannel = NotificationChannel(
                CHANNEL_ID_COMPLETE,
                "Enhancement Status",
                NotificationManager.IMPORTANCE_DEFAULT
            ).apply {
                description = "Alerts when offline video export finishes"
            }
            notificationManager.createNotificationChannel(progressChannel)
            notificationManager.createNotificationChannel(completeChannel)
        }
    }

    private fun acquirePartialWakeLock() {
        try {
            val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
            wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, WAKE_LOCK_TAG).apply {
                setReferenceCounted(false)
                // Guard timeout 6 hours (matches mediaProcessing window; released immediately on finish)
                acquire(6 * 60 * 60 * 1000L)
            }
        } catch (_: Throwable) {
        }
    }

    private fun releaseWakeLockSafely() {
        try {
            wakeLock?.let { wl ->
                if (wl.isHeld) wl.release()
            }
        } catch (_: Throwable) {
        }
        wakeLock = null
    }

    companion object {
        const val ACTION_START_ENHANCEMENT = "com.example.action.START_ENHANCEMENT"
        const val ACTION_CANCEL_ENHANCEMENT = "com.example.action.CANCEL_ENHANCEMENT"
        const val CHANNEL_ID_PROGRESS = "luminasr_video_processing"
        const val CHANNEL_ID_COMPLETE = "luminasr_processing_complete"
        const val NOTIFICATION_ID_PROGRESS = 1001
        const val NOTIFICATION_ID_COMPLETE = 1002
        private const val REQ_RESUME_APP = 2001
        private const val REQ_CANCEL_JOB = 2002
        private const val WAKE_LOCK_TAG = "LuminaSR:MediaProcessingWakeLock"
    }
}
