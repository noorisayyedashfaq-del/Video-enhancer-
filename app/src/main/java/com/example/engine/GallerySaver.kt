package com.example.engine

import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

object GallerySaver {

    suspend fun saveToMoviesGallery(
        context: Context,
        sourceMp4: File,
        displayName: String
    ): Result<Uri> = withContext(Dispatchers.IO) {
        runCatching {
            if (!sourceMp4.exists() || sourceMp4.length() == 0L) {
                throw IllegalStateException("Enhanced video file is missing or empty.")
            }
            val safeName = displayName
                .substringBeforeLast('.')
                .replace(Regex("[^A-Za-z0-9_-]"), "_")
                .take(40)
                .ifBlank { "LuminaSR_Video" } + ".mp4"

            val resolver = context.contentResolver
            val collection = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                MediaStore.Video.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
            } else {
                MediaStore.Video.Media.EXTERNAL_CONTENT_URI
            }

            val values = ContentValues().apply {
                put(MediaStore.Video.Media.DISPLAY_NAME, safeName)
                put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    put(MediaStore.Video.Media.RELATIVE_PATH, "${Environment.DIRECTORY_MOVIES}/LuminaSR")
                    put(MediaStore.Video.Media.IS_PENDING, 1)
                }
            }

            val uri = resolver.insert(collection, values)
                ?: throw IllegalStateException("Unable to create MediaStore video entry.")

            try {
                resolver.openOutputStream(uri)?.use { out ->
                    sourceMp4.inputStream().use { inp ->
                        inp.copyTo(out)
                    }
                } ?: throw IllegalStateException("Unable to open gallery output stream.")

                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    val publish = ContentValues().apply {
                        put(MediaStore.Video.Media.IS_PENDING, 0)
                    }
                    resolver.update(uri, publish, null, null)
                }
                uri
            } catch (t: Throwable) {
                resolver.delete(uri, null, null)
                throw t
            }
        }
    }

    suspend fun copyToUri(
        context: Context,
        sourceMp4: File,
        destUri: Uri
    ): Result<Unit> = withContext(Dispatchers.IO) {
        runCatching {
            context.contentResolver.openOutputStream(destUri)?.use { out ->
                sourceMp4.inputStream().use { inp ->
                    inp.copyTo(out)
                }
            } ?: throw IllegalStateException("Unable to open target document for saving.")
            Unit
        }
    }
}
