package com.example.ads

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.util.Log
import com.google.android.gms.ads.AdError
import com.google.android.gms.ads.AdRequest
import com.google.android.gms.ads.FullScreenContentCallback
import com.google.android.gms.ads.LoadAdError
import com.google.android.gms.ads.MobileAds
import com.google.android.gms.ads.interstitial.InterstitialAd
import com.google.android.gms.ads.interstitial.InterstitialAdLoadCallback
import kotlinx.coroutines.CoroutineExceptionHandler
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import java.util.concurrent.atomic.AtomicBoolean

enum class AdGateDecision {
    SHOW_INTERNET_DIALOG,
    SHOW_AD_LOADING_TOAST,
    SHOW_AD_THEN_EXECUTE
}

/**
 * Pure decision evaluator for the mandatory pre-task Interstitial Ad gate.
 */
object AdGatePolicy {
    fun evaluate(isInternetAvailable: Boolean, isAdLoaded: Boolean): AdGateDecision {
        return when {
            !isInternetAvailable -> AdGateDecision.SHOW_INTERNET_DIALOG
            !isAdLoaded -> AdGateDecision.SHOW_AD_LOADING_TOAST
            else -> AdGateDecision.SHOW_AD_THEN_EXECUTE
        }
    }
}

/**
 * Singleton manager handling Google AdMob initialization, network availability checks,
 * preloading, and full-screen Interstitial Ad presentation.
 */
object AdManager {
    private const val TAG = "LuminaSRAdManager"

    // Google's official Test Interstitial Ad Unit ID
    const val TEST_INTERSTITIAL_AD_UNIT_ID = "ca-app-pub-3940256099942544/1033173712"

    private val adInitHandler = CoroutineExceptionHandler { _, throwable ->
        Log.e(TAG, "Unhandled AdManager init exception", throwable)
    }
    private val adScope = CoroutineScope(SupervisorJob() + Dispatchers.IO + adInitHandler)
    private val mainHandler = Handler(Looper.getMainLooper())
    private val initInProgress = AtomicBoolean(false)

    @Volatile
    private var isSdkInitialized: Boolean = false

    @Volatile
    private var isLoading: Boolean = false

    @Volatile
    private var interstitialAd: InterstitialAd? = null

    fun initialize(context: Context) {
        val appContext = context.applicationContext
        if (isSdkInitialized) {
            loadAdIfNeeded(appContext)
            return
        }
        if (!initInProgress.compareAndSet(false, true)) return

        adScope.launch {
            try {
                MobileAds.initialize(appContext) {
                    isSdkInitialized = true
                    initInProgress.set(false)
                    Log.d(TAG, "AdMob SDK initialized")
                    mainHandler.post {
                        loadAdIfNeeded(appContext)
                    }
                }
            } catch (t: Throwable) {
                initInProgress.set(false)
                Log.e(TAG, "AdMob initialization failed", t)
            }
        }
    }

    fun isAdReady(): Boolean = interstitialAd != null

    fun isAdLoading(): Boolean = isLoading

    fun isSdkReady(): Boolean = isSdkInitialized

    fun loadAdIfNeeded(context: Context) {
        val appContext = context.applicationContext
        if (!isSdkInitialized) {
            // Avoid calling InterstitialAd.load before MobileAds.initialize finishes on cold start
            if (!initInProgress.get()) {
                initialize(appContext)
            }
            return
        }
        if (interstitialAd != null || isLoading) return
        if (!isInternetAvailable(appContext)) {
            Log.d(TAG, "Skipping ad load: internet unavailable")
            return
        }

        mainHandler.post {
            if (interstitialAd != null || isLoading || !isSdkInitialized) return@post
            isLoading = true
            try {
                val adRequest = AdRequest.Builder().build()
                InterstitialAd.load(
                    appContext,
                    TEST_INTERSTITIAL_AD_UNIT_ID,
                    adRequest,
                    object : InterstitialAdLoadCallback() {
                        override fun onAdLoaded(ad: InterstitialAd) {
                            interstitialAd = ad
                            isLoading = false
                            Log.d(TAG, "Test Interstitial Ad loaded and ready")
                        }

                        override fun onAdFailedToLoad(loadAdError: LoadAdError) {
                            interstitialAd = null
                            isLoading = false
                            Log.w(TAG, "Interstitial Ad failed to load: ${loadAdError.message} (code=${loadAdError.code})")
                        }
                    }
                )
            } catch (t: Throwable) {
                interstitialAd = null
                isLoading = false
                Log.e(TAG, "Exception while requesting Interstitial Ad load", t)
            }
        }
    }

    /**
     * Shows the currently loaded Interstitial Ad. The [onAdClosed] callback fires ONLY after the
     * user watches and dismisses the full-screen ad. If showing fails, [onAdFailedToShow] is invoked
     * and the next ad preload begins.
     */
    fun showInterstitialAd(
        activity: Activity,
        onAdClosed: () -> Unit,
        onAdFailedToShow: () -> Unit
    ) {
        val ad = interstitialAd
        if (ad == null) {
            loadAdIfNeeded(activity.applicationContext)
            onAdFailedToShow()
            return
        }

        interstitialAd = null
        ad.fullScreenContentCallback = object : FullScreenContentCallback() {
            override fun onAdDismissedFullScreenContent() {
                Log.d(TAG, "Interstitial Ad dismissed; preloading next ad and starting task")
                loadAdIfNeeded(activity.applicationContext)
                onAdClosed()
            }

            override fun onAdFailedToShowFullScreenContent(adError: AdError) {
                Log.w(TAG, "Interstitial Ad failed to show: ${adError.message}")
                loadAdIfNeeded(activity.applicationContext)
                onAdFailedToShow()
            }

            override fun onAdShowedFullScreenContent() {
                Log.d(TAG, "Interstitial Ad displayed full screen")
            }
        }

        try {
            ad.show(activity)
        } catch (t: Throwable) {
            Log.e(TAG, "Failed to display Interstitial Ad", t)
            loadAdIfNeeded(activity.applicationContext)
            onAdFailedToShow()
        }
    }

    fun isInternetAvailable(context: Context): Boolean {
        return try {
            val cm = context.getSystemService(Context.CONNECTIVITY_SERVICE) as? ConnectivityManager
                ?: return false
            val network = cm.activeNetwork ?: return false
            val capabilities = cm.getNetworkCapabilities(network) ?: return false
            capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET) &&
                (capabilities.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) ||
                    capabilities.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR) ||
                    capabilities.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET) ||
                    capabilities.hasTransport(NetworkCapabilities.TRANSPORT_VPN))
        } catch (t: Throwable) {
            Log.w(TAG, "Failed checking internet connectivity", t)
            false
        }
    }

    fun openWifiSettings(context: Context) {
        val wifiIntent = Intent(Settings.ACTION_WIFI_SETTINGS).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }
        try {
            context.startActivity(wifiIntent)
        } catch (_: Throwable) {
            val fallbackIntent = Intent(Settings.ACTION_WIRELESS_SETTINGS).apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            try {
                context.startActivity(fallbackIntent)
            } catch (_: Throwable) {
                // Settings activity unavailable on restricted device
            }
        }
    }
}
