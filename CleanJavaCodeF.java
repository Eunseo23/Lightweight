package com.reindeercrafts.notificationpeek;

import android.app.Notification;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.SharedPreferences;
import android.preference.PreferenceManager;
import android.service.notification.NotificationListenerService;
import android.service.notification.StatusBarNotification;

import com.reindeercrafts.notificationpeek.blacklist.AppList;
import com.reindeercrafts.notificationpeek.peek.NotificationPeek;
import com.reindeercrafts.notificationpeek.settings.PreferenceKeys;
import com.reindeercrafts.notificationpeek.utils.AccessChecker;



public class NotificationService extends NotificationListenerService {

    private static final String TAG = NotificationService.class.getSimpleName();
    private static final int INVAID_ID = -1;

    public static final String ACTION_DISMISS_NOTIFICATION =
            NotificationActionReceiver.class.getSimpleName() + ".dismiss_notification";

    public static final String ACTION_PREFERENCE_CHANGED =
            NotificationService.class.getSimpleName() + ".preference_changed";

    public static final String ACTION_QUIET_HOUR_CHANGE =
            NotificationService.class.getSimpleName() + ".quiet_hour_changed";


    public static final String EXTRA_PACKAGE_NAME = "PackageName";
    public static final String EXTRA_NOTIFICATION_ID = "NotificationId";
    public static final String EXTRA_NOTIFICATION_TAG = "NotificationTag";

    private int mPeekTimeoutMultiplier;
    private int mSensorTimeoutMultiplier;
    private boolean mShowContent;

    private NotificationHub mNotificationHub;
    private NotificationPeek mNotificationPeek;

    private NotificationActionReceiver mReceiver;

    private AppList mAppList;


    @Override
    public void onCreate() {
        super.onCreate();
        mNotificationHub = NotificationHub.getInstance();
        mNotificationPeek = new NotificationPeek(mNotificationHub, this);

        SharedPreferences preferences = PreferenceManager.getDefaultSharedPreferences(this);
                mPeekTimeoutMultiplier =
                Integer.parseInt(preferences.getString(PreferenceKeys.PREF_PEEK_TIMEOUT, "1"));

                mSensorTimeoutMultiplier =
                Integer.parseInt(preferences.getString(PreferenceKeys.PREF_SENSOR_TIMEOUT, "1"));

                mShowContent = preferences.getBoolean(PreferenceKeys.PREF_ALWAYS_SHOW_CONTENT, false);

        mAppList = AppList.getInstance(this);

        registerNotificationActionReceiver();
    }

    @Override
    public void onNotificationPosted(StatusBarNotification sbn) {

        Notification postedNotification = sbn.getNotification();

        if (postedNotification.tickerText == null ||
                sbn.isOngoing() || !sbn.isClearable() ||
                isInBlackList(sbn)) {
            return;
        }

        if (mAppList.isInQuietHour(sbn.getPostTime())) {
                        mNotificationPeek.unregisterEventListeners();
            return;
        }

        mNotificationHub.addNotification(sbn);

        if (AccessChecker.isDeviceAdminEnabled(this)) {
            mNotificationPeek
                    .showNotification(sbn, false, mPeekTimeoutMultiplier, mSensorTimeoutMultiplier,
                            mShowContent);
        }

    }

    @Override
    public void onNotificationRemoved(StatusBarNotification sbn) {
        mNotificationHub.removeNotification(sbn);

    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        mNotificationPeek.unregisterScreenReceiver();
        unregisterReceiver(mReceiver);
    }

    
    private boolean isInBlackList(StatusBarNotification notification) {

        return notification.getNotification().priority < Notification.PRIORITY_DEFAULT ||
                mAppList.shouldPeekWakeUp(notification);

    }

    private void registerNotificationActionReceiver() {
        mReceiver = new NotificationActionReceiver();
        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction(ACTION_DISMISS_NOTIFICATION);
        intentFilter.addAction(ACTION_PREFERENCE_CHANGED);
        intentFilter.addAction(ACTION_QUIET_HOUR_CHANGE);

                intentFilter.addAction(Intent.ACTION_WALLPAPER_CHANGED);
        registerReceiver(mReceiver, intentFilter);
    }

    public class NotificationActionReceiver extends BroadcastReceiver {


        @Override
        public void onReceive(Context context, Intent intent) {
            String action = intent.getAction();
            if (action.equals(ACTION_DISMISS_NOTIFICATION)) {
                String packageName = intent.getStringExtra(EXTRA_PACKAGE_NAME);
                String tag = intent.getStringExtra(EXTRA_NOTIFICATION_TAG);
                int id = intent.getIntExtra(EXTRA_NOTIFICATION_ID, INVAID_ID);
                cancelNotification(packageName, tag, id);
            } else if (action.equals(ACTION_PREFERENCE_CHANGED)) {
                                String changedKey = intent.getStringExtra(PreferenceKeys.INTENT_ACTION_KEY);
                String newValue = intent.getStringExtra(PreferenceKeys.INTENT_ACTION_NEW_VALUE);

                if (changedKey.equals(PreferenceKeys.PREF_PEEK_TIMEOUT)) {
                    mPeekTimeoutMultiplier = Integer.parseInt(newValue);
                } else if (changedKey.equals(PreferenceKeys.PREF_SENSOR_TIMEOUT)) {
                    mSensorTimeoutMultiplier = Integer.parseInt(newValue);
                } else if (changedKey.equals(PreferenceKeys.PREF_ALWAYS_SHOW_CONTENT)) {
                    mShowContent = Boolean.parseBoolean(newValue);
                } else if (changedKey.equals(PreferenceKeys.PREF_BACKGROUND) ||
                        changedKey.equals(PreferenceKeys.PREF_DIM) ||
                        changedKey.equals(PreferenceKeys.PREF_RADIUS)) {
                    mNotificationPeek.updateBackgroundImageView();
                }
            } else if (action.equals(Intent.ACTION_WALLPAPER_CHANGED)) {
                                mNotificationPeek.updateBackgroundImageView();
            } else if (action.equals(ACTION_QUIET_HOUR_CHANGE)) {
                                mAppList.updateQuietHour();
            }
        }
    }

}