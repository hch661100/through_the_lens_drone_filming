package zhenyuyang.ucsb.edu.throughthelens.utils;

import dji.sdk.products.DJIAircraft;
import dji.sdk.products.DJIHandHeld;
import zhenyuyang.ucsb.edu.throughthelens.common.DJIApplication;

/**
 * Created by Zhenyu on 2018-02-12.
 */

public class DJIModuleVerificationUtil {
    public static boolean isProductModuleAvailable() {
        return (null != DJIApplication.getProductInstance());
    }

    public static boolean isAircraft() {
        return DJIApplication.getProductInstance() instanceof DJIAircraft;
    }

    public static boolean isHandHeld() {
        return DJIApplication.getProductInstance() instanceof DJIHandHeld;
    }
    public static boolean isCameraModuleAvailable() {
        return isProductModuleAvailable() &&
                (null != DJIApplication.getProductInstance().getCamera());
    }

    public static boolean isPlaybackAvailable() {
        return isCameraModuleAvailable() &&
                (null != DJIApplication.getProductInstance().getCamera().getPlayback());
    }

    public static boolean isMediaManagerAvailable() {
        return isCameraModuleAvailable() &&
                (null != DJIApplication.getProductInstance().getCamera().getMediaManager());
    }

    public static boolean isRemoteControllerAvailable() {
        return isProductModuleAvailable() && isAircraft() &&
                (null != DJIApplication.getAircraftInstance().getRemoteController());
    }

    public static boolean isFlightControllerAvailable() {
        return isProductModuleAvailable() && isAircraft() &&
                (null != DJIApplication.getAircraftInstance().getFlightController());
    }

    public static boolean isCompassAvailable() {
        return isFlightControllerAvailable() && isAircraft() &&
                (null != DJIApplication.getAircraftInstance().getFlightController().getCompass());
    }

    public static boolean isFlightLimitationAvailable() {
        return isFlightControllerAvailable() && isAircraft() &&
                (null != DJIApplication.getAircraftInstance().
                        getFlightController().getFlightLimitation());
    }

    public static boolean isGimbalModuleAvailable() {
        return isProductModuleAvailable() &&
                (null != DJIApplication.getProductInstance().getGimbal());
    }

    public static boolean isAirlinkAvailable() {
        return isProductModuleAvailable() &&
                (null != DJIApplication.getProductInstance().getAirLink());
    }

    public static boolean isWiFiAirlinkAvailable() {
        return isAirlinkAvailable() &&
                (null != DJIApplication.getProductInstance().getAirLink().getWiFiLink());
    }

    public static boolean isLBAirlinkAvailable() {
        return isAirlinkAvailable() &&
                (null != DJIApplication.getProductInstance().getAirLink().getLBAirLink());
    }

}