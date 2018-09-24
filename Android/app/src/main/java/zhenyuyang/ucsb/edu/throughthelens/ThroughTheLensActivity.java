package zhenyuyang.ucsb.edu.throughthelens;

import android.content.Context;
import android.graphics.SurfaceTexture;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.view.TextureView;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import com.badlogic.gdx.ApplicationListener;
import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.backends.android.AndroidFragmentApplication;
import com.mygdx.game.MyGdxGame;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.TimeUnit;

import dji.common.camera.DJICameraSettingsDef;
import dji.common.error.DJIError;
import dji.common.product.Model;
import dji.common.util.DJICommonCallbacks;
import dji.internal.geofeature.flyforbid.Utils;
import dji.sdk.airlink.DJILBAirLink;
import dji.sdk.base.DJIBaseProduct;
import dji.sdk.camera.DJICamera;
import dji.sdk.codec.DJICodecManager;
import dji.sdk.flightcontroller.DJIFlightController;
import dji.sdk.flightcontroller.DJIFlightControllerDelegate;
import dji.sdk.products.DJIAircraft;
import dji.sdk.sdkmanager.DJISDKManager;
import zhenyuyang.ucsb.edu.throughthelens.common.DJIApplication;
import zhenyuyang.ucsb.edu.throughthelens.gdx.GameFragment;
import zhenyuyang.ucsb.edu.throughthelens.gdx.MyGdxGame2;
import zhenyuyang.ucsb.edu.throughthelens.utils.DJIModuleVerificationUtil;



/**
 * Created by Zhenyu on 2018-01-29.
 */



public class ThroughTheLensActivity extends AppCompatActivity implements TextureView.SurfaceTextureListener,  AndroidFragmentApplication.Callbacks{
    private TextureView mVideoSurface = null;
    private TextView responseTextView = null;
    private TextView textView_test = null;
    private EditText addressEditText = null;
    private EditText portEditText = null;
    private Button button_sendWayPoint = null;
    private Button button_previewWayPoint = null;
    private Button button_video = null;

    Timer timer = new Timer();
    private long timeCounter = 0;
    private long hours = 0;
    private long minutes = 0;
    private long seconds = 0;
    String time = "";
    boolean isCameraRecording = false;

    private DJICamera.CameraReceivedVideoDataCallback mReceivedVideoDataCallback = null;
    private DJILBAirLink.DJIOnReceivedVideoCallback mOnReceivedVideoCallback = null;
    private DJICodecManager mCodecManager = null;
    private DJIBaseProduct mProduct = null;
    private DJIAircraft mAircraft;
    private DJIFlightController mFlightController;
    public static float[] boundingBox = {0,0,0,0};
    public static float[] localization = new float[3];
    public static StringBuilder builder = new StringBuilder();

    private  Thread GSPSocketClientThread;

    private volatile boolean gpsSocketClientIsRunning = true;
    private float gpsSocketClientUpdateInterval = 0.5f;  //seconds

    float[] skeleton = new float[42];
    MyGdxGame myGdxGame;



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_throught_the_lens);

        initUI();



        //libGDX part
        // Create libgdx fragment
        GameFragment libgdxFragment = new GameFragment();


        // Put it inside the framelayout (which is defined in the layout.xml file).
        getSupportFragmentManager().beginTransaction().
                add(R.id.content_framelayout, libgdxFragment).
                commit();



        //for camera recording
        DJICamera camera = DJIApplication.getCameraInstance();
        if (DJIModuleVerificationUtil.isCameraModuleAvailable()) {
            DJIApplication.getProductInstance().getCamera().setCameraMode(
                    DJICameraSettingsDef.CameraMode.RecordVideo,
                    new DJICommonCallbacks.DJICompletionCallback() {
                        @Override
                        public void onResult(DJIError djiError) {
                            //Utils.setResultToToast(getContext(), "SetCameraMode to recordVideo");
                            Toast.makeText(getApplicationContext(), "SetCameraMode to recordVideo, error = "+djiError, Toast.LENGTH_SHORT).show();
                        }
                    }
            );
        }
    }




    private void initUI() {
//        LayoutInflater layoutInflater = (LayoutInflater) getContext().getSystemService(Service.LAYOUT_INFLATER_SERVICE);
//
//        View content = layoutInflater.inflate(R.layout.view_fpv_display, null, false);
//        addView(content, new RelativeLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
//                ViewGroup.LayoutParams.MATCH_PARENT));
//
//        Log.v("TAG","Start to test");

        mVideoSurface = (TextureView) findViewById(R.id.texture_video_previewer_surface);
        textView_test = (TextView)findViewById(R.id.textView_test);
        button_sendWayPoint = (Button)findViewById(R.id.button_sendWayPoint);
        button_previewWayPoint = (Button)findViewById(R.id.button_previeWayPoint);
        button_video = (Button)findViewById(R.id.button_video);

        //register button listeners
        button_sendWayPoint.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                mAircraft = (DJIAircraft) DJISDKManager.getInstance().getDJIProduct();
                mFlightController = DJIApplication.getAircraftInstance().getFlightController();


                ApplicationListener applicationListener = Gdx.app.getApplicationListener();
                myGdxGame =(MyGdxGame) applicationListener;
                localization = myGdxGame.getLocalization();

                float[] messageToSend = new float[4];
                messageToSend[0] = localization[0];
                messageToSend[1] = localization[1];
                messageToSend[2] = localization[2];
                messageToSend[3] = 1;


                byte[] message = FloatArray2ByteArray(messageToSend);

                mFlightController.sendDataToOnboardSDKDevice(message,
                        new DJICommonCallbacks.DJICompletionCallback() {
                            @Override
                            public void onResult(DJIError djiError) {
                                if (djiError == null) {
                                    Toast.makeText(getApplicationContext(), "Success upstream from Mobile Device to OES", Toast.LENGTH_SHORT).show();
                                    //DJIDialog.showDialog(getApplicationContext(),"Success upstream from Mobile Device to OES");
                                } else {
                                    Toast.makeText(getApplicationContext(), "Error on upstream from Mobile Device to OES. Description:" + djiError.getDescription(), Toast.LENGTH_SHORT).show();
                                    //DJIDialog.showDialog(getApplicationContext(), "Error on upstream from Mobile Device to OES. Description:" + djiError.getDescription());
                                }
                            }
                        });
            }
        });


        button_previewWayPoint.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                mAircraft = (DJIAircraft) DJISDKManager.getInstance().getDJIProduct();
                mFlightController = DJIApplication.getAircraftInstance().getFlightController();


                ApplicationListener applicationListener = Gdx.app.getApplicationListener();
                myGdxGame =(MyGdxGame) applicationListener;
                localization = myGdxGame.getLocalization();

                float[] messageToSend = new float[4];
                messageToSend[0] = localization[0];
                messageToSend[1] = localization[1];
                messageToSend[2] = localization[2];
                messageToSend[3] = -1;


                byte[] message = FloatArray2ByteArray(messageToSend);

                mFlightController.sendDataToOnboardSDKDevice(message,
                        new DJICommonCallbacks.DJICompletionCallback() {
                            @Override
                            public void onResult(DJIError djiError) {
                                if (djiError == null) {
                                    Toast.makeText(getApplicationContext(), "Success upstream from Mobile Device to OES", Toast.LENGTH_SHORT).show();
                                    //DJIDialog.showDialog(getApplicationContext(),"Success upstream from Mobile Device to OES");
                                } else {
                                    Toast.makeText(getApplicationContext(), "Error on upstream from Mobile Device to OES. Description:" + djiError.getDescription(), Toast.LENGTH_SHORT).show();
                                    //DJIDialog.showDialog(getApplicationContext(), "Error on upstream from Mobile Device to OES. Description:" + djiError.getDescription());
                                }
                            }
                        });
            }
        });

        button_video.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                if(!isCameraRecording){
                    //Utils.setResultToText(context, mTexInfo, "00:00:00");
                    textView_test.setText("00:00:00");
                    if (DJIModuleVerificationUtil.isCameraModuleAvailable()) {
                        Toast.makeText(getApplicationContext(), "CameraModuleAvailable", Toast.LENGTH_SHORT).show();

                        DJIApplication.getProductInstance().getCamera().getCameraMode(new DJICommonCallbacks.DJICompletionCallbackWith<DJICameraSettingsDef.CameraMode>() {
                            @Override
                            public void onSuccess(DJICameraSettingsDef.CameraMode cameraMode) {
                                Toast.makeText(getApplicationContext(), "cameraMode: "+cameraMode.toString(), Toast.LENGTH_SHORT).show();
                            }

                            @Override
                            public void onFailure(DJIError djiError) {

                            }
                        });



                        DJIApplication.getProductInstance().getCamera().startRecordVideo(
                                new DJICommonCallbacks.DJICompletionCallback() {
                                    @Override
                                    public void onResult(DJIError djiError) {
                                        final DJIError error = djiError;
                                        //success so, start recording
                                        if (null == djiError) {
                                            //Utils.setResultToToast(getContext(), "Start record");
                                            Toast.makeText(getApplicationContext(), "Start record", Toast.LENGTH_SHORT).show();
                                            runOnUiThread (new Thread(new Runnable() {
                                                public void run() {
                                                    button_video.setBackgroundResource(R.color.endRecord);
                                                    button_video.setText("STOP");
                                                }
                                            }));
                                            isCameraRecording = true;
                                            timer = new Timer();
                                            timer.schedule(new TimerTask() {
                                                @Override
                                                public void run() {
                                                    timeCounter = timeCounter + 1;
                                                    hours = TimeUnit.MILLISECONDS.toHours(timeCounter);
                                                    minutes = TimeUnit.MILLISECONDS.toMinutes(timeCounter) - (hours * 60);
                                                    seconds = TimeUnit.MILLISECONDS.toSeconds(timeCounter) - ((hours * 60 * 60) + (minutes * 60));
                                                    time = String.format("%02d:%02d:%02d", hours, minutes, seconds);
                                                    //Utils.setResultToText(context, mTexInfo, time);
                                                    runOnUiThread (new Thread(new Runnable() {
                                                        public void run() {
                                                            textView_test.setText(time);
                                                        }
                                                    }));
                                                }
                                            }, 0, 1);
                                        }
                                        else{
                                            runOnUiThread (new Thread(new Runnable() {
                                                public void run() {
                                                    textView_test.setText("djiError = "+error+" :"+error.getDescription());
                                                }
                                            }));
                                        }

                                    }
                                }
                        );
                    }
                }
                else{
                    //stop recording
                    if (DJIModuleVerificationUtil.isCameraModuleAvailable()) {
                        DJIApplication.getProductInstance().getCamera().stopRecordVideo(
                                new DJICommonCallbacks.DJICompletionCallback() {
                                    @Override
                                    public void onResult(DJIError djiError) {
                                        //Utils.setResultToToast(getContext(), "StopRecord");
                                        Toast.makeText(getApplicationContext(), "Stop Record", Toast.LENGTH_SHORT).show();
                                        isCameraRecording = false;
                                        //Utils.setResultToText(context, mTexInfo, "00:00:00");
                                        runOnUiThread (new Thread(new Runnable() {
                                            public void run() {
                                                textView_test.setText("00:00:00");
                                                button_video.setBackgroundResource(R.color.startRecord);
                                                button_video.setText("RECORD");
                                            }
                                        }));
                                        timer.cancel();
                                        timeCounter = 0;
                                    }
                                }
                        );
                    }

                }
                }

        });


        if (null != mVideoSurface) {
            mVideoSurface.setSurfaceTextureListener(this);

            // This callback is for
            mOnReceivedVideoCallback = new DJILBAirLink.DJIOnReceivedVideoCallback() {
                @Override
                public void onResult(byte[] videoBuffer, int size) {
                    if (mCodecManager != null) {
                        mCodecManager.sendDataToDecoder(videoBuffer, size);
                    }
                }
            };

            mReceivedVideoDataCallback = new DJICamera.CameraReceivedVideoDataCallback() {
                @Override
                public void onResult(byte[] videoBuffer, int size) {
                    if (null != mCodecManager) {
                        mCodecManager.sendDataToDecoder(videoBuffer, size);
                    }
                }
            };
        }
        initSDKCallback();




        DJIAircraft mAircraft = (DJIAircraft) DJISDKManager.getInstance().getDJIProduct();
        DJIFlightController mFlightController = mAircraft.getFlightController();
        mFlightController.setReceiveExternalDeviceDataCallback(new DJIFlightControllerDelegate.FlightControllerReceivedDataFromExternalDeviceCallback() {
            @Override
            public void onResult(byte[] data) {
                float[] dataTemp = toFloatArray(data);

                if(dataTemp[0]>0){
                    for(int i = 1;i<22;i++){
                        skeleton[i-1] = dataTemp[i];
                    }
                }
                else{
                    for(int i = 1;i<22;i++){
                        skeleton[21+i-1] = dataTemp[i];
                    }
                }




//                ((EditText)findViewById(R.id.debug)).setText("data = "+builder.toString());
                if(dataTemp[0]<0){

                    ApplicationListener applicationListener = Gdx.app.getApplicationListener();
                    myGdxGame =(MyGdxGame) applicationListener;
                    myGdxGame.setData(skeleton);

                    //textView_test.setText("data = "+data.toString());
                    String test = "";
                    for(int i = 0;i<skeleton.length;i++){
                        test+=skeleton[i]+",";
                    }
                    //textView_test.setText("test = "+test);


                }

            }
        });
    }

    private void initSDKCallback() {
        try {
            mProduct = DJIApplication.getProductInstance();

            if (mProduct.getModel() != Model.UnknownAircraft) {
                mProduct.getCamera().setDJICameraReceivedVideoDataCallback(mReceivedVideoDataCallback);

            } else {
                mProduct.getAirLink().getLBAirLink().setDJIOnReceivedVideoCallback(mOnReceivedVideoCallback);
            }
        } catch (Exception exception) {}
    }

    @Override
    public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {
        if (mCodecManager == null) {
            mCodecManager = new DJICodecManager(getApplicationContext(), surface, width, height);

        }
    }

    @Override
    public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height) {

    }

    @Override
    public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {
        if (mCodecManager != null) {
            mCodecManager.cleanSurface();
            mCodecManager = null;
        }
        return false;
    }

    @Override
    public void onSurfaceTextureUpdated(SurfaceTexture surface) {
        //videoFrameCount++;
        try {
            //Utils.setResultToToast(getContext(), "tsurface = "+surface.getTimestamp());

            int p = mVideoSurface.getBitmap().getPixel(200,200);

            int R = (p >> 16) & 0xff;
            int G = (p >> 8) & 0xff;
            int B = p & 0xff;
            //textView_test.setText("R = "+R+", G = "+G+", B = "+B);
            //setResultToToast(getContext(), "mVideoSurface = "+mVideoSurface);





            //b = mVideoSurface.getBitmap();
            //            tmpMAT = new Mat (b.getWidth(), b.getHeight(), CvType.CV_8UC1);  //something is null here....


        } catch(Exception ex) {};




    }

    private static float[] toFloatArray(byte[] bytes) {

        ByteBuffer buffer = ByteBuffer.wrap(bytes);
        FloatBuffer fb = buffer.asFloatBuffer();
        float[] floatArray = new float[fb.limit()];
        fb.get(floatArray);
        return floatArray;
    }

    private byte[] FloatArray2ByteArray(float[] values){
        ByteBuffer buffer = ByteBuffer.allocate(4 * values.length);

        for (float value : values){
            buffer.putFloat(value);
        }
        return buffer.array();
    }

    @Override
    public void exit() {

    }
}
