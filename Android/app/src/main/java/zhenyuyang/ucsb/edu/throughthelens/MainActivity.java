package zhenyuyang.ucsb.edu.throughthelens;

import android.Manifest;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.Build;
import android.os.Handler;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.support.v7.widget.Toolbar;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import com.badlogic.gdx.backends.android.AndroidFragmentApplication;
import com.mygdx.game.MyGdxGame;

import dji.common.product.Model;
import dji.sdk.airlink.DJILBAirLink;
import dji.sdk.base.DJIBaseProduct;
import dji.sdk.camera.DJICamera;
import dji.sdk.flightcontroller.DJIFlightController;
import dji.sdk.flightcontroller.DJIFlightControllerDelegate;
import dji.sdk.products.DJIAircraft;
import dji.sdk.sdkmanager.DJISDKManager;
import dji.thirdparty.eventbus.EventBus;
import zhenyuyang.ucsb.edu.throughthelens.common.DJIApplication;
import zhenyuyang.ucsb.edu.throughthelens.gdx.GameFragment;

public class MainActivity extends AppCompatActivity implements DJIBaseProduct.DJIVersionCallback, AndroidFragmentApplication.Callbacks{
	private TextView mTextConnectionStatus;
	private TextView mTextProduct;
	private TextView mTextModelAvailable;
	private Button mBtnOpen;
	//private Button mBtnBluetooth;
	private static boolean connected = false;

	String FLAG_CONNECTION_CHANGE = "com_example_dji_sdkdemo3_connection_change";

	//private static DJIBluetoothProductConnector connector = null;

	private Handler mHandler;
	private Handler mHandlerUI;
	//private HandlerThread mHandlerThread = new HandlerThread("Bluetooth");

	private DJIBaseProduct mProduct;

	private String TAG = "MainActivity";





	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);
		Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
		setSupportActionBar(toolbar);

		FloatingActionButton fab = (FloatingActionButton) findViewById(R.id.fab);
		fab.setOnClickListener(new View.OnClickListener() {
			@Override
			public void onClick(View view) {
				Snackbar.make(view, "Replace with your own action", Snackbar.LENGTH_LONG)
						.setAction("Action", null).show();
			}
		});



		if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
			ActivityCompat.requestPermissions(this,
					new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.VIBRATE,
							Manifest.permission.INTERNET, Manifest.permission.ACCESS_WIFI_STATE,
							Manifest.permission.WAKE_LOCK, Manifest.permission.ACCESS_COARSE_LOCATION,
							Manifest.permission.ACCESS_NETWORK_STATE, Manifest.permission.ACCESS_FINE_LOCATION,
							Manifest.permission.CHANGE_WIFI_STATE, Manifest.permission.MOUNT_UNMOUNT_FILESYSTEMS,
							Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.SYSTEM_ALERT_WINDOW,
							Manifest.permission.READ_PHONE_STATE,
					}
					, 1);
		}


		EventBus.getDefault().register(this);


		initUI();


		        // Create libgdx fragment
//		GameFragment libgdxFragment = new GameFragment();
//		// Put it inside the framelayout (which is defined in the layout.xml file).
//		getSupportFragmentManager().beginTransaction().
//				add(R.id.content_framelayout2, libgdxFragment).
//				commit();

	}
	public void onEventMainThread(int wrapper) {
		//pushView(wrapper);
	}
	private void initUI() {
		Log.v(TAG, "initUI");

		mTextConnectionStatus = (TextView) findViewById(R.id.text_connection_status);
		mTextModelAvailable = (TextView) findViewById(R.id.text_model_available);
		mTextProduct = (TextView) findViewById(R.id.text_product_info);
		mBtnOpen = (Button) findViewById(R.id.btn_open);
		//mBtnBluetooth = (Button) findViewById(R.id.btn_bluetooth);
		//mBtnBluetooth.setEnabled(false);
//        mHandlerThread.start();
//        final long currentTime = System.currentTimeMillis();
//        mHandler = new Handler(mHandlerThread.getLooper()){
//            @Override
//            public void handleMessage(Message msg){
//                switch(msg.what){
//                    case 0:
//                        //connected = DJISampleApplication.getBluetoothConnectStatus();
//                        //connector = DJISDKManager.getInstance().getDJIBluetoothProductConnector();
//                        connector = DJIApplication.getBluetoothProductConnector();
//
//                        if(connector != null){
//                            Toast.makeText(getApplicationContext(), "mBtnBluetooth connector != null", Toast.LENGTH_SHORT).show();
////                            mBtnBluetooth.post(new Runnable() {
////                                @Override
////                                public void run() {
////                                    mBtnBluetooth.setEnabled(true);
////                                }
////                            });
//                            return;
//                        }else if((System.currentTimeMillis()-currentTime)>=5000){
//                            //DJIDialog.showDialog(getContext(),"Fetch Connector failed, reboot if you want to connect the Bluetooth");
//                            Toast.makeText(getApplicationContext(), "Fetch Connector failed, reboot if you want to connect the Bluetooth", Toast.LENGTH_SHORT).show();
//                            return;
//                        }else if(connector == null){
//                            sendEmptyMessageDelayed(0, 100);
//                        }
//                        break;
//                    case 1:
//                        break;
//                    case 2:
//                        break;
//                }
//            }
//        };
//        mHandler.sendEmptyMessage(0);


		mBtnOpen.setOnClickListener(new View.OnClickListener() {
			@Override
			public void onClick(View v) {
				//Log.i(TAG,"mBtnOpen onClick");
				Toast.makeText(getApplicationContext(), "mBtnOpen onClick", Toast.LENGTH_SHORT).show();
				Intent intent = new Intent(getApplicationContext(),ThroughTheLensActivity.class);
				startActivity(intent);

//                if (Utils.isFastDoubleClick()) return;
//                //EventBus.getDefault().post(new SetViewWrapper(R.layout.content_component_list, R.string.activity_component_list, getContext()));
//                int mTitleId = 1122252;
//                try {
//                    EventBus.getDefault().post(new SetViewWrapper(getView(), mTitleId));
//                } catch (ClassNotFoundException e) {
//                    Utils.setResultToToast(getContext(), "e = "+e.toString());
//                    e.printStackTrace();
//                } catch (NoSuchMethodException e) {
//                    e.printStackTrace();
//                    Utils.setResultToToast(getContext(), "e = "+e.toString());
//                } catch (IllegalAccessException e) {
//                    e.printStackTrace();
//                    Utils.setResultToToast(getContext(), "e = "+e.toString());
//                } catch (InvocationTargetException e) {
//                    e.printStackTrace();
//                    Utils.setResultToToast(getContext(), "e = "+e.toString());
//                } catch (InstantiationException e) {
//                    e.printStackTrace();
//                    Utils.setResultToToast(getContext(), "e = "+e.toString());
//                }
			}
		});
//        mBtnBluetooth.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View v) {
//                if(Utils.isFastDoubleClick()) return;
//                EventBus.getDefault().post(new SetViewWrapper(R.layout.content_bluetooth, R.string.component_listview_bluetooth,getContext()));
//
//            }
//        });
	}


	private void refreshSDKRelativeUI() {
		mProduct = DJIApplication.getProductInstance();
		//mProduct = DJISDKManager.getInstance().getDJIProduct();
		Log.d(TAG, "mProduct: " + (mProduct == null? "null" : "unnull") );
		if (null != mProduct && mProduct.isConnected()) {
			mBtnOpen.setEnabled(true);

			String str = mProduct instanceof DJIAircraft ? "DJIAircraft" : "DJIHandHeld";
			mTextConnectionStatus.setText("Status: " + str + " connected");
			mProduct.setDJIVersionCallback(this);
			updateVersion();

			if (null != mProduct.getModel()) {
				mTextProduct.setText("" + mProduct.getModel().getDisplayName());
			} else {
				mTextProduct.setText("Model not recognized");
			}
		} else {
			mBtnOpen.setEnabled(false);

			mTextProduct.setText("Model not recognized");
			mTextConnectionStatus.setText("connection_loose");
		}
	}
	private void updateVersion() {
		String version = null;
		if(mProduct != null) {
			version = mProduct.getFirmwarePackageVersion();
		}

		if(version == null) {
			mTextModelAvailable.setText("N/A"); //Firmware version:
		} else {
			mTextModelAvailable.setText(version); //"Firmware version: " +
		}
	}

	@Override
	public boolean onCreateOptionsMenu(Menu menu) {
		// Inflate the menu; this adds items to the action bar if it is present.
		getMenuInflater().inflate(R.menu.menu_main, menu);
		return true;
	}

	@Override
	public boolean onOptionsItemSelected(MenuItem item) {
		// Handle action bar item clicks here. The action bar will
		// automatically handle clicks on the Home/Up button, so long
		// as you specify a parent activity in AndroidManifest.xml.
		int id = item.getItemId();

		//noinspection SimplifiableIfStatement
		if (id == R.id.action_settings) {
			return true;
		}

		return super.onOptionsItemSelected(item);
	}

	@Override
	public void onProductVersionChange(String s, String s1) {
		updateVersion();
	}
	protected BroadcastReceiver mReceiver = new BroadcastReceiver() {

		@Override
		public void onReceive(Context context, Intent intent) {
			Log.d(TAG, "Comes into the BroadcastReceiver");
			refreshSDKRelativeUI();
		}

	};


	@Override
	public void onResume(){
		super.onResume();
		refreshSDKRelativeUI();
		IntentFilter filter = new IntentFilter();
		filter.addAction(DJIApplication.FLAG_CONNECTION_CHANGE);
		getApplicationContext().registerReceiver(mReceiver, filter);

	}


	@Override
	public void onPause(){
		super.onPause();
		getApplicationContext().unregisterReceiver(mReceiver);
		EventBus.getDefault().unregister(this);
	}

	@Override
	public void exit() {

	}
}
