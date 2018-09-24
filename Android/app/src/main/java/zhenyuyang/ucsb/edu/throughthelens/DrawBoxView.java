package zhenyuyang.ucsb.edu.throughthelens;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;

/**
 * Created by Zhenyu on 2018-01-29.
 */

public class DrawBoxView extends View {

    private float xDown = 0,yDown = 0, xUp = 0, yUp = 0;
    public static float[] coordinate = {0, 0, 0, 0};
    private int i = 0;
    Paint mPaint;
    boolean touched = false;


    float viewWidth = 10;
    float viewHeight = 10;

//    final Handler handler = new Handler() {
//        public void handleMessage(Message msg) {
//            // 要做的事情
//            super.handleMessage(msg);
//        }
//    };

    public DrawBoxView(Context context)
    {
        super(context);
        mPaint = new Paint();
        mPaint.setColor(Color.YELLOW);
        mPaint.setStrokeWidth(5.0f);
        mPaint.setStyle(Paint.Style.STROKE);

//        new Thread(new Runnable() {
//            @Override
//            public void run() {
//                i++;
//            }
//        }).start();


//        final View view = findViewById(R.id.draw_box_view);
//        ViewTreeObserver viewTreeObserver =view.getViewTreeObserver();
//        if (viewTreeObserver.isAlive()) {
//            viewTreeObserver.addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() {
//                @Override
//                public void onGlobalLayout() {
//                    view.getViewTreeObserver().removeOnGlobalLayoutListener(this);
//                    viewWidth = view.getWidth();
//                    viewHeight = view.getHeight();
//
//                }
//            });
//        }

    }

    public DrawBoxView(Context context, AttributeSet attrs)
    {
        super(context, attrs);
        mPaint = new Paint();
        mPaint.setColor(Color.YELLOW);
        mPaint.setStrokeWidth(5.0f);
        mPaint.setStyle(Paint.Style.STROKE);

//        new Thread(new Runnable() {
//            @Override
//            public void run() {
//                i++;
//            }
//        }).start();
    }

    @Override
    protected void onDraw(Canvas canvas)
    {
        canvas.drawColor(Color.TRANSPARENT);
        if(touched)
        {
            mPaint.setColor(Color.YELLOW);
            mPaint.setStrokeWidth(5.0f);
            canvas.drawRect(xDown, yDown, xUp, yUp, mPaint);
        }

        mPaint.setStrokeWidth(5.0f);
        mPaint.setColor(Color.RED);
        canvas.drawRect(ThroughTheLensActivity.boundingBox[0], ThroughTheLensActivity.boundingBox[1], ThroughTheLensActivity.boundingBox[2], ThroughTheLensActivity.boundingBox[3], mPaint);

        mPaint.setTextSize(40);
        mPaint.setStrokeWidth(3.0f);
        mPaint.setColor(Color.BLUE);
        //canvas.drawText(Integer.toString(i) + ". data:" + ThroughTheLensActivity.builder.toString(),200, 200, mPaint);
        //setResultToToast(getContext(), "data BaseFpvView.boundingBox[0]: " + BaseThreeBtnView.boundingBox[0]);

    }

    @Override
    public boolean onTouchEvent (MotionEvent event) {

        viewWidth = findViewById(R.id.draw_box_view).getWidth();
        viewHeight = findViewById(R.id.draw_box_view).getHeight();
        switch (event.getAction()){
            case MotionEvent.ACTION_DOWN:
                xDown = event.getX();
                yDown = event.getY();
                //coordinate[0] = xDown / Resources.getSystem().getDisplayMetrics().widthPixels * 640; //(int) (320 * Resources.getSystem().getDisplayMetrics().density);
                //coordinate[1] = yDown / Resources.getSystem().getDisplayMetrics().widthPixels * 480 / 640 * 480 *2; //(int)(240 * Resources.getSystem().getDisplayMetrics().density);
                coordinate[0] = (xDown / viewWidth)*640; //(int) (320 * Resources.getSystem().getDisplayMetrics().density);
                coordinate[1] = (yDown /viewHeight)*480; //(int)(240 * Resources.getSystem().getDisplayMetrics().density);
                xUp = 0;
                yUp = 0;
                break;
            case MotionEvent.ACTION_MOVE:
                xUp = event.getX();
                yUp = event.getY();
                touched = true;
                break;
            case MotionEvent.ACTION_UP:
                xUp = event.getX();
                yUp = event.getY();
                touched = true;
                //coordinate[2] = xUp / Resources.getSystem().getDisplayMetrics().widthPixels * 640;
                //coordinate[3] = yUp / Resources.getSystem().getDisplayMetrics().widthPixels * 480 / 640 * 480 *2;
                coordinate[2] = (xUp / viewWidth)*640;
                coordinate[3] = (yUp / viewHeight)*480;
                //setResultToToast(getContext(), "viewWidth =  " + viewWidth+"\n viewHeight = "+viewHeight);
                break;
        }

        //Log.i("Rec","LeftTop:" + String.valueOf(coordinate[0]) + "," + String.valueOf(coordinate[1]));
        //Log.i("Rec","RightBottom:" + String.valueOf(coordinate[2]) + "," + String.valueOf(coordinate[3]));
        invalidate();
        return true;
    }


//    public class MyThread implements Runnable {
//        @Override
//        public void run() {
//            // TODO Auto-generated method stub
//            while (true) {
//                try {
//                    Thread.sleep(1000);// 线程暂停10秒，单位毫秒
//                    Message message = new Message();
//                    message.what = 1;
//                    handler.sendMessage(message);// 发送消息
//                } catch (InterruptedException e) {
//                    // TODO Auto-generated catch block
//                    e.printStackTrace();
//                }
//            }
//        }
//    }


}
