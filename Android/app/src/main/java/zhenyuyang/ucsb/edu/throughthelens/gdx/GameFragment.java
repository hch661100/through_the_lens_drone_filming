package zhenyuyang.ucsb.edu.throughthelens.gdx;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import com.badlogic.gdx.backends.android.AndroidFragmentApplication;
import com.mygdx.game.MyGdxGame;

/**
 * Created by Zhenyu on 2018-01-29.
 */

public class GameFragment extends AndroidFragmentApplication {

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState)
    {
        // return the GLSurfaceView on which libgdx is drawing game stuff
        return initializeForView(new MyGdxGame());
    }
}

