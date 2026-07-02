package com.limelight.ogl;

import android.content.Context;
import android.graphics.SurfaceTexture;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.opengl.GLUtils;
import android.util.Log;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;

/**
 * Renders a MediaCodec output SurfaceTexture into a visible TextureView using a side-by-side
 * phone-VR shader. This is mono SBS: both eyes receive the same Moonlight frame, with optional
 * lens-style warp and clipping.
 */
public class VideoTextureRenderer extends TextureSurfaceRenderer implements SurfaceTexture.OnFrameAvailableListener {
    private static final String LOG_TAG = "Moonlight.SBS.Video";

    private static final String VERTEX_SHADER =
            "attribute vec4 vPosition;" +
            "attribute vec4 vTexCoordinate;" +
            "uniform mat4 textureTransform;" +
            "varying vec2 v_TexCoordinate;" +
            "void main() {" +
            "  v_TexCoordinate = (textureTransform * vTexCoordinate).xy;" +
            "  gl_Position = vPosition;" +
            "}";

    private static final String FRAGMENT_SHADER =
            "#extension GL_OES_EGL_image_external : require\n" +
            "precision highp float;" +
            "uniform samplerExternalOES texture;" +
            "uniform float zoomFactor;" +
            "uniform float distFactor;" +
            "uniform float wrapEnabled;" +
            "uniform float singleView;" +
            "varying vec2 v_TexCoordinate;" +
            "vec2 Warp(vec2 tex) {" +
            "  vec2 newPos = tex;" +
            "  float c = -distFactor / 10.0;" +
            "  float zoomU = zoomFactor * 0.75;" +
            "  float u = tex.x * zoomU - (zoomU / 2.0);" +
            "  float v = tex.y * zoomFactor - (zoomFactor / 2.0);" +
            "  newPos.x = c * u / (pow(v, 2.0) + c);" +
            "  newPos.y = c * v / (pow(u, 2.0) + c);" +
            "  newPos.x = (newPos.x + 1.0) * 0.5;" +
            "  newPos.y = (newPos.y + 1.0) * 0.5;" +
            "  return newPos;" +
            "}" +
            "void main() {" +
            "  if (singleView < 0.5) {" +
            "    vec2 newPos = v_TexCoordinate;" +
            "    if (newPos.x < 0.5) {" +
            "      newPos.x = newPos.x * 2.0;" +
            "    } else {" +
            "      newPos.x = (newPos.x - 0.5) * 2.0;" +
            "    }" +
            "    newPos = Warp(newPos);" +
            "    vec4 color = texture2D(texture, newPos);" +
            "    if (wrapEnabled < 0.5) {" +
            "      vec2 borderStep = step(0.0, newPos) * step(newPos, vec2(1.0, 1.0));" +
            "      color *= borderStep.x * borderStep.y;" +
            "    }" +
            "    gl_FragColor = color;" +
            "  } else {" +
            "    float squeezeFactor = distFactor / 100.0;" +
            "    gl_FragColor = texture2D(texture, vec2(v_TexCoordinate.x, v_TexCoordinate.y * squeezeFactor - (squeezeFactor * 0.5)));" +
            "  }" +
            "}";

    private static final float[] SQUARE_COORDS = {
            -1.0f,  1.0f, 0.0f,
            -1.0f, -1.0f, 0.0f,
             1.0f, -1.0f, 0.0f,
             1.0f,  1.0f, 0.0f
    };

    private static final float[] TEXTURE_COORDS = {
            0.0f, 1.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 0.0f, 1.0f,
            1.0f, 0.0f, 0.0f, 1.0f,
            1.0f, 1.0f, 0.0f, 1.0f
    };

    private static final short[] DRAW_ORDER = { 0, 1, 2, 0, 2, 3 };

    private final Context context;
    private final int[] textures = new int[1];
    private final float[] videoTextureTransform = new float[16];

    private FloatBuffer textureBuffer;
    private FloatBuffer vertexBuffer;
    private ShortBuffer drawListBuffer;

    private int vertexShaderHandle;
    private int fragmentShaderHandle;
    private int shaderProgram;

    private float zoomFactor = 3.2f;
    private float distortionFactor = 81.0f;
    private float wrapEnabled = 1.0f;
    private float singleView = 0.0f;
    private boolean zoomedIn = false;

    private SurfaceTexture videoTexture;
    private boolean frameAvailable = false;
    private int videoWidth;
    private int videoHeight;
    private boolean adjustViewport = false;

    public VideoTextureRenderer(Context context, SurfaceTexture outputTexture, int width, int height, OnGlReadyListener listener) {
        super(outputTexture, width, height, listener);
        this.context = context;
    }

    public boolean isZoomedIn() {
        return zoomedIn;
    }

    public void setZoomedIn(boolean zoomedIn) {
        this.zoomedIn = zoomedIn;
    }

    public void setZoomFactor(float zoomFactor) {
        this.zoomFactor = zoomFactor / 15.625f;
    }

    public void setDistortionFactor(float distortionFactor) {
        this.distortionFactor = distortionFactor;
    }

    public void setWrapEnabled(boolean enabled) {
        this.wrapEnabled = enabled ? 1.0f : 0.0f;
    }

    public void setSingleView(boolean enabled) {
        this.singleView = enabled ? 1.0f : 0.0f;
    }

    public SurfaceTexture getVideoTexture() {
        return videoTexture;
    }

    public void setVideoSize(int width, int height) {
        this.videoWidth = width;
        this.videoHeight = height;
        if (videoTexture != null) {
            videoTexture.setDefaultBufferSize(width, height);
        }
        adjustViewport = true;
    }

    @Override
    protected void initGLComponents() {
        setupVertexBuffer();
        setupTexture();
        loadShaders();
    }

    @Override
    protected void deinitGLComponents() {
        GLES20.glDeleteTextures(1, textures, 0);
        GLES20.glDeleteProgram(shaderProgram);
        if (videoTexture != null) {
            videoTexture.setOnFrameAvailableListener(null);
            videoTexture.release();
        }
    }

    @Override
    protected boolean draw() {
        synchronized (this) {
            if (frameAvailable) {
                videoTexture.updateTexImage();
                videoTexture.getTransformMatrix(videoTextureTransform);
                frameAvailable = false;
            } else {
                return false;
            }
        }

        if (adjustViewport) {
            adjustViewport();
        }

        GLES20.glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);

        GLES20.glUseProgram(shaderProgram);

        int textureParamHandle = GLES20.glGetUniformLocation(shaderProgram, "texture");
        int textureCoordinateHandle = GLES20.glGetAttribLocation(shaderProgram, "vTexCoordinate");
        int positionHandle = GLES20.glGetAttribLocation(shaderProgram, "vPosition");
        int textureTransformHandle = GLES20.glGetUniformLocation(shaderProgram, "textureTransform");
        int zoomHandle = GLES20.glGetUniformLocation(shaderProgram, "zoomFactor");
        int distHandle = GLES20.glGetUniformLocation(shaderProgram, "distFactor");
        int wrapHandle = GLES20.glGetUniformLocation(shaderProgram, "wrapEnabled");
        int singleHandle = GLES20.glGetUniformLocation(shaderProgram, "singleView");

        GLES20.glEnableVertexAttribArray(positionHandle);
        GLES20.glVertexAttribPointer(positionHandle, 3, GLES20.GL_FLOAT, false, 4 * 3, vertexBuffer);

        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textures[0]);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glUniform1i(textureParamHandle, 0);

        GLES20.glEnableVertexAttribArray(textureCoordinateHandle);
        GLES20.glVertexAttribPointer(textureCoordinateHandle, 4, GLES20.GL_FLOAT, false, 0, textureBuffer);

        GLES20.glUniformMatrix4fv(textureTransformHandle, 1, false, videoTextureTransform, 0);
        GLES20.glUniform1f(zoomHandle, zoomedIn ? zoomFactor * 1.8f : zoomFactor);
        GLES20.glUniform1f(distHandle, distortionFactor);
        GLES20.glUniform1f(wrapHandle, wrapEnabled);
        GLES20.glUniform1f(singleHandle, singleView);

        GLES20.glDrawElements(GLES20.GL_TRIANGLES, DRAW_ORDER.length, GLES20.GL_UNSIGNED_SHORT, drawListBuffer);
        GLES20.glDisableVertexAttribArray(positionHandle);
        GLES20.glDisableVertexAttribArray(textureCoordinateHandle);
        return true;
    }

    private void setupVertexBuffer() {
        ByteBuffer drawBuffer = ByteBuffer.allocateDirect(DRAW_ORDER.length * 2).order(ByteOrder.nativeOrder());
        drawListBuffer = drawBuffer.asShortBuffer();
        drawListBuffer.put(DRAW_ORDER);
        drawListBuffer.position(0);

        ByteBuffer vertexByteBuffer = ByteBuffer.allocateDirect(SQUARE_COORDS.length * 4).order(ByteOrder.nativeOrder());
        vertexBuffer = vertexByteBuffer.asFloatBuffer();
        vertexBuffer.put(SQUARE_COORDS);
        vertexBuffer.position(0);

        ByteBuffer textureByteBuffer = ByteBuffer.allocateDirect(TEXTURE_COORDS.length * 4).order(ByteOrder.nativeOrder());
        textureBuffer = textureByteBuffer.asFloatBuffer();
        textureBuffer.put(TEXTURE_COORDS);
        textureBuffer.position(0);
    }

    private void setupTexture() {
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glGenTextures(1, textures, 0);
        checkGlError("texture generate");

        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textures[0]);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
        checkGlError("texture bind");

        videoTexture = new SurfaceTexture(textures[0]);
        if (videoWidth > 0 && videoHeight > 0) {
            videoTexture.setDefaultBufferSize(videoWidth, videoHeight);
        }
        videoTexture.setOnFrameAvailableListener(this);
    }

    private void loadShaders() {
        vertexShaderHandle = compileShader(GLES20.GL_VERTEX_SHADER, VERTEX_SHADER, "vertex");
        fragmentShaderHandle = compileShader(GLES20.GL_FRAGMENT_SHADER, FRAGMENT_SHADER, "fragment");

        shaderProgram = GLES20.glCreateProgram();
        GLES20.glAttachShader(shaderProgram, vertexShaderHandle);
        GLES20.glAttachShader(shaderProgram, fragmentShaderHandle);
        GLES20.glLinkProgram(shaderProgram);
        checkGlError("program link");

        int[] status = new int[1];
        GLES20.glGetProgramiv(shaderProgram, GLES20.GL_LINK_STATUS, status, 0);
        if (status[0] != GLES20.GL_TRUE) {
            Log.e(LOG_TAG, "Shader link error:\n" + GLES20.glGetProgramInfoLog(shaderProgram));
        }
    }

    private int compileShader(int type, String source, String name) {
        int shader = GLES20.glCreateShader(type);
        GLES20.glShaderSource(shader, source);
        GLES20.glCompileShader(shader);
        checkGlError(name + " shader compile");

        int[] status = new int[1];
        GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, status, 0);
        if (status[0] != GLES20.GL_TRUE) {
            Log.e(LOG_TAG, name + " shader error:\n" + GLES20.glGetShaderInfoLog(shader));
        }
        return shader;
    }

    private void adjustViewport() {
        if (videoWidth <= 0 || videoHeight <= 0) {
            GLES20.glViewport(0, 0, width, height);
        } else {
            float surfaceAspect = height / (float) width;
            float videoAspect = videoHeight / (float) videoWidth;

            if (surfaceAspect > videoAspect) {
                float heightRatio = height / (float) videoHeight;
                int newWidth = (int) (videoWidth * heightRatio);
                int xOffset = (newWidth - width) / 2;
                GLES20.glViewport(-xOffset, 0, newWidth, height);
            } else {
                float widthRatio = width / (float) videoWidth;
                int newHeight = (int) (videoHeight * widthRatio);
                int yOffset = (newHeight - height) / 2;
                GLES20.glViewport(0, -yOffset, width, newHeight);
            }
        }
        adjustViewport = false;
    }

    public void checkGlError(String op) {
        int error;
        while ((error = GLES20.glGetError()) != GLES20.GL_NO_ERROR) {
            Log.e(LOG_TAG, op + ": glError " + GLUtils.getEGLErrorString(error));
        }
    }

    @Override
    public void onFrameAvailable(SurfaceTexture surfaceTexture) {
        synchronized (this) {
            frameAvailable = true;
        }
    }
}
