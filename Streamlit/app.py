import streamlit as st
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import TFSMLayer
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from skimage.segmentation import mark_boundaries
import cv2
import lime
from lime import lime_image

os.makedirs("audio_files", exist_ok=True)
st.set_page_config(page_title="Deepfake Audio Detection", page_icon="")

class_names = ['real', 'fake']

# --- File handling ---
def save_file(sound_file):
    path = os.path.join('audio_files/', sound_file.name)
    with open(path, 'wb') as f:
        f.write(sound_file.getbuffer())
    return sound_file.name

# --- Spectrogram ---
def create_spectrogram(sound):
    audio_file = os.path.join('audio_files/', sound)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    y, sr = librosa.load(audio_file)
    ms = librosa.feature.melspectrogram(y=y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(log_ms, sr=sr)
    plt.savefig('melspectrogram.png')
    plt.close(fig)
    image_data = load_img('melspectrogram.png', target_size=(224, 224))
    st.image(image_data)
    return image_data

# --- Preprocessing ---
def preprocess_image(image_data):
    img_array = img_to_array(image_data)
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension
    img_array = img_array / 255.0  # normalize
    return tf.convert_to_tensor(img_array, dtype=tf.float32)

# --- Load model for inference only ---
def load_inference_model(saved_model_path='saved_model/model'):
    """
    Load SavedModel as inference-only using TFSMLayer (Keras 3 compatible).
    """
    layer = TFSMLayer(saved_model_path, call_endpoint='serving_default', trainable=False)
    model = Sequential([layer])
    return model

# --- Predictions ---
def predictions(image_data, model):
    x = preprocess_image(image_data)
    prediction = model.predict(x)
    class_label = np.argmax(prediction)
    return class_label, prediction

# --- LIME explanation ---
def lime_predict(image_data, model):
    x = preprocess_image(image_data)
    explainer = lime_image.LimeImageExplainer()
    img_array = np.array(image_data)
    explanation = explainer.explain_instance(
        img_array.astype('float64'),
        lambda imgs: model.predict(np.array(imgs)/255.0),
        hide_color=0, num_samples=1000
    )
    class_label = np.argmax(model.predict(x))
    fig, axs = plt.subplots(1, 2, figsize=(10, 25))
    temp, mask = explanation.get_image_and_mask(class_label, positive_only=False, num_features=8, hide_rest=True)
    axs[0].imshow(image_data)
    axs[1].imshow(mark_boundaries(temp, mask))
    axs[1].set_title(f"Predicted class: {class_names[class_label]}")
    plt.tight_layout()
    st.pyplot(fig)
    return fig

# --- Grad-CAM explanation ---
def grad_predict(image_data, model, class_idx):
    img_array = img_to_array(image_data)
    x = np.expand_dims(img_array, axis=0) / 255.0
    x = tf.convert_to_tensor(x, dtype=tf.float32)

    # Example using VGG16 (replace with your own if needed)
    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=True)
    last_conv_layer = base_model.get_layer('block5_conv3')
    grad_model = tf.keras.models.Model([base_model.inputs], [last_conv_layer.output, base_model.output])

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(x)
        class_output = preds[:, class_idx]
    grads = tape.gradient(class_output, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = cv2.resize(np.float32(heatmap), (x.shape[2], x.shape[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(x[0], 0.6, heatmap, 0.4, 0, dtype=cv2.CV_32F)

    fig1, ax = plt.subplots(1, 2, figsize=(10, 25))
    ax[0].imshow(image_data)
    ax[1].imshow(superimposed_img)
    ax[1].set_title(f"Predicted class: {class_names[class_idx]}")
    plt.tight_layout()
    st.pyplot(fig1)
    return superimposed_img

# --- Main ---
def main():
    page = st.sidebar.selectbox("App Selections", ["Homepage", "About"])
    if page == "Homepage":
        st.title("Deepfake Audio Detection using XAI")
        homepage()
    else:
        about()

def about():
    st.title("About this work")
    st.markdown("**Deepfake audio refers to synthetically created audio...**")

def homepage():
    st.write('___')
    st.subheader("Choose a wav file")
    uploaded_file = st.file_uploader(' ', type='wav')
    if uploaded_file is not None:
        st.write('### Play audio')
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes, format='audio/wav')

        save_file(uploaded_file)
        sound = uploaded_file.name
        with st.spinner('Fetching Results...'):
            spec = create_spectrogram(sound)
            # Load model (inference-only)
            model_path = os.path.join(os.path.dirname(__file__), "saved_model", "model")
            model = load_inference_model(model_path)

        st.write('### Classification results:')
        class_label, prediction = predictions(spec, model)
        st.write("#### The uploaded audio file is " + class_names[class_label])

        if st.button('Show XAI Metrics'):
            st.write('### XAI Metrics using LIME')
            with st.spinner('Fetching Results...'):
                lime_predict(spec, model)
            st.write('### XAI Metrics using Grad CAM')
            with st.spinner('Fetching Results...'):
                grad_predict(spec, model, class_label)
    else:
        st.info("Please upload a .wav file")

if __name__ == "__main__":
    main()
