import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Nature vs City Classification")
st.write("This is a model that is used to predict whether the given image belongs to city or nature using CNN")
furniture_names = ['city', 'nature']

def classify_images(image):
    input_image = image.resize((180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = np.expand_dims(input_image_array, axis=0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = 'The Image belongs to ' + furniture_names[np.argmax(result)] + ' with a score of ' + str(np.max(result) * 100)
    return outcome

model = tf.keras.models.load_model("C:\\Users\\PUKAR\\Desktop\\pukar\\pukar\\city_classification.keras")

uploadfile = st.file_uploader("Upload your files here", type=["jpg", "png", "jpeg", "avif"])

if uploadfile is not None:
    img = Image.open(uploadfile)
    st.image(img, caption="Image to predict", use_column_width=True)
    outcome = classify_images(img)
    st.write(outcome)