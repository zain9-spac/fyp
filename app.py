import streamlit as st
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator     
import matplotlib.pyplot as plt
#File Processing Pkgs
from PIL import Image,ImageOps


st.title("Deep Learning Based Plant Diseases Classification")

@st.cache(allow_output_mutation=True)

def load_model():
  model=tf.keras.models.load_model('/content/model_inception.h5')
  return model

model =  load_model()

def load_img(image_file):
  img = Image.open(image_file)
  return img

#model = tf.keras.modells.load_model('model_inception.h5')
uploaded_file =  st.file_uploader("Choose a image file",type="jpg")

def import_and_predict(image_data,model):
  size = (150,150)
  image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
  x = np.asarray(image)
  x = np.expand_dims(x, axis=0)
  x = x/255
  images = np.vstack([x])
  classes = model.predict(images, batch_size=10)
  return classes

if uploaded_file is not None:
  #st.write(type(uploaded_file))
  st.image(load_img(uploaded_file))
  image = Image.open(uploaded_file)
  Genrate_pred = st.button("Generate Prediction")
  if Genrate_pred:
    classes = import_and_predict(image,model)
    if (classes[0][0] > 0.6):
      st.title("Model Prediction: Late Blight")
    else:
      st.title("Model Prediction: Early Blight")