import os
import io
import numpy as np
from PIL import Image
import base64

import tensorflow as tf

import keras
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras import backend as K

from flask import Flask, request, redirect, jsonify, render_template

app = Flask(__name__)
model = None
graph = None

def load_model():
    global model
    global graph
    model = keras.models.load_model("mnist_cnn_trained.h5")
    graph = K.get_session().graph

load_model()

def prepare_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image_size = (28, 28)
    image = image.resize(image_size)
    image = img_to_array(image)[:,:,0]
    image /= 255
    image = 1 - image
    return image.reshape((1, 28, 28, 1))


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    data = {"success": False}
    if request.method == 'POST':
        if request.form.get('digit'):
            # read the base64 encoded string
            image_string = request.form.get('digit')

            # Remove the header
            image_string = image_string.replace("data:image/png;base64,", "")

            # Decode the string
            image_data = base64.b64decode(image_string)

            # Open the image
            image = Image.open(io.BytesIO(image_data))

            # Preprocess the image and prepare it for classification
            image_preprocessed = prepare_image(image)

            # Get the tensorflow default graph
            global graph
            with graph.as_default():

                # Use the model to make a prediction
                predicted_digit = model.predict_classes(image_preprocessed)[0]
                data["prediction"] = str(predicted_digit)

                # indicate that the request was a success
                data["success"] = True

        return jsonify(data)
    return render_template("index.html")

if __name__ == "__main__":
    load_model()
    app.run()
