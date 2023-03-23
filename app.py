import os

import numpy as np
from tensorflow.compat.v1 import ConfigProto, InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Flask utils
from flask import Flask, render_template, request
# Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='model_resnet152V2_2.2.0.h5'

# Load your trained model
model = load_model(MODEL_PATH)


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds == 0:
        preds = "The leaf is diseased cotton leaf"
    elif preds == 1:
        preds = "The leaf is diseased cotton plant"
    elif preds == 2:
        preds = "The leaf is fresh cotton leaf"
    else:
        preds = "The leaf is fresh cotton plant"
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        f.close()

        # Make prediction
        preds = model_predict(file_path, model)

        if os.path.exists(file_path):
            os.remove(file_path)

        return preds


if __name__ == '__main__':
    app.run(port=5001,debug=True)
