
from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf


from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='best_model_vgg16.h5'

# Load your trained model
model = load_model(MODEL_PATH)




def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
   # x = preprocess_input(x)

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    print(preds)
    if preds==0:
        preds="The Disease type is Tomato_Bacterial_spot"
    elif preds==1:
        preds="The Disease type is Tomato_Early_blight'"
    elif preds==2:
        preds="The Disease type is Tomato_Late_blight'"
    elif preds==3:
        preds="Te Disease type is Tomato_Leaf_Mold'"
    elif preds==4:
        preds="The Disease type is Tomato_Septoria_leaf_spot'"
    elif preds==5:
        preds="The Disease type is Tomato_Spider_mites Two-spotted_spider_mite"
    elif preds==6:
        preds="The Disease type is Tomato_Target_spots'"
    elif preds==7:
        preds="The Disease type is Tomato_Yellow_Leaf_Curl_Virus"
    elif preds==8:
        preds="The Disease type is Tomato_mosaic_virus"
    elif preds==9:
        preds="The Plant is healthy"
    
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
