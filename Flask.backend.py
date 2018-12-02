##################### Flask backend
######## Libraries:

#render_template: allows to take a html file and call that
from flask import Flask, request, render_template
#From the image that the user draws modify using Scintific Python
from scipy.misc import imread, imsave, imresize
import numpy as np
import keras.models
import re
import sys
import os

##### Absolute path to model folder:
sys.path.append(os.path.abspath("./model"))

##### Load the model
from load import *

##### Initialize the Flask app:
app = Flask(__name__)

global graph, model
model, graph = init()

#Decoding an image from base64 into raw representation:
def convertImage(imgData1):
    imgstr = re.search(r'base64,(.*)', imgData1).group(1)
    with open('output.png', 'wb') as output:
        output.write(imgstr.decode('base64'))

@app.route('/')
def index():
    return render_template('index.html') #Use our html file

@app.route('/predict', methods= ['GET', 'POST'])
def predict():
    imgData = request.get_data() #obtain the raw data


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
