import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input,decode_predictions
from tensorflow.keras.models import Model,load_model,model_from_json
from flask import Flask, redirect, url_for, request, render_template,jsonify
# from tensorflow.python.keras.backend import set_session
from flask_cors import CORS, cross_origin
import requests
from PIL import Image
import numpy as np
import pickle

# Design a flask app
app=Flask(__name__)
CORS(app)

import pandas as pd
images=pd.read_csv('./images.csv')['link'].to_numpy()
test_images=pd.read_csv('./images.csv')[:5]
test_images=test_images['link'].to_list()


config = tf.ConfigProto(
    device_count={'GPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)
session = tf.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

with open('./models/resnet50.json','r') as f:
    model_json=f.read()

knn_model = pickle.load(open('./models/knn.pkl', 'rb'))    
graph = tf.compat.v1.get_default_graph()    
model=model_from_json(model_json) 
model.load_weights('./models/resnet50.h5')
model._make_predict_function()


def predict_model(url):
    
    global knn_model  
    
    with session.as_default():
            with session.graph.as_default():
                im = Image.open(requests.get(url, stream=True).raw).resize((224,224))
                x = image.img_to_array(im)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                feature = model.predict(x).reshape(1,-1)
    distances, indices = knn_model.kneighbors(feature)
    return distances[0],indices[0]

    
        
@app.route('/', methods=['GET','POST'])
def index():
    
    # Main page
    if request.method=='POST':
        
        url=request.get_json()
        distances,indices=predict_model(url)
        data={'distance':distances.tolist(),'images':images[indices].tolist()}
        return jsonify(data)

    dict={'images':test_images,'name':'Visual Search'}
    return render_template('index.html',result=dict)

if __name__ =='__main__':
    
    app.run(debug=True)
    