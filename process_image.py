from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.models import Model,load_model,model_from_json
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import pandas as pd
import numpy as np
import requests
import multiprocessing
import time
import pdb
import pickle


with open('./models/resnet50.json','r') as f:
    model_json=f.read()
model=model_from_json(model_json) 
model.load_weights('./models/resnet50.h5')

images=pd.read_csv('./images.csv')
images=images['link'].to_numpy()


def process_image(url):

    global features
    try:
        im = Image.open(requests.get(url, stream=True).raw).resize((224,224))
        x = image.img_to_array(im)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features.append(model.predict(x).reshape(-1))
    
    except :
        pass

if __name__ == '__main__':    
    
    features=[]
    ts = time.time()   
    print('Starting multiprocessing images..')
    pool=multiprocessing.Pool(processes = multiprocessing.cpu_count())    
    for idx,url in enumerate(images.tolist()):
        process_image(url)
        if(idx % 1000==0):
            print('Currently running..{}'.format(idx))
    pickle_out = open("./models/features.pkl","wb")
    pickle.dump(features, pickle_out)
    pickle_out.close()            
    print('Time for getting features:', time.time() - ts)
    print('fitting neighbours...')
    try:
        nei_clf =  NearestNeighbors(algorithm='auto').fit(np.array(features))
        knnPickle = open('./models/knn.pkl', 'wb') 
        pickle.dump(nei_clf, knnPickle)
        print('Successfully pickled all models..!')
    except Exception as e:
        print(e)