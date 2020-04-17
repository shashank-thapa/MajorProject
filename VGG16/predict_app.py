import base64
import numpy as np 
import io
import os
from PIL import Image
import tensorflow as tf
from flask import Flask,request, jsonify
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.optimizers import Adam

app= Flask(__name__)

def get_model():
    global model
    model=load_model('chestxray1.h5')
    print(' *Model Loaded! ')

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image=image.convert("RGB")
    image=image.resize(target_size)
    image=img_to_array(image)
    image=np.expand_dims(image,axis=0)

    return image

print(" * Loading Keras model... ")
get_model()
graph = tf.get_default_graph() 

@app.route('/predict.html',methods=['POST'])
def predict():
    message=request.get_json(force=True)
    encoded=message['image']
    decoded=base64.b64decode(encoded)
    image=Image.open(io.BytesIO(decoded))
    processed_image=preprocess_image(image,target_size=(224,224))
    global graph
    with graph.as_default():
        prediction=model.predict(processed_image).tolist()
    response= {
        'prediction': {
            'Normal': prediction[0][0],
            'Pneumonia': prediction[0][1]
        }
    }
    return jsonify(response)

if __name__=='__main__':
    port=int(os.environ.get('PORT',5000))
    app.run(host='0.0.0.0', port=port)

