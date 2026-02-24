import io
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

app = Flask(__name__)
model = load_model('flower.h5')
flower = ['daisy','dandelion','rose','sunflower','tulip']

@app.route('/')
def home():
    return jsonify({
        'message':'This is Flower Classification App',
        'status':'completed'
    })

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = Image.open(file.stream)
    img = img.resize((150,150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prob = model.predict(img_array)[0]
    label_index = np.argmax(prob)
    label_name = flower[label_index]
    confidence = round(float(prob[label_index]), 4) * 100
    return jsonify({
        'prediction': label_name ,
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)