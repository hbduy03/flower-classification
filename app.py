import io
import numpy as np
from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, create_access_token, create_refresh_token, jwt_required, get_jwt_identity
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = 'abcd15151qcfaf61esaafsfwg123'
jwt = JWTManager(app)

USER={
    'admin':'123456'
}

model = load_model('flower.h5')
flower = ['daisy','dandelion','rose','sunflower','tulip']

@app.route('/')
def home():
    return jsonify({
        'message':'This is Flower Classification App',
        'status':'completed'
    })

@app.route('/login', methods=['POST'])
def login():
    username = request.json.get('username',None)
    password = request.json.get('password',None)
    if username not in USER or USER[username] != password:
        return jsonify({
            'message':'Wrong username or password'
        }), 401

    access_token = create_access_token(identity=username)
    refresh_token = create_refresh_token(identity=username)
    return jsonify({
        "access_token": access_token,
        "refresh_token": refresh_token
    })

@app.route('/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh():
    current_user = get_jwt_identity()
    new_access_token = create_access_token(identity=current_user)
    return jsonify({"access_token": new_access_token})

@app.route('/predict', methods=['POST'])
@jwt_required()
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