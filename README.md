# 🌸 Flower Classification
A deep learning project aimed at classifying 5 types of flowers from images using CNN, RESTful APIs built by Flask secured by JWT authentication
## Tech Stack
- **Backend:** Python, Flask, Flask-JWT-Extended
- **Deep Learning:** TensorFlow, Keras, NumPy, Pillow
- **Deployment:** Docker

## Project Structure

<img width="319" height="421" alt="image" src="https://github.com/user-attachments/assets/0079b58b-28ed-4145-87eb-86ed837f0736" />

## 📊 Evaluation
- Loss: Decreased steadily, indicating effective learning.
- Accuracy: Improved consistently, reaching ~75% on validation.
- Precision & Recall: Both increased, showing better class-wise prediction quality.
<img width="1062" height="613" alt="Screenshot 2025-09-24 035502" src="https://github.com/user-attachments/assets/1b298ac2-0977-4d59-8271-73175cbacb93" />

## 🛠️ How to Use
- With Docker: docker run -p 5000:5000 hbduy03/flower-app:latest
- No Docker:
1. Place the dataset [Flowers Recognition](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition?resource=download) into project's folder
 2. run the training script to create CNN model: py main.py
 3. Start the server: py app.py
## Test the API: 
You can test the prediction by using Postman (Recommended) or curl:
1. Login to get Tokens
- **Method:** `POST`
- **URL:** http://localhost:5000/login
- **Input**:
  {
      "username": "admin",
      "password": "123456"
  }
2. Predict Flower Type
- **Method:** `POST`
- **URL:** http://localhost:5000/predict
- **Authorization:** Your available **access token** after first step
- **Input**: file (.jpg/.png)
 3. Refresh Expired Token
- **Method:** `POST`
- **URL:** http://localhost:5000/refresh
- **Authorization:** Your available **refresh token** after first step
## Result:
**1. Login API:**
<img width="999" height="203" alt="image" src="https://github.com/user-attachments/assets/2bbc906f-b140-4da7-aae2-ad14f9496773" />
**2. Predict API:**

<img width="331" height="140" alt="Screenshot 2026-06-19 190428" src="https://github.com/user-attachments/assets/021280ae-33b9-4960-8c5a-2ee5d954dce4" />
<img width="315" height="136" alt="Screenshot 2026-06-19 190844" src="https://github.com/user-attachments/assets/45efa4a1-6cf6-4ffe-984f-856ea4b309f3" />
<img width="328" height="138" alt="image" src="https://github.com/user-attachments/assets/bfae1ae4-e937-4e29-a7d6-dd72dc4bcf31" />


