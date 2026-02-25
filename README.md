# 🌸 Flower Classification
A deep learning project aimed at classifying 5 types of flowers from images using CNN, RESTful APIs and Flask.

## Tech Stack
- Backend: Python, Flask
- Machine Learning: TensorFlow, Keras, NumPy, Pillow
- Deployment: Docker

## Dataset
- Source: https://www.kaggle.com/datasets/alxmamaev/flowers-recognition?resource=download

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
1. Place the dataset into project's folder
2. run the training script to create CNN model: py main.py
3. Start the server: py app.py
4. Test the API:
- You can test the prediction by using Postman or curl.
- The server will respond with a JSON object.

## Result:

<img width="928" height="117" alt="image" src="https://github.com/user-attachments/assets/9f746e3e-4143-4725-9469-a9e4f470fb72" />

<img width="872" height="102" alt="image" src="https://github.com/user-attachments/assets/47ff2d29-76ae-46ac-b138-49adf0ec966c" />

