# Sử dụng Python slim để nhẹ hơn (TensorFlow khá nặng)
FROM python:3.11-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Copy file requirements trước để tận dụng cache của Docker
COPY requirements.txt .

# Cài đặt các thư viện cần thiết
# --no-cache-dir giúp giảm dung lượng image
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code vào container (trừ những gì đã liệt kê trong .dockerignore)
COPY . .

# Mở port 5000 (mặc định của Flask)
EXPOSE 5000

# Lệnh chạy app
CMD ["python", "app.py"]