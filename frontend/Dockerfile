#frontend/Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN apt-get update && apt-get install -y ffmpeg
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
