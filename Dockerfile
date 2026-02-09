FROM python:3.10-slim-buster

WORKDIR /app

COPY . /app

# 1. Install CPU Torch explicitly FIRST (saves ~3GB)
# We use --no-cache-dir to keep the image small
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. Install the rest of the dependencies
# sentence-transformers will see torch is already there and skip downloading the GPU version
RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python3", "app.py" ]