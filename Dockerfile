FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependency list and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy files into the container
COPY . .

# Expose container-side ports:
EXPOSE 7860 8000 9100

# Launch app
CMD ["python", "app.py"]