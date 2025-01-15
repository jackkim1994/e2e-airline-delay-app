# Use an official Python runtime as a parent image
FROM --platform=linux/x86_64 python:3.9-slim

# Set the working directory in Docker
WORKDIR /app

# Install and upgrade pip
RUN /usr/local/bin/python -m pip install --upgrade pip

# Copy the requirements file from the parent directory into the current directory
COPY requirements.txt .

# Copy other necessary files or directories from the parent directory
COPY src/ ./src/
COPY config/ ./config/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 80

# Run pipeline.py when the container launches
CMD ["streamlit", "run", "--server.port=80", "src/app.py"]