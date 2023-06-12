# Use the official Python image as the base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to the working directory
COPY requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files to the working directory
COPY . .

# Set environment variables
ENV PINECONE_API_KEY="352dabcd-86a6-4a12-9c2c-c1cf4f95d7c9"
ENV PINECONE_ENVIRONMENT_REGION="us-west4-gcp-free"
ENV PINECONE_INDEX_NAME="poc"

# Expose the port on which the Flask app will run
EXPOSE 8051

# Start the Flask app
CMD ["streamlit", "run", "chat_ui.py"]
