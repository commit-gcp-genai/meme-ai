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
ARG PINECONE_API_KEY
ARG PINECONE_ENVIRONMENT_REGION
ARG PINECONE_INDEX_NAME
ENV PINECONE_API_KEY=$PINECONE_API_KEY
ENV PINECONE_ENVIRONMENT_REGION=$PINECONE_ENVIRONMENT_REGION
ENV PINECONE_INDEX_NAME=$PINECONE_INDEX_NAME

# Expose the port on which the Flask app will run
EXPOSE 8501

# Start the Flask app
CMD ["streamlit", "run", "chat_ui.py"]
