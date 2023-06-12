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
ENV CLOUD_STORAGE_BUCKET="commit-ai-apis-demo"
ENV IMAGE_GENERATION_PROJECT="860550299524"
ENV IMAGE_GENERATION_ENDPOINT="8743606735144484864"
ENV IMAGE_GENERATION_LOCATION="europe-west1"

# Expose the port on which the Flask app will run
EXPOSE 8080

# Start the Flask app
CMD ["python", "app.py"]
