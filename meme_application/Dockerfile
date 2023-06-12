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
ENV 

# Expose the port on which the Flask app will run
EXPOSE 8080

# Start the Flask app
CMD ["python", "app.py"]
