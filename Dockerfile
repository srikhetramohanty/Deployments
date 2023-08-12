# Use the official Python image as the base image
FROM python:3.8-slim-buster

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install required system packages for graphics support
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0

# Install the required dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY ./ /app

# Set the working directory inside the container
WORKDIR /app


# Expose port 5000
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]
