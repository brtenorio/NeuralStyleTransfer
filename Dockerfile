# Use the official Python image from the Docker Hub
FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Update pip
RUN pip install --upgrade pip

# Install HDF5 and its dependencies using apt
RUN apt-get update && apt-get install -y pkg-config gcc libhdf5-dev

COPY . /app

# Install the app dependencies 
RUN python3 -m pip install -r requirements.txt

# Expose the port that the app runs on
EXPOSE 8080

# Command to run the application
CMD ["streamlit", "run", "neural_style_transfer/app.py", "--server.port", "8080"]
