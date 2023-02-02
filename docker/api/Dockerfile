# Use an official Python runtime as the base image
FROM python:3.10

# Set the working directory to /app
WORKDIR /app

# # Copy the current directory contents into the container at /app
# COPY . /app

# Copy the requirements.txt file to the container
COPY requirements.txt /app

# Install dependencies
RUN pip install -r requirements.txt

# Copy the main.py file to the container
COPY main.py /app

# # Set the environment variable for the FastAPI app
# ENV FASTAPI_APP=main:api

# Specify the command to run when the container starts
CMD uvicorn main:api --host 0.0.0.0
# CMD ["uvicorn", "main:api", "--workers", "4", "--host", "0.0.0.0", "--port", "8000"]
