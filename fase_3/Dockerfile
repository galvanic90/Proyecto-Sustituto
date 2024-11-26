# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /usr/local/fase_3

COPY requirenments.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirenments.txt

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# This line is the command use to raise the server
CMD ["uvicorn", "apirest:app", "--host", "0.0.0.0"]