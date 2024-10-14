# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /usr/local/fase_2

COPY requirenments.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirenments.txt

# Copy the current directory contents into the container at /usr/src/app
COPY . .

CMD ["python", "./train.py"]
CMD ["python", "./predict.py"]
