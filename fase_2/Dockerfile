# Use an official Python runtime as a parent image
FROM python:3.10-slim

CMD ["ls", "-la", "/fase_2"]

COPY . requirenments.txt /fase_2/

# Set the working directory in the container
WORKDIR /usr/local/fase_2

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip3 install --no-cache-dir -r requirenments.txt

CMD ["python", "./train.py"]
CMD ["python", "./predict.py"]
