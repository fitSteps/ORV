# Use an official Python runtime as a parent image
FROM python:3.10.7

# Set the working directory in the container
WORKDIR /usr/src/orv

# Copy the current directory contents into the container at /usr/src/orv
COPY . .

# Install any needed packages specified in requirements.txt
# Assuming you have a requirements.txt; if not, create one or list packages directly
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 3000

# Define environment variable
ENV NAME World

# Create the directories where the shared volume will be mounted
# Ensure these directories match those specified in your docker run command
RUN mkdir -p /app/photos
RUN mkdir -p /app/videos

# Set the CMD to your startup script
# Ensure this is the correct path to your Python script that starts your app
CMD ["python", "Skripte/startup.py"]