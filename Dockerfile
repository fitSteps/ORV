# Use an official Python runtime as a parent image
FROM python:3.10.7

# Set the working directory in the container
WORKDIR /usr/src/orv

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
# Assuming you have a requirements.txt; if not, create one or list packages directly
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run ai_model.py when the container launches
CMD ["python", "Skripte/startup.py"]