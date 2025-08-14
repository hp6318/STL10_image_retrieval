# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files into the container
COPY . .

# Expose the port that Streamlit runs on
EXPOSE 8501

# Define the command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]