# Dockerfile
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install Python dependencies listed in the requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the source code from src/ to the container
COPY src/ .

# Expose port 8000 for the FastAPI app to be accessed externally
EXPOSE 8000

# Command to run the FastAPI server using Uvicorn
CMD ["uvicorn", "deploy:app", "--host", "0.0.0.0", "--port", "8000"]
