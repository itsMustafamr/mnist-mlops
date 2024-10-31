# Dockerfile

FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code from src/
COPY src/ .

# Expose port 8000 for the FastAPI app
EXPOSE 8000

# Command to run the FastAPI server
CMD ["uvicorn", "deploy:app", "--host", "0.0.0.0", "--port", "8000"]
