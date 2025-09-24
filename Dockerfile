# Start with a Python base image.
FROM python:3.10-slim

# Set the working directory.
WORKDIR /app

# Copy all project files from the root into the container.
COPY . .

# Move into the backend directory for installing dependencies.
WORKDIR /app/backend

# Install the Python dependencies from requirements.txt.
RUN pip install --no-cache-dir -r requirements.txt

# Return to the root directory.
WORKDIR /app

# Define the command to run your application using Uvicorn.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]