# 1. Choose a lightweight base image (Python 3.13 on Debian Slim)
# "slim" reduces the image size from ~1GB to ~150MB, making deployments much faster.
FROM python:3.13-slim

# 2. Set the working directory inside the container to /app
WORKDIR /app

# 3. SYSTEMS ENGINEERING FLEX: Docker Layer Caching
# We copy ONLY the requirements.txt first and install dependencies.
# Docker caches this layer. If you change your code later, Docker won't 
# reinstall the dependencies; it will just use the cached layer!
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Now, copy the rest of our application code and the model
COPY . .

# 5. Expose the port the API runs on (Documentation purposes)
EXPOSE 8000

# 6. The command to run when the container starts
# CRITICAL: We must use --host 0.0.0.0. If we use 127.0.0.1, the API 
# will only be accessible from *inside* the container, not from your browser!
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]