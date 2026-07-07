# 1. Choose a lightweight base image
FROM python:3.13-slim

# 2. Set the working directory
WORKDIR /app

# 3. Docker Layer Caching: Install dependencies first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of the code and model
COPY . .

# 5. Expose port (Documentation purposes)
EXPOSE 8000

# 6. UNIVERSAL STARTUP COMMAND
# We use 'sh -c' to evaluate environment variables.
# It checks if the $PORT environment variable exists (Render/AWS).
# If it does, it uses it. If not, it defaults to 8000 (Local Docker).
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]