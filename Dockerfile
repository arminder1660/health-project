# CardiaNova - container image for the FastAPI app.
#
# Build:  docker build -t cardianova .
# Run:    docker run -p 8000:8000 --env-file .env cardianova
#
# The trained model at models/cardio_model.joblib is committed to the repo and is
# copied in with the source below, so the container starts with a ready artifact.
# Training stays out of the build: it adds minutes to every build and would produce
# a model that differs from the one CI and the tests were run against.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first so this layer is cached across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# The app has no database and stores no patient data, so a plain uvicorn
# process is the whole service.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
