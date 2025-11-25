#!/bin/bash
# This script is for debugging the recsys-api service.
# It builds and starts the service, and prints the logs.

echo "Building and starting the recsys-api service..."
docker compose up --build --no-deps recsys-api
