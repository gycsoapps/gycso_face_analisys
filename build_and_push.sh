#!/bin/bash
# Script to build and push the Docker image to ECR for AWS Lambda (ARM64 Architecture)

echo "Building Docker image for ARM64 architecture..."
docker build --platform linux/arm64 -t face_recognition_python:latest .

# Check if the build was successful
if [ $? -ne 0 ]; then
  echo "Docker build failed!"
  exit 1
fi

echo "Tagging image for ECR..."
docker tag face_recognition_python:latest 292840721816.dkr.ecr.mx-central-1.amazonaws.com/face_recognition_python:latest

echo "Logging in to ECR..."
aws ecr get-login-password --region mx-central-1 | docker login --username AWS --password-stdin 292840721816.dkr.ecr.mx-central-1.amazonaws.com

# Check if login was successful
if [ $? -ne 0 ]; then
  echo "ECR login failed! Check your AWS credentials."
  exit 1
fi

echo "Pushing image to ECR..."
docker push 292840721816.dkr.ecr.mx-central-1.amazonaws.com/face_recognition_python:latest

# Check if push was successful
if [ $? -ne 0 ]; then
  echo "Image push failed!"
  exit 1
fi

echo "Process completed successfully!"
echo "Image URI: 292840721816.dkr.ecr.mx-central-1.amazonaws.com/face_recognition_python:latest"
echo ""
echo "=== IMPORTANT LAMBDA CONFIGURATION ==="
echo "1. Choose ARM64 architecture when configuring Lambda"
echo "2. Set timeout to at least 60-120 seconds"
echo "3. Allocate at least 2048 MB of memory (4096 MB recommended)"
echo "4. Set environment variables:"
echo "   - AWS_REGION=mx-central-1"
echo "   - DEFAULT_MODEL=Facenet512"
echo "   - DEFAULT_DETECTOR=opencv"
echo "   - LOG_LEVEL=INFO"
echo "   - (and any other required variables such as S3_BUCKET, etc.)" 