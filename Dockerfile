# Dockerfile optimized for AWS Lambda with Face Recognition
FROM public.ecr.aws/lambda/python:3.9


# Set environment variable to redirect DeepFace cache to writable /tmp
ENV DEEPFACE_HOME=/tmp/.deepface

# Install system dependencies
RUN yum update -y && yum install -y \
    mesa-libGL \
    glib2 \
    && yum clean all

# Copy requirements first for better caching
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt


# Preload models during container build (with reduced logging)
# Preload model en una capa separada para aprovechar cache
RUN python -c "import os; os.environ['DEEPFACE_HOME'] = '/tmp/.deepface'; from deepface import DeepFace; DeepFace.build_model('Facenet512')"

# Copy Python files
COPY *.py ${LAMBDA_TASK_ROOT}/

# Handle .env file if it exists
COPY .env ${LAMBDA_TASK_ROOT}/

# Set the handler
CMD ["app.handler"] 