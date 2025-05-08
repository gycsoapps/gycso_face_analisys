# Dockerfile optimized for AWS Lambda with Face Recognition
FROM public.ecr.aws/lambda/python:3.9-arm64

# Set environment variable to redirect DeepFace cache to writable /tmp
ENV DEEPFACE_HOME=/tmp

# Install system dependencies
RUN yum update -y && yum install -y \
    mesa-libGL \
    glib2 \
    && yum clean all

# Copy requirements first for better caching
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Preload models during container build
RUN python -c "import os; \
    os.environ['DEEPFACE_HOME'] = '/tmp'; \
    from deepface import DeepFace; \
    print('Preloading Facenet512 model...'); \
    DeepFace.build_model('Facenet512'); \
    print('Preloading SSD detector...'); \
    from deepface.detectors import Ssd; \
    detector = Ssd.SsdClient(); \
    _ = detector.build_model(); \
    print('Models loaded successfully!')"

# Copy Python files
COPY *.py ${LAMBDA_TASK_ROOT}/

# Handle .env file if it exists
COPY .env ${LAMBDA_TASK_ROOT}/

# Set the handler
CMD ["app.lambda_handler"] 