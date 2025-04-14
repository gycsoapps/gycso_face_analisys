# Face Recognition API

A modern face recognition API built with FastAPI and DeepFace that allows comparing faces using base64-encoded images and AWS S3 storage.

## Features

- Compare two base64-encoded images to determine if they contain the same person
- Compare a base64-encoded image with an image stored in AWS S3
- In-memory image processing (no temporary files stored on disk)
- Configurable face recognition models
- Interactive API documentation with Swagger UI

## Installation and Local Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/face-recognition-api.git
cd face-recognition-api
```

### Step 2: Create and Activate a Virtual Environment (Optional but Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including:
- deepface
- fastapi
- python-multipart
- uvicorn
- boto3
- pillow
- numpy
- opencv-python

### Step 4: Configure AWS S3 (Optional - Only if Using S3 Integration)

Set up environment variables for AWS S3:

```bash
# On Windows
set AWS_ACCESS_KEY_ID=your_access_key
set AWS_SECRET_ACCESS_KEY=your_secret_key
set AWS_REGION=your_region
set S3_BUCKET=your_bucket_name

# On macOS/Linux
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_REGION=your_region
export S3_BUCKET=your_bucket_name
```

### Step 5: Run the API Server

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at http://localhost:8000

## Using the API

### Interactive API Documentation

The API comes with built-in interactive documentation powered by Swagger UI. To access it:

1. Start the API server as described above
2. Open your browser and navigate to http://localhost:8000/docs

In the Swagger UI, you can:
- See all available endpoints
- Expand each endpoint to view request parameters
- Try out the API directly by filling in the request forms
- View response schemas and status codes

Alternatively, you can access the ReDoc documentation at http://localhost:8000/redoc for a different documentation layout.

### Available Endpoints

1. **POST /compare-base64**: Compare two base64-encoded images
2. **POST /compare-with-s3**: Compare a base64-encoded image with an image stored in S3

## Face Recognition Models

DeepFace supports multiple state-of-the-art face recognition models. You can specify which model to use by setting the `model_name` parameter in the DeepFace.verify function.

| Model | Description | Size | Accuracy | Speed | Notes |
|-------|-------------|------|----------|-------|-------|
| VGG-Face | Based on the VGG-16 architecture | 515 MB | 96.7% | Medium | Default model, good balance of accuracy and speed |
| Facenet | Google's FaceNet model | 91 MB | 97.4% | Fast | Good for mobile and edge devices |
| Facenet512 | Improved FaceNet with 512D embeddings | 92 MB | 98.4% | Fast | Higher accuracy than original FaceNet |
| OpenFace | Open source face recognition | 93 MB | 78.7% | Fast | Lightweight but less accurate |
| DeepFace | Facebook's DeepFace model | 95 MB | 69.0% | Medium | Historical importance but outperformed by newer models |
| DeepID | Deep hidden IDentity features | 29 MB | 66.5% | Fast | Compact but less accurate |
| ArcFace | Additive Angular Margin Loss | 112 MB | 96.7% | Medium | Excellent for challenging angles and lighting |
| Dlib | Dlib's ResNet model | 94 MB | 96.8% | Slow | Robust but slower than other options |
| SFace | Efficient face recognition model | 112 MB | 93.0% | Fast | Good for resource-constrained environments |
| GhostFaceNet | Lightweight face recognition | 24 MB | 93.3% | Very Fast | Best for mobile applications |

### Model Performance Comparison

The following table shows the measured accuracy scores in DeepFace compared to the scores reported in the original studies:

| Model | Measured Score | Declared Score |
|-------|---------------|----------------|
| Facenet512 | 98.4% | 99.6% |
| Human-beings | 97.5% | 97.5% |
| Facenet | 97.4% | 99.2% |
| Dlib | 96.8% | 99.3% |
| VGG-Face | 96.7% | 98.9% |
| ArcFace | 96.7% | 99.5% |
| GhostFaceNet | 93.3% | 99.7% |
| SFace | 93.0% | 99.5% |
| OpenFace | 78.7% | 92.9% |
| DeepFace | 69.0% | 97.3% |
| DeepID | 66.5% | 97.4% |

### Choosing the Right Model

- For highest accuracy: **Facenet512**
- For best speed/accuracy balance: **Facenet** or **VGG-Face**
- For resource-constrained environments: **GhostFaceNet**
- For challenging poses and lighting: **ArcFace**

## Face Detection Backends

DeepFace supports multiple face detection backends:

| Detector | Speed | Accuracy | Notes |
|----------|-------|----------|-------|
| opencv | Very Fast | Moderate | Default, best for speed |
| ssd | Fast | Good | Good balance of speed and accuracy |
| dlib | Medium | Good | Reliable detection |
| mtcnn | Slow | Very Good | Accurate but slower |
| retinaface | Slow | Excellent | Best accuracy, especially for challenging images |
| mediapipe | Fast | Good | Good for video streams |
| yolov8 | Medium | Very Good | Good for multiple faces |
| yunet | Fast | Good | Efficient detection |
| centerface | Fast | Good | Fast with good accuracy |

## Similarity Metrics

DeepFace supports different distance metrics to determine the similarity between face embeddings. The choice of metric can affect both accuracy and the interpretation of results.

| Metric | Description | Threshold | Interpretation | Best For |
|--------|-------------|-----------|----------------|----------|
| cosine | Measures the cosine of the angle between vectors | 0.68 | Lower is more similar (0-1 range) | Default and works well in most cases |
| euclidean | Measures straight-line distance between vectors | 100 | Lower is more similar (unbounded) | When vector magnitudes matter |
| euclidean_l2 | Normalized euclidean distance | 0.80 | Lower is more similar (0-1 range) | When you need normalized results |

### Choosing the Right Similarity Metric

The choice of similarity metric depends on your specific use case:

- **cosine**: Best for general face recognition tasks. It focuses on the direction of vectors rather than magnitude, making it robust to certain variations.

- **euclidean**: Considers both direction and magnitude. May be more sensitive to the scale of features.

- **euclidean_l2**: Provides normalized results, which can be easier to interpret and set thresholds for.

You can specify the similarity metric in your API calls:

```python
result = DeepFace.verify(
    img1_path=img1_array,
    img2_path=img2_array,
    model_name="ArcFace",
    distance_metric="cosine"  # Options: "cosine", "euclidean", "euclidean_l2"
)
```

### Impact on Verification Results

Different metrics will produce different distance values and use different thresholds:

- With **cosine**, two faces are considered the same person if the distance is less than 0.68
- With **euclidean**, the threshold is typically around 100
- With **euclidean_l2**, the threshold is approximately 0.80

The API returns both the calculated distance and the threshold used, allowing you to understand how confident the system is in its verification result.

## Configuration Options

The API can be configured with various options:

- **Face Recognition Model**: Change the model used for face comparison
- **Face Detection Backend**: Select the method used to detect faces in images
- **Similarity Metrics**: Choose between cosine, euclidean, or euclidean_l2
- **Enforce Detection**: Set to true to ensure faces are detected before comparison

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: If you encounter errors about missing modules, ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. **Model Download Issues**: The first time you run a comparison, DeepFace will download the required model files. Ensure you have internet connectivity.

3. **Memory Issues**: Face recognition can be memory-intensive. If you encounter memory errors, try:
   - Using a lighter model like GhostFaceNet
   - Processing smaller images
   - Increasing your system's swap space

4. **S3 Connection Issues**: If you're using S3 integration, verify your AWS credentials and network connectivity.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [DeepFace](https://github.com/serengil/deepface) for the face recognition framework
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework 