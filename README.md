# Adaptive ML-Based Video Compression Using Segment-wise CRF Prediction

## Project Overview
This project implements an adaptive video compression system that uses machine learning to predict optimal CRF (Constant Rate Factor) values for individual video segments based on content characteristics, targeting specific VMAF (Video Multimethod Assessment Fusion) quality scores.

## Features
- Segment-wise video processing
- Content-adaptive CRF prediction
- VMAF-driven quality optimization
- Automated feature extraction from video segments
- Performance evaluation and metrics logging

## Requirements
- Python 3.x
- FFmpeg with libvmaf support
- Required Python packages:
  - numpy
  - pandas
  - scikit-learn
  - opencv-python
  - joblib
  - tqdm
  - matplotlib
  - seaborn

## Project Structure
```
ml_project/
├── script.ipynb          # Main Jupyter notebook containing all code
├── crf_model.pkl         # Trained CRF prediction model
├── vmaf_model.pkl        # VMAF prediction model
├── final_dataset.csv     # Processed dataset for model training
├── crf_log.csv          # Encoding decisions log
├── metrics_summary.csv   # Quality metrics summary
└── segments/            # Directory for video segments
```

## Pipeline Steps

### 1. Data Collection and Preprocessing
- Video segmentation into fixed-duration chunks
- Feature extraction from video segments:
  - Spatial variance
  - Motion estimation
  - Entropy calculation
  - Resolution features

### 2. Model Training
- Random Forest model for CRF prediction
- Features used:
  - Spatial complexity metrics
  - Temporal complexity (motion)
  - Resolution information
  - Target VMAF score

### 3. Adaptive Encoding
- Per-segment CRF prediction
- FFmpeg-based encoding with predicted CRF values
- Quality metrics computation (PSNR, SSIM, VMAF)

### 4. Performance Evaluation
- VMAF score verification
- Compression ratio analysis
- Target vs achieved quality comparison

## Key Components

### Feature Extraction
```python
def extract_features(y_frame):
    # Compute spatial variance, entropy, and motion features
    # from video frame data
```

### CRF Prediction
```python
def predict_crf(features, target_vmaf):
    # Use trained model to predict optimal CRF value
    # based on content features and target quality
```

### Quality Assessment
```python
def compute_vmaf(reference, encoded):
    # Calculate VMAF score between reference and
    # encoded video segments
```

## Performance Metrics
- VMAF prediction accuracy
- CRF prediction error
- Compression efficiency
- Processing time per segment

## Usage
1. Place input video in YUV format in the project directory
2. Configure parameters in the notebook:
   - Video dimensions
   - Frame rate
   - Segment duration
   - Target VMAF score
3. Run the notebook cells sequentially
4. Check output logs and metrics for results

## Results
The system achieves:
- Adaptive compression based on content complexity
- Consistent quality across different content types
- Automated CRF selection for optimal quality-size trade-off
- Detailed performance logging and analysis

## Future Improvements
- Real-time processing capabilities
- Support for more video formats
- Enhanced motion estimation
- GPU acceleration
- Integration with streaming platforms

## License
This project is open-source and available under the MIT License.

## Acknowledgments
- VMAF development by Netflix
- FFmpeg project
- scikit-learn community
