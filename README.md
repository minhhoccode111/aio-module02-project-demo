# 👤 Check Employee Attendance System

A smart face recognition system for employee attendance tracking using FAISS vector search and deep learning embeddings for real-time identification.

## 🚀 Live Demo

[![Streamlit App](https://img.shields.io/badge/👁️‍🗨️_Streamlit-Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://aio2025-check-attendance.streamlit.app/)

> 🎯 **Try the live demo above to test system without any setup!**

## 🎯 Project Overview

This project builds a real-time employee face attendance system that:

- Uses FAISS for efficient similarity search of face embeddings
- Implements a two-level embedding approach (raw pixel vectors and deep learning feature maps)
- Leverages InceptionResNetV1 (via `facenet-pytorch`) for robust feature extraction
- Provides a user-friendly Streamlit interface for real-time face recognition
- Identifies employees by comparing image vectors with a predefined threshold

## 🏗️ Architecture

- **Vector Search**: FAISS for fast similarity search
- **Deep Learning Model**: InceptionResNetV1 (via `facenet-pytorch`) for feature extraction (used as a black box, no retraining)
- **Web Framework**: Streamlit for the user interface
- **Data Processing**: NumPy for raw pixel vector conversion, OpenCV for image processing
- **Input**: Real-time camera input or uploaded images
- **Visualization**: Matplotlib and Plotly for result analysis

## 📊 Dataset

- **Source**: [Google Drive](https://drive.google.com/file/d/1WINKn7eU2RizsImBhsb-pPWisulzPK09/view?usp=sharing)
- **Format**: Images in standard formats (e.g., JPG, PNG)
- **Preparation**: Images should be high-resolution with consistent lighting for best results (you can add your images)

## 🛠️ Technology Stack

- **Vector Search**: [FAISS](https://github.com/facebookresearch/faiss) (`faiss-cpu==1.11.0`)
- **Web Framework**: Streamlit (`streamlit==1.46.1`)
- **Deep Learning**: InceptionResNetV1 via `facenet-pytorch==2.6.0`, `torchvision==0.17.2`
- **Data Processing**: NumPy (`numpy==1.26.0`), OpenCV (`opencv-python-headless==4.8.1.78`)
- **Visualization**: Matplotlib (`matplotlib==3.10.3`), Plotly (`plotly==6.2.0`)
- **Machine Learning**: scikit-learn (`scikit-learn==1.7.0`)
- **Language**: Python 3.8+ (Current Use: 3.11)

## 🚀 Quick Start

> **💡 Want to try it first?** Check out the [live demo](#-live-demo) above for instant access without any setup!

### Local Setup

```bash
# Clone the repository
git clone https://github.com/dangnha/AIO_Module02_Project_Demo
cd AIO_Module02_Project_Demo

# Install dependencies
pip install -r requirements.txt

# Add your tailord images in ./Dataset folder

# Start the Streamlit app
streamlit run app.py
```

## 📋 Prerequisites

- Python 3.8+
- Git
- Webcam or camera device (for real-time input)
- Dependencies listed in `requirements.txt`:
  - `numpy==1.26.0`
  - `facenet-pytorch==2.6.0`
  - `faiss-cpu==1.11.0`
  - `torchvision==0.17.2`
  - `streamlit==1.46.1`
  - `opencv-python-headless==4.8.1.78`
  - `matplotlib==3.10.3`
  - `plotly==6.2.0`
  - `scikit-learn==1.7.0`
- Building Faiss index colab code: [Google Colab code](https://drive.google.com/file/d/1zsrSx0oHLhw7dTL4MljzEVLFMKTGwV0V/view?usp=drive_link)

## 🎮 Features

### 🔍 Recognition Workflow

- **Real-time Input**: Capture images via webcam or upload
- **Two-level Embedding**:
  - Level 1: Raw pixel vectors using NumPy
  - Level 2: Feature maps from InceptionResNetV1 (via `facenet-pytorch`)
- **Similarity Search**: FAISS queries for closest vector match
- **Threshold-based Recognition**: Displays employee name if similarity exceeds threshold, else "Not Recognized"
- **Performance Testing**: Evaluate with in-dataset and out-of-dataset images

### 🎨 User Interface

- Real-time camera feed display
- Instant recognition results
- Confidence scores for matches
- Manual image upload option
- Visualization of results with Matplotlib/Plotly

### 🧠 AI Features

- Efficient vector search with FAISS
- Robust feature extraction with InceptionResNetV1
- L2 distance-based/ Dot Product similarity comparison

## 📁 Project Structure

```text
AIO_Module02_Project_Demo/
├── app.py                  # Main Streamlit application
├── facenet_label_map.npy   # Labels of Index
├── facenet_features.index  # Faiss Index
├── requirements.txt        # Python dependencies
├── Dataset/                # Dataset
│   └── Avatar_employee-name.jpg
└── README.md           # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.
