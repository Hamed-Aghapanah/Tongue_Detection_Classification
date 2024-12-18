# Tongue Detection and Classification

![Project Logo](https://github.com/Hamed-Aghapanah/Tongue_Detection_Classification/blob/main/images/l3.JPG)

This repository provides the implementation of tongue detection and classification tasks. The detection part is based on the implementation from [Tongue Detection](https://github.com/Hamed-Aghapanah/Tongue_Detection).

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![Dataset](https://img.shields.io/badge/Dataset-Google%20Drive-blue)](https://drive.google.com/drive/folders/1fXWutGWE33VfQ6bAHpHwL9Nube3xS3fc?usp=sharing)

## Overview

The purpose of this repository is to extend the detection functionality to include classification of tongue images into predefined categories. The detection model identifies the region of interest, and the classification model categorizes the detected region.

## Dataset

The dataset used for training and testing can be accessed at the following link:
[![Dataset Link](https://github.com/Hamed-Aghapanah/Tongue_Detection_Classification/blob/main/images/001.JPG)](https://github.com/Hamed-Aghapanah/Tongue_Detection_Classification/blob/main/images/001.JPG)

Ensure you download and prepare the dataset as described in the dataset documentation.

## Installation

To set up the environment, install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Files and Scripts

- `train.py`: Script for training the classification model.
- `train_aux.py`: Additional training script with auxiliary functionalities.
- `test.py`: Script to evaluate the trained model.
- `detect.py`: Script for performing detection on new images.
- `camera1.py`: Example script for real-time detection using a webcam.
- `requirements.txt`: List of dependencies.
- `read_me.txt`: Additional notes and documentation.

## Training the Model

Run the following command to train the model:

```bash
python train.py --cfg cfg/training/yolov7-tiny.yaml --weights best_tongue.pt --data data/Tongue.yaml --batch-size 8 --epochs 600 --workers 0
```

Make sure to adjust the parameters as per your requirements.

## Testing and Inference

### Testing the Model
To test the model and evaluate its performance, use:

```bash
python test.py --weights best_tongue.pt --data data/Tongue.yaml --batch-size 32
```

### Real-time Detection
For real-time detection using a webcam, run:

```bash
python detect.py --weights best_tongue.pt --source 0
```

## Requirements

The key dependencies are:

- Python
- PyTorch
- OpenCV
- NumPy

Refer to `requirements.txt` for the full list of dependencies:

```plaintext
matplotlib>=3.2.2
numpy>=1.18.5,<1.24.0
opencv-python>=4.1.1
Pillow>=7.1.2
PyYAML>=5.3.1
requests>=2.23.0
scipy>=1.4.1
torch>=1.7.0,!=1.12.0
torchvision>=0.8.1,!=0.13.0
tqdm>=4.41.0
protobuf<4.21.3
```

## Sample Results

### Example Video

Check out a sample video showcasing the detection and classification process:

![Sample Video](https://github.com/Hamed-Aghapanah/Tongue_Detection_Classification/blob/main/images/avc_classification_s-ezgif.com-speed.gif "Sample Video")

## Credits

The detection functionality is adapted from the [Tongue Detection](https://github.com/Hamed-Aghapanah/Tongue_Detection) repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
