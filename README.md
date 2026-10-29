
# Real-Time Object Detection Using OpenCV

## 📌 Project Overview

This project implements a real-time object detection system using Python, OpenCV, NumPy, and ONNX Runtime.

The system captures video from a webcam and detects objects in real time using a pretrained MobileNet ONNX model. It displays bounding boxes, object names, and confidence scores around detected objects.

## 🎯 Objectives

- Detect objects using a webcam in real time.
- Use a pretrained deep learning model for object detection.
- Display detected objects with bounding boxes.
- Show object names and confidence scores.
- Learn the basics of computer vision and deep learning model deployment.

## 🛠️ Technologies Used

- **Python** – Programming language
- **OpenCV** – Webcam access and image processing
- **NumPy** – Numerical operations and image array handling
- **ONNX Runtime** – Running the pretrained ONNX model
- **MobileNet** – Pretrained object detection model

## ✨ Features

- Real-time webcam object detection
- Detection of multiple object categories
- Bounding boxes around detected objects
- Confidence scores displayed as percentages
- Confidence threshold of 50%
- Simple keyboard control to stop detection

## 📂 Project Structure

```text
Object-Detection/
│
├── object_detection.py
├── mobilenet.onnx
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/SakthiMeera-06/Real-time-object-detection.git
```

Replace `SakthiMeera-06` and `Real-time-object-detection` with your GitHub username and repository name.

### 2. Navigate to the Project Folder

```bash
cd Real-time-object-detection
```

### 3. Create a Virtual Environment (Optional)

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install the Required Libraries

```bash
pip install opencv-python numpy onnxruntime
```

### 5. Download the Model

Place the pretrained `mobilenet.onnx` model in the project folder.

Ensure that the model is compatible with the input format and output names expected by the Python script.

### 6. Run the Project

```bash
python object_detection.py
```

Allow webcam access if prompted.

## ▶️ How It Works

1. The program loads the pretrained MobileNet ONNX model.
2. The webcam captures video frames.
3. Each frame is converted from BGR to RGB format.
4. The image is passed to the model for object detection.
5. The model returns bounding boxes, class labels, and confidence scores.
6. Detections below the 50% confidence threshold are ignored.
7. Bounding boxes and labels are drawn on the video frame.
8. The processed video is displayed in a window.

Press **Q** to stop the program.

## 📦 Supported Object Categories

The model is configured to recognize COCO object categories, including:

- Person
- Bicycle
- Car
- Motorcycle
- Bus
- Truck
- Cat and dog
- Birds and other animals
- Bottles and cups
- Chairs and other furniture
- Laptops and cell phones

The actual detections depend on the pretrained model and its supported classes.

## 🖥️ Expected Output

A webcam window titled `Real-Time Object Detection` appears.

When an object is detected, the system displays:
- A green bounding box around the object
- The detected object's name
- Its confidence score as a percentage

Example: `person: 95%`

The displayed confidence score varies depending on the detection.

## 🔮 Future Enhancements

- Add object tracking across video frames.
- Save detected images and video recordings.
- Add a graphical user interface.
- Improve detection speed and accuracy.
- Display the total number of detected objects.
- Add support for video file input.
- Explore GPU acceleration for faster inference.

## 🎓 Learning Outcomes

Through this project, I learned about:

- Real-time computer vision
- Webcam handling using OpenCV
- Pretrained deep learning models
- ONNX Runtime inference
- Bounding box visualization
- Confidence-based object filtering

## 👩‍💻 Author

**Your Name**

GitHub: [Your GitHub Profile](https://github.com/SakthiMeera-06)

## 📄 License

This project is intended for educational and learning purposes.
