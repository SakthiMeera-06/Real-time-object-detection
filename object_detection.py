
import cv2
import numpy as np
import onnxruntime as ort

# Load the pretrained object detection model
model_path = "mobilenet.onnx"
session = ort.InferenceSession(
    model_path,
    providers=["CPUExecutionProvider"]
)

# COCO object labels
classes = [
    "background", "person", "bicycle", "car", "motorcycle",
    "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "N/A", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella",
    "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "N/A", "dining table", "N/A", "N/A",
    "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "N/A", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

# Start webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Webcam could not be opened.")
    raise SystemExit

print("Object detection started. Press Q to quit.")

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read webcam frame.")
        break

    # Model input expects RGB, uint8, NHWC format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(rgb_frame, axis=0).astype(np.uint8)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_data})

    # Read model outputs by their names
    output_names = [item.name for item in session.get_outputs()]
    result = dict(zip(output_names, outputs))

    boxes = result["detection_boxes"][0]
    scores = result["detection_scores"][0]
    labels = result["detection_classes"][0]
    count = int(np.asarray(result["num_detections"]).reshape(-1)[0])

    height, width = frame.shape[:2]

    for i in range(count):
        score = float(scores[i])

        if score < 0.50:
            continue

        class_id = int(labels[i])

        if class_id < 0 or class_id >= len(classes):
            continue

        name = classes[class_id]

        if name == "N/A" or name == "background":
            continue

        # Convert normalized box coordinates to pixels
        top, left, bottom, right = boxes[i]

        x1 = max(0, min(width - 1, int(left * width)))
        y1 = max(0, min(height - 1, int(top * height)))
        x2 = max(0, min(width - 1, int(right * width)))
        y2 = max(0, min(height - 1, int(bottom * height)))

        if x2 <= x1 or y2 <= y1:
            continue

        # Draw box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{name}: {score:.0%}"
        cv2.putText(
            frame, text, (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

    cv2.imshow("Real-Time Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
