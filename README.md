# Object Detection using YOLOv8 with a Custom Dataset

This project demonstrates how to perform object detection using YOLOv8 on a custom dataset. We will walk through the process of training a YOLOv8 model on a custom dataset and performing inference on both images and videos.

## Requirements

Before you begin, you will need the following dependencies:

- Python 3.x
- Ultralytics YOLOv8: Install this by running `!pip install ultralytics`.
- IPython: For displaying outputs in Jupyter notebooks.

## Setup

### Install Dependencies

First, install the required libraries by running the following command:

```bash
!pip install ultralytics
```

### Clear Output (Optional)

In Jupyter notebooks, use the following code to clear the output:

```python
from IPython import display
display.clear_output()
```

### Check Installation

Verify that YOLOv8 is installed properly using the following command:

```python
import ultralytics
ultralytics.checks()
```

## Training the YOLOv8 Model

### Import YOLOv8 Model

To use YOLOv8, import the model from the `ultralytics` package:

```python
from ultralytics import YOLO
```

### Training Command

Use the following command to train the YOLOv8 model on your custom dataset. The dataset should be prepared in the YOLO format, and you will need to define the path to the dataset configuration file (`data.yaml`).

```bash
!yolo task=detect mode=train model=yolov8s.pt data=/content/drive/MyDrive/screwdata/data.yaml epochs=200 imgsz=800 plots=True
```

**Explanation:**
- `task=detect`: The task is object detection.
- `mode=train`: We are training the model.
- `model=yolov8s.pt`: YOLOv8 small model weights as the starting point.
- `data=/content/drive/MyDrive/screwdata/data.yaml`: Path to your custom dataset YAML file.
- `epochs=200`: Train for 200 epochs.
- `imgsz=800`: Resize images to 800x800 pixels.
- `plots=True`: Display training plots.

This command will train the YOLOv8 model on your dataset and save the best weights during training.

## Image Inference

To perform object detection on images, use the following code:

```python
from IPython.display import display, Image
```

Run the inference command after training is complete:

```bash
!yolo task=detect mode=predict model=/content/runs/detect/train/weights/best.pt source='/content/drive/MyDrive/screwdata/testimage.jpg'
```

**Explanation:**
- `task=detect`: The task is object detection.
- `mode=predict`: Run inference mode.
- `model=/content/runs/detect/train/weights/best.pt`: Path to the best-trained model.
- `source='/content/drive/MyDrive/screwdata/testimage.jpg'`: Path to the image you want to infer.

This command will display the image with detected objects.

## Video Inference

To perform object detection on a video, use the following command:

```bash
!yolo task=detect mode=predict model=/content/runs/detect/train/weights/best.pt source='/content/drive/MyDrive/screwdata/testvideo.mp4'
```

**Explanation:**
- `task=detect`: The task is object detection.
- `mode=predict`: Run inference mode.
- `model=/content/runs/detect/train/weights/best.pt`: Path to the best-trained model.
- `source='/content/drive/MyDrive/screwdata/testvideo.mp4'`: Path to the video you want to infer.

This will process the video and output the detected objects in each frame.

## Conclusion

In this project, we've demonstrated how to use YOLOv8 for object detection with a custom dataset. The key steps include:
1. Installing necessary libraries.
2. Training the model using your dataset.
3. Running inference on images and videos.

Feel free to customize the training parameters, dataset, and inference files for your specific use case. Also Use best.pt file for Detection Counting and Tracking Happy detecting!
