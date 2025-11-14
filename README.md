# img_det

A Python-based image object detection web app using a YOLO model.

## Overview

This application uses a YOLOv8 model to detect objects in images and serves the results via a web interface. It is built with Flask for the backend and a simple HTML/JavaScript frontend.

## Features

* Upload an image and get object detection results with bounding boxes and labels.
* Uses `yolov8s.pt` pretrained model.
* Lightweight web UI for easy testing and demo.
* Suitable for prototyping image detection workflows.

## Tech Stack

* Python 3.x
* Flask
* YOLOv8 (PyTorch)
* HTML / JavaScript frontend

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/manvip28/img_det.git
   cd img_det
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the Flask server:

```bash
python app.py
```

Then open your browser and navigate to:

```
http://127.0.0.1:5000
```

Upload an image to view detection results.

## Model

The file `yolov8s.pt` contains the pretrained YOLOv8 model weights used for detection. You can replace it with your own custom-trained model if needed.

## Why this project

This project helps you quickly set up an image detection pipeline with minimal code and infrastructure. Useful for demos, prototyping, or as a foundation for larger computer vision applications.

## Contributing

Feel free to fork the repo, enhance the UI, improve performance, or add new features. Pull requests are welcome.

## License

MIT License (or specify your preferred license).
