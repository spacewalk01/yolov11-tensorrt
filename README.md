<div align="center">

TensorRT-YOLOv11
===========================

[![python](https://img.shields.io/badge/python-3.10.12-green)](https://www.python.org/downloads/release/python-31012/)
[![cuda](https://img.shields.io/badge/cuda-11.6-green)](https://developer.nvidia.com/cuda-downloads)
[![trt](https://img.shields.io/badge/TRT-8.6-green)](https://developer.nvidia.com/tensorrt)
[![mit](https://img.shields.io/badge/license-MIT-blue)](https://github.com/spacewalk01/TensorRT-YOLOv9/tree/main?tab=MIT-1-ov-file#readme)

<div align="left">
 
This repo hosts a C++ and python implementation of the [YOLOv11](https://github.com/ultralytics/ultralytics) state of the art object detection model, leveraging the TensorRT API for efficient real-time inference.
<p align="center" margin: 0 auto;>
</p>

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/spacewalk01/yolov11-tensorrt.git
   cd yolov11-tensorrt
   ```

2. Install dependencies:
   - For Python:
     ```bash
     pip install --upgrade ultralytics
     ```
   - For C++: Set OpenCV and TensorRT paths in the `CMakeLists.txt` file.

3. Compile the C++ code:
   ```bash
   mkdir build && cd build
   cmake ..
   cmake --build . --config Release
   ```

## Usage

### Python

    1. Set desired model name in export.py 
    2. Run the Python inference script to export the model into onnx:

    ```bash
    python export.py
    ```

### C++

Create an engine from the onnx model:
```bash
./yolov11-tensorrt.exe yolov11.onnx ""
```

Run an inference on the image:
```bash
./yolov11-tensorrt.exe yolov11.engine "zidane.jpg"
```

Run an inference on the video:
```bash
./yolov11-tensorrt.exe yolov11.engine "road.mp4"
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
