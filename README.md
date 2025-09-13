# Pi TFLite inference (hand-first)

This folder contains a small script to run TFLite inference on the Raspberry Pi with a hand-first flow (MediaPipe). It crops around the detected hand, runs the model, and sends simple UART commands to an ESP32.

Files:
- `tflite_inference.py` - main script
- `requirements.txt` - suggested Python packages; pin versions on Pi as needed

Quick start on Raspberry Pi (example):

1. Create and activate a virtual environment:

```powershell
python3 -m venv ~/venv-smartbin; . ~/venv-smartbin/Scripts/Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

2. Copy your `model.tflite` and `labels.txt` into this folder.

3. Run the script (camera 0):

```powershell
python tflite_inference.py --model model.tflite --labels labels.txt --camera 0 --serial /dev/ttyUSB0 --show
```

Notes:
- On Raspberry Pi you should prefer `tflite-runtime` instead of full `tensorflow` for speed and lower memory.
- If you built a quantized INT8 model you will need a representative dataset for conversion and `tflite_runtime` that supports quantized ops.
- Adjust `--threshold` and `--stability-frames` for your use case.
