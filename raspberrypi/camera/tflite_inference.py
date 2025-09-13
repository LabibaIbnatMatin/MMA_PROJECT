#!/usr/bin/env python3
"""
tflite_inference.py

Pi-side script: detect a hand with MediaPipe, crop the ROI, run a TFLite model, and send simple UART commands
to an ESP32 when a stable prediction is observed.

Usage examples:
  python3 tflite_inference.py --model model.tflite --labels labels.txt --serial /dev/ttyUSB0 --baud 115200

On Windows the serial port might be COM3. For MJPG stream as camera source:
  python3 tflite_inference.py --model model.tflite --camera http://192.168.1.100:8080/?action=stream

This script tries to use `tflite_runtime` if available, falling back to `tensorflow`'s Interpreter.
"""
import argparse
import collections
import time
import sys
import math

try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
except Exception:
    try:
        from tensorflow.lite.python.interpreter import Interpreter  # type: ignore
    except Exception:
        Interpreter = None

import cv2
import numpy as np

try:
    import serial
except Exception:
    serial = None

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False


def load_labels(path):
    if not path:
        return None
    labels = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            labels.append(line.strip())
    return labels


def create_interpreter(model_path):
    if Interpreter is None:
        raise RuntimeError('No TFLite Interpreter available. Install tflite-runtime or tensorflow.')
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def set_input_tensor(interpreter, image):
    input_details = interpreter.get_input_details()[0]
    tensor_index = input_details['index']
    # If model expects uint8, cast appropriately
    if input_details.get('dtype') == np.uint8:
        input_data = (image * 255).astype(np.uint8)
    else:
        input_data = (image.astype(np.float32))
    interpreter.set_tensor(tensor_index, np.expand_dims(input_data, axis=0))


def get_output(interpreter):
    output_details = interpreter.get_output_details()[0]
    output_data = interpreter.get_tensor(output_details['index'])[0]
    return output_data


def send_serial_command(ser, cmd):
    if ser is None:
        print('Serial not available; would send:', cmd)
        return
    try:
        ser.write((cmd + '\n').encode('utf-8'))
    except Exception as e:
        print('Serial write failed:', e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to .tflite model')
    parser.add_argument('--labels', default=None, help='Path to labels txt (one per line)')
    parser.add_argument('--camera', default='0', help='Camera source (0 for local camera or MJPG url)')
    parser.add_argument('--serial', default=None, help='Serial port to ESP32 (e.g., /dev/ttyUSB0 or COM3)')
    parser.add_argument('--baud', type=int, default=115200, help='Serial baud rate')
    parser.add_argument('--threshold', type=float, default=0.75, help='Confidence threshold to trigger')
    parser.add_argument('--stability-frames', type=int, default=5, help='Consecutive frames required for stability')
    parser.add_argument('--input-size', type=int, default=224, help='Model input size (assumes square)')
    parser.add_argument('--show', action='store_true', help='Show annotated window (requires a display)')
    args = parser.parse_args()

    labels = load_labels(args.labels) if args.labels else None
    if labels is None:
        # default mapping for 2-class classifier
        labels = ['biodegradable', 'non-biodegradable']

    interpreter = create_interpreter(args.model)
    input_details = interpreter.get_input_details()[0]
    input_shape = input_details['shape']
    in_h = in_w = args.input_size

    # open serial
    ser = None
    if args.serial:
        if serial is None:
            print('pyserial not installed; serial commands will be skipped')
        else:
            try:
                ser = serial.Serial(args.serial, args.baud, timeout=1)
                time.sleep(1.0)
            except Exception as e:
                print('Failed to open serial port:', e)
                ser = None

    # open camera
    cam_src = args.camera
    if cam_src.isdigit():
        cam_index = int(cam_src)
        cap = cv2.VideoCapture(cam_index)
    else:
        cap = cv2.VideoCapture(cam_src)

    if not cap.isOpened():
        print('Failed to open camera source:', cam_src)
        sys.exit(1)

    # mediapipe setup
    if MP_AVAILABLE:
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    else:
        hands = None

    # smoothing buffer
    buf = collections.deque(maxlen=args.stability_frames)
    last_sent = None
    sent_cooldown = 1.5  # seconds

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('Frame read failed, retrying...')
                time.sleep(0.1)
                continue

            orig = frame.copy()
            h, w = frame.shape[:2]

            roi = None
            if hands is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                if results.multi_hand_landmarks:
                    lm = results.multi_hand_landmarks[0]
                    # derive bounding box from landmarks
                    xs = [p.x for p in lm.landmark]
                    ys = [p.y for p in lm.landmark]
                    x_min = max(0, int(min(xs) * w) - 10)
                    x_max = min(w, int(max(xs) * w) + 10)
                    y_min = max(0, int(min(ys) * h) - 10)
                    y_max = min(h, int(max(ys) * h) + 10)
                    # make square crop
                    box_w = x_max - x_min
                    box_h = y_max - y_min
                    side = max(box_w, box_h, 20)
                    cx = x_min + box_w // 2
                    cy = y_min + box_h // 2
                    x1 = max(0, cx - side // 2)
                    y1 = max(0, cy - side // 2)
                    x2 = min(w, cx + side // 2)
                    y2 = min(h, cy + side // 2)
                    roi = orig[y1:y2, x1:x2]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                    # no hand detected
                    roi = None
            # fallback: if no mediapipe or no hand, we can skip inference to save CPU
            if roi is None:
                # annotate and show
                display_text = 'No hand'
                pred_name = None
            else:
                # preprocess ROI
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                roi_resized = cv2.resize(roi_rgb, (in_w, in_h))
                # normalize to [0,1]
                roi_input = roi_resized.astype(np.float32) / 255.0
                set_input_tensor(interpreter, roi_input)
                interpreter.invoke()
                output = get_output(interpreter)
                # handle different output shapes
                if output.ndim == 0:
                    probs = np.array([output])
                else:
                    probs = np.array(output)

                top_idx = int(np.argmax(probs))
                top_conf = float(probs[top_idx])
                pred_name = labels[top_idx] if top_idx < len(labels) else str(top_idx)
                display_text = f'{pred_name} {top_conf:.2f}'

                # smoothing
                buf.append((pred_name, top_conf))
                # check stability: all last frames predict same label and avg confidence > threshold
                if len(buf) == buf.maxlen:
                    names = [p[0] for p in buf]
                    avg_conf = sum(p[1] for p in buf) / len(buf)
                    if names.count(names[0]) == len(names) and avg_conf >= args.threshold:
                        # stable prediction
                        cmd = 'OPEN_BIO' if 'bio' in names[0].lower() else 'OPEN_NONBIO'
                        now = time.time()
                        if last_sent is None or (now - last_sent) > sent_cooldown or cmd != last_sent:
                            print('Sending command:', cmd, 'avg_conf=', avg_conf)
                            send_serial_command(ser, cmd)
                            last_sent = now

            # annotate frame
            if 'display_text' in locals():
                cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

            if args.show:
                cv2.imshow('TFLite Inference', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

    except KeyboardInterrupt:
        print('Interrupted by user')
    finally:
        try:
            cap.release()
        except Exception:
            pass
        if args.show:
            cv2.destroyAllWindows()
        if ser is not None:
            ser.close()


if __name__ == '__main__':
    main()
