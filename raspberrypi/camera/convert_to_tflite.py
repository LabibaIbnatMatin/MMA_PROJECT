"""
convert_to_tflite.py

Helper to convert a Keras .h5 model to a (float16) TFLite model. Also includes an optional
representative dataset generator stub for INT8 quantization.

Usage (Colab):
  python3 convert_to_tflite.py --h5 best_model.h5 --out model.tflite --float16

For INT8 quantization provide --representative_dir pointing to a small folder of images.
"""
import argparse
import tensorflow as tf
import numpy as np
import os
from PIL import Image


def representative_data_gen(image_dir, input_size=(224,224), num_samples=100):
    files = []
    for root, _, fnames in os.walk(image_dir):
        for f in fnames:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                files.append(os.path.join(root, f))
    files = files[:num_samples]
    for f in files:
        img = Image.open(f).convert('RGB').resize(input_size)
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)
        yield [arr]


def convert(h5_path, out_path, float16=False, representative_dir=None):
    model = tf.keras.models.load_model(h5_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if float16:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    if representative_dir:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: representative_data_gen(representative_dir, num_samples=50)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    with open(out_path, 'wb') as f:
        f.write(tflite_model)
    print('Saved tflite model to', out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--float16', action='store_true')
    parser.add_argument('--representative_dir', default=None)
    args = parser.parse_args()
    convert(args.h5, args.out, float16=args.float16, representative_dir=args.representative_dir)
