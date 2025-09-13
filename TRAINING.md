# Training on Google Colab (step-by-step)

This doc lists exact steps and copy-paste cells to run training and convert to TFLite on Google Colab.

1) Open a new Colab notebook and run the following cells in order.

-- Cell 1: mount Drive and copy dataset to VM (optional but recommended)

```python
from google.colab import drive
drive.mount('/content/drive')
# Adjust this path to your dataset in Drive
DRIVE_DATASET = '/content/drive/MyDrive/archive/Dataset'
!cp -r "{DRIVE_DATASET}" /content/Dataset
print('Copied to /content/Dataset')
```

-- Cell 2: install minimal dependencies (safe for Colab)

```python
# Colab already provides a recent TensorFlow; install only small extras we need
%pip install -q pillow
```

-- Cell 3: full safe single-cell: mount -> flatten nested subclasses -> train -> convert -> validate -> copy

```python
# Paste and run this entire cell. It is safe (copies files, non-destructive) and auto-maps nested subclass
# folders into two semantic classes: 'biodegradable' and 'non-biodegradable'. Adjust DATASET_ROOT if needed.
import os, pathlib, shutil, uuid, random, pprint
from google.colab import drive
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# ---------------- Settings ----------------
DATASET_ROOT = '/content/Dataset'         # original dataset root (from Drive copy)
FLAT_ROOT = '/content/Dataset_flat_final' # flattened output used for flow_from_directory
OUTPUT_DIR = '/content/output'            # models, labels, tflite go here
os_required = True

# mount Drive (if not already mounted)
drive.mount('/content/drive', force_remount=False)

%pip install -q pillow

print('TensorFlow version:', tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print('GPUs visible to TF:', gpus)

os.makedirs(OUTPUT_DIR, exist_ok=True)
if not os.path.isdir(DATASET_ROOT):
    raise RuntimeError(f'DATASET_ROOT not found: {DATASET_ROOT}. Copy your dataset into that path first.')

# ---------------- Subclass -> semantic mapping (adjust as needed) ----------------
BIO_KEYS = ['bio', 'organic', 'food', 'fruit', 'vegetable', 'leaf', 'leaf_', 'paper', 'cardboard', 'wood', 'compost', 'garden', 'food_waste', 'paper_waste','leaf_waste','wood_waste']
NONBIO_KEYS = ['plastic', 'metal', 'can', 'bottle', 'bag', 'e_waste', 'ewaste', 'electronics', 'battery', 'glass', 'tin', 'styrofoam', 'rubber', 'plastic_bottles','plastic_bags','metal_cans','ewaste','e_waste']

semantic = ['biodegradable', 'non-biodegradable']

def decide_parent(name):
    n = name.lower().replace('-', '_').replace(' ', '_')
    for k in BIO_KEYS:
        if k in n:
            return 'biodegradable'
    for k in NONBIO_KEYS:
        if k in n:
            return 'non-biodegradable'
    return None

# ---------------- Flatten routine ----------------
def copy_images(src_dir, dst_dir, prefix):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    os.makedirs(dst_dir, exist_ok=True)
    count = 0
    for p in pathlib.Path(src_dir).rglob('*'):
        if p.is_file() and p.suffix.lower() in exts:
            new_name = f"{prefix}_{uuid.uuid4().hex}{p.suffix.lower()}"
            shutil.copy2(str(p), os.path.join(dst_dir, new_name))
            count += 1
    return count

print('DATASET_ROOT listing (top-level):', sorted(os.listdir(DATASET_ROOT)))
has_train_val = os.path.isdir(os.path.join(DATASET_ROOT, 'train')) and os.path.isdir(os.path.join(DATASET_ROOT, 'val'))
print('Detected train/val split:', has_train_val)

if os.path.exists(FLAT_ROOT):
    print('Warning: FLAT_ROOT exists; new run will reuse existing files in', FLAT_ROOT)
else:
    os.makedirs(FLAT_ROOT, exist_ok=True)

mapped = {}
unmapped = set()
copied_counts = {}

if has_train_val:
    for split in ['train','val']:
        split_root = os.path.join(DATASET_ROOT, split)
        for entry in sorted(os.listdir(split_root)):
            entry_path = os.path.join(split_root, entry)
            if not os.path.isdir(entry_path):
                continue
            if entry.lower().replace('-','_') in ('biodegradable','non_biodegradable','non-biodegradable','non_biodegradable'):
                parent_norm = 'biodegradable' if 'bio' in entry.lower() else 'non-biodegradable'
                for sub in sorted(os.listdir(entry_path)):
                    subpath = os.path.join(entry_path, sub)
                    if os.path.isdir(subpath):
                        dst = os.path.join(FLAT_ROOT, split, parent_norm)
                        n = copy_images(subpath, dst, prefix=f"{split}_{entry}_{sub}")
                        copied_counts[(split,parent_norm)] = copied_counts.get((split,parent_norm),0) + n
                        mapped[f'{split}/{entry}/{sub}'] = parent_norm
                dst_direct = os.path.join(FLAT_ROOT, split, parent_norm)
                n_direct = copy_images(entry_path, dst_direct, prefix=f"{split}_{entry}_direct")
                copied_counts[(split,parent_norm)] = copied_counts.get((split,parent_norm),0) + n_direct
            else:
                parent = decide_parent(entry)
                if parent is None:
                    unmapped.add(f'{split}/{entry}')
                    parent = 'non-biodegradable'
                mapped[f'{split}/{entry}'] = parent
                dst = os.path.join(FLAT_ROOT, split, parent)
                n = copy_images(entry_path, dst, prefix=f"{split}_{entry}")
                copied_counts[(split,parent)] = copied_counts.get((split,parent),0) + n
                print(f'Copied {n} images: {entry_path} -> {dst} (mapped as {parent})')
else:
    for entry in sorted(os.listdir(DATASET_ROOT)):
        entry_path = os.path.join(DATASET_ROOT, entry)
        if not os.path.isdir(entry_path):
            continue
        if entry.lower().replace('-','_') in ('biodegradable','non_biodegradable','non-biodegradable'):
            parent_norm = 'biodegradable' if 'bio' in entry.lower() else 'non-biodegradable'
            for sub in sorted(os.listdir(entry_path)):
                subpath = os.path.join(entry_path, sub)
                if os.path.isdir(subpath):
                    dst = os.path.join(FLAT_ROOT, 'train', parent_norm)
                    n = copy_images(subpath, dst, prefix=f"flat_{entry}_{sub}")
                    copied_counts[('train',parent_norm)] = copied_counts.get(('train',parent_norm),0) + n
                    mapped[f'{entry}/{sub}'] = parent_norm
            n_direct = copy_images(entry_path, os.path.join(FLAT_ROOT, 'train', parent_norm), prefix=f"flat_{entry}_direct")
            copied_counts[('train',parent_norm)] = copied_counts.get(('train',parent_norm),0) + n_direct
        else:
            parent = decide_parent(entry) or 'non-biodegradable'
            mapped[f'{entry}'] = parent
            dst = os.path.join(FLAT_ROOT, 'train', parent)
            n = copy_images(entry_path, dst, prefix=f"flat_{entry}")
            copied_counts[('train',parent)] = copied_counts.get(('train',parent),0) + n
            print(f'Copied {n} images: {entry_path} -> {dst} (mapped as {parent})')
    for parent in semantic:
        tdir = os.path.join(FLAT_ROOT, 'train', parent)
        vdir = os.path.join(FLAT_ROOT, 'val', parent)
        os.makedirs(vdir, exist_ok=True)
        files = sorted([str(p) for p in pathlib.Path(tdir).glob('*') if p.is_file()]) if os.path.isdir(tdir) else []
        if files:
            split_i = int(len(files)*0.8)
            for f in files[split_i:]:
                shutil.copy2(f, os.path.join(vdir, os.path.basename(f)))
            print(f'{parent}: train {split_i}, val {len(files)-split_i}')

print('\nMapping summary (sample):')
pprint.pprint(dict(list(mapped.items())[:40]))
if unmapped:
    print('\nUnmapped entries (defaulted to non-biodegradable):')
    pprint.pprint(sorted(unmapped))
print('\nCopied counts summary:')
pprint.pprint(copied_counts)
print('\nFlattening complete. Using FLAT_ROOT =', FLAT_ROOT)

train_check = {semantic[0]: 0, semantic[1]: 0}
for parent in semantic:
    p = os.path.join(FLAT_ROOT, 'train', parent)
    if os.path.isdir(p):
        train_check[parent] = len([f for f in pathlib.Path(p).glob('*') if f.is_file()])
print('Train counts per parent:', train_check)
if train_check[semantic[0]] == 0 or train_check[semantic[1]] == 0:
    print('WARNING: one semantic class has zero images. Verify mapping or dataset copy before training.')

# ---------------- Data generators ----------------
IMG_SIZE = (224,224)
BATCH_SIZE = 32
EPOCHS = 12

train_dir = os.path.join(FLAT_ROOT, 'train')
val_dir = os.path.join(FLAT_ROOT, 'val')
if not (os.path.isdir(train_dir) and os.path.isdir(val_dir)):
    raise RuntimeError(f'After flattening expected train/val under {FLAT_ROOT}. Found: {os.listdir(FLAT_ROOT)}')

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.1,
                                   height_shift_range=0.1, shear_range=0.1, zoom_range=0.15,
                                   horizontal_flip=True, fill_mode='nearest')
train_gen = train_datagen.flow_from_directory(train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True)
val_datagen = ImageDataGenerator(rescale=1./255)
val_gen = val_datagen.flow_from_directory(val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

# ---------------- Build & train model ----------------
def build_model(num_classes, input_shape=(224,224,3), dropout=0.3):
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=outputs)
    for layer in base.layers:
        layer.trainable = False
    return model

num_classes = train_gen.num_classes
print('Num classes:', num_classes, 'class_indices:', train_gen.class_indices)

model = build_model(num_classes, input_shape=(IMG_SIZE[0],IMG_SIZE[1],3))
model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

ckpt_path = os.path.join(OUTPUT_DIR, 'best_model.h5')
ckpt = ModelCheckpoint(ckpt_path, monitor='val_accuracy', save_best_only=True, mode='max')
rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
early = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

steps_per_epoch = max(1, train_gen.samples // BATCH_SIZE)
val_steps = max(1, val_gen.samples // BATCH_SIZE)
print('Train samples:', train_gen.samples, 'Val samples:', val_gen.samples)
history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, steps_per_epoch=steps_per_epoch, validation_steps=val_steps, callbacks=[ckpt, rlrop, early])

# ---------------- Optional fine-tune ----------------
for layer in model.layers:
    layer.trainable = True
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=6, steps_per_epoch=steps_per_epoch, validation_steps=val_steps, callbacks=[rlrop, early])

final_h5 = os.path.join(OUTPUT_DIR, 'final_model.h5')
model.save(final_h5)
print('Saved final model to', final_h5)

# ---------------- Save labels ----------------
idx_to_label = {v:k for k,v in train_gen.class_indices.items()}
labels_path = os.path.join(OUTPUT_DIR, 'labels.txt')
with open(labels_path, 'w') as f:
    for i in range(len(idx_to_label)):
        f.write(idx_to_label[i] + '\n')
print('Saved labels to', labels_path, 'mapping:', idx_to_label)

# ---------------- Convert to float16 TFLite ----------------
from tensorflow import lite
best_h5 = ckpt_path if os.path.exists(ckpt_path) else final_h5
tflite_out = os.path.join(OUTPUT_DIR, 'model_float16.tflite')
print('Converting', best_h5, '->', tflite_out)
try:
    converter = lite.TFLiteConverter.from_keras_model(tf.keras.models.load_model(best_h5))
    converter.optimizations = [lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    open(tflite_out, 'wb').write(tflite_model)
    print('Saved tflite:', tflite_out)
except Exception as e:
    print('TFLite conversion failed:', e)

# ---------------- Quick validation ----------------
print('Running quick validation...')
try:
    interpreter = lite.Interpreter(model_path=tflite_out)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    out_details = interpreter.get_output_details()[0]
    input_h, input_w = input_details['shape'][1], input_details['shape'][2]
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    samples = []
    for i in sorted(idx_to_label.keys()):
        label = idx_to_label[i]
        cand_dirs = [os.path.join(FLAT_ROOT, 'val', label), os.path.join(FLAT_ROOT, 'train', label)]
        found = None
        for d in cand_dirs:
            if os.path.isdir(d):
                imgs = [p for p in pathlib.Path(d).glob('*') if p.is_file() and p.suffix.lower() in exts]
                if imgs:
                    found = str(random.choice(imgs)); break
        if found:
            samples.append((label, found))
    for label, img_path in samples:
        img = Image.open(img_path).convert('RGB').resize((input_w, input_h))
        arr = np.array(img).astype(np.float32) / 255.0
        inp = np.expand_dims(arr, axis=0)
        if input_details['dtype'] == np.uint8:
            inp = (inp * 255).astype(np.uint8)
        interpreter.set_tensor(input_details['index'], inp)
        interpreter.invoke()
        out = interpreter.get_tensor(out_details['index'])[0]
        pred_idx = int(np.argmax(out))
        pred_label = idx_to_label.get(pred_idx, str(pred_idx))
        pred_conf = float(out[pred_idx])
        print(f'{img_path} -> pred {pred_label} ({pred_conf:.3f}) ; true {label}')
except Exception as e:
    print('Validation failed:', e)

# ---------------- Copy outputs to Drive ----------------
try:
    drive_root = '/content/drive/MyDrive'
    shutil.copy2(tflite_out, os.path.join(drive_root, 'model_float16.tflite'))
    shutil.copy2(labels_path, os.path.join(drive_root, 'labels.txt'))
    shutil.copy2(final_h5, os.path.join(drive_root, 'final_model.h5'))
    print('Copied outputs to Drive:', drive_root)
except Exception as e:
    print('Could not copy to Drive:', e)

print('Done. Outputs in', OUTPUT_DIR)
```

-- Optional: quick flatten-check cell (run this before training to verify mapping)

```python
# Run only the flattening and mapping check to confirm classes and counts without training.
import os, pathlib, shutil, uuid, pprint
DATASET_ROOT = '/content/Dataset'
FLAT_ROOT = '/content/Dataset_flat_check'
os.makedirs(FLAT_ROOT, exist_ok=True)

BIO_KEYS = ['bio', 'organic', 'food', 'fruit', 'vegetable', 'leaf', 'leaf_', 'paper', 'cardboard', 'wood', 'compost', 'garden', 'food_waste', 'paper_waste','leaf_waste','wood_waste']
NONBIO_KEYS = ['plastic', 'metal', 'can', 'bottle', 'bag', 'e_waste', 'ewaste', 'electronics', 'battery', 'glass', 'tin', 'styrofoam', 'rubber', 'plastic_bottles','plastic_bags','metal_cans','ewaste','e_waste']

def decide_parent(name):
    n = name.lower().replace('-', '_').replace(' ', '_')
    for k in BIO_KEYS:
        if k in n:
            return 'biodegradable'
    for k in NONBIO_KEYS:
        if k in n:
            return 'non-biodegradable'
    return None

def count_images(p):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    return sum(1 for _ in pathlib.Path(p).glob('*') if _.is_file() and _.suffix.lower() in exts)

print('Top-level:', sorted(os.listdir(DATASET_ROOT)))
has_train_val = os.path.isdir(os.path.join(DATASET_ROOT,'train')) and os.path.isdir(os.path.join(DATASET_ROOT,'val'))
print('Detected train/val split:', has_train_val)

mapped = {}
unmapped = []
if has_train_val:
    for split in ['train','val']:
        for entry in sorted(os.listdir(os.path.join(DATASET_ROOT,split))):
            p = os.path.join(DATASET_ROOT,split,entry)
            if not os.path.isdir(p):
                continue
            parent = decide_parent(entry)
            if parent is None:
                # maybe entry is a semantic parent containing subclasses
                if entry.lower().replace('-','_') in ('biodegradable','non_biodegradable','non-biodegradable'):
                    # count nested
                    total = 0
                    for sub in os.listdir(p):
                        subp = os.path.join(p, sub)
                        if os.path.isdir(subp):
                            total += count_images(subp)
                    mapped[f'{split}/{entry}'] = ('semantic-parent', total)
                else:
                    unmapped.append(f'{split}/{entry}')
            else:
                mapped[f'{split}/{entry}'] = (parent, count_images(p))
else:
    for entry in sorted(os.listdir(DATASET_ROOT)):
        p = os.path.join(DATASET_ROOT, entry)
        if not os.path.isdir(p):
            continue
        parent = decide_parent(entry)
        mapped[entry] = (parent or 'unknown', count_images(p))

print('\nSample mapping:')
pprint.pprint(dict(list(mapped.items())[:50]))
if unmapped:
    print('\nUnmapped entries:')
    pprint.pprint(unmapped)
print('\nFlat-check output dir:', FLAT_ROOT)
```

-- Quick troubleshooting checklist

- If the flatten cell shows only one class: verify your dataset copy in Drive actually contains both top-level semantic folders or subclass folders; run the quick flatten-check cell and paste its output here.
- If training fails with TensorFlow import errors: create a fresh Colab runtime (Runtime -> Factory reset runtime) and run the single large cell; Colab provides TF by default.
- If TFLite conversion fails: paste the exact exception; often it's due to a custom layer or unsupported op and I can suggest fixes.

-- Cell 4: training (paste contents of `train_colab.py` here or upload the file and run)

```python
# paste the entire train_colab.py contents here and run
```

-- Cell 5: convert to float16 TFLite

```python
from convert_to_tflite import convert
convert('/content/best_model.h5', '/content/model.tflite', float16=True)
```

-- Cell 6: copy outputs back to Drive

```python
!cp /content/model.tflite /content/drive/MyDrive/model.tflite
!cp /content/labels.txt /content/drive/MyDrive/labels.txt
```

Notes:
- I pinned TensorFlow to 2.13.0 in the example for stability with cimport tensorflow as tf, time, numpy as np

print('TF version:', tf.__version__)

gpus = tf.config.list_physical_devices('GPU')
print('Physical GPUs:', gpus)

# nicely print name via nvidia-smi output from Python if available
import subprocess
try:
    print('nvidia-smi summary:')
    print(subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv'], text=True))
except Exception as e:
    print('nvidia-smi not available from Python:', e)

# optional: enable memory growth so TF doesn't pre-allocate all GPU memory
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print('Enabled memory growth on GPUs')
    except Exception as e:
        print('Could not set memory growth:', e)

# small device-placement test: run a tiny op and print device used
tf.debugging.set_log_device_placement(True)
a = tf.constant(np.random.rand(1024,1024), dtype=tf.float32)
b = tf.constant(np.random.rand(1024,1024), dtype=tf.float32)
t0 = time.time()
c = tf.matmul(a, b)
_ = c.numpy()   # force execution and copy back
print('MatMul done, elapsed:', time.time() - t0)ommon tflite tooling; adjust if you need a different version.
- If you want INT8 quantization, run the converter with a `--representative_dir` pointing to a small sample of images and test the resulting model carefully.
