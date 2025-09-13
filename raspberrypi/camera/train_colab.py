"""
train_colab.py

Small, self-contained training script intended to be pasted into a Google Colab Python cell
or executed as a script in the Colab VM. Uses Keras ImageDataGenerator for simplicity and MobileNetV2
transfer learning. Saves best model to `best_model.h5` and a final `model.tflite` via a helper conversion.

Usage in Colab (high level):
- Mount Drive
- Copy dataset to /content/Dataset or use Drive path
- Run the cells that set DATASET_DIR and then run main()

This script is intentionally simple and focused on reproducibility.
"""
import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


def build_model(num_classes, input_shape=(224,224,3), dropout=0.3):
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=outputs)
    # freeze base initially
    for layer in base.layers:
        layer.trainable = False
    return model


def get_data_generators(dataset_dir, image_size=(224,224), batch_size=32, val_split=0.2):
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.1,
                                       height_shift_range=0.1, shear_range=0.1, zoom_range=0.15,
                                       horizontal_flip=True, fill_mode='nearest', validation_split=val_split)

    train_gen = train_datagen.flow_from_directory(dataset_dir, target_size=image_size, batch_size=batch_size,
                                                  class_mode='categorical', subset='training')
    val_gen = train_datagen.flow_from_directory(dataset_dir, target_size=image_size, batch_size=batch_size,
                                                class_mode='categorical', subset='validation')
    return train_gen, val_gen


def train(dataset_dir, output_dir='/content', epochs=12, batch_size=32, lr=1e-3):
    train_gen, val_gen = get_data_generators(dataset_dir, batch_size=batch_size)
    num_classes = train_gen.num_classes
    model = build_model(num_classes)
    model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])

    os.makedirs(output_dir, exist_ok=True)
    ckpt = ModelCheckpoint(os.path.join(output_dir, 'best_model.h5'), monitor='val_accuracy', save_best_only=True, mode='max')
    rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    early = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

    steps_per_epoch = max(1, train_gen.samples // batch_size)
    val_steps = max(1, val_gen.samples // batch_size)

    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, steps_per_epoch=steps_per_epoch,
                        validation_steps=val_steps, callbacks=[ckpt, rlrop, early])

    # Unfreeze and fine-tune
    for layer in model.layers:
        layer.trainable = True
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    ft_history = model.fit(train_gen, validation_data=val_gen, epochs=6, steps_per_epoch=steps_per_epoch,
                          validation_steps=val_steps, callbacks=[rlrop, early])

    model.save(os.path.join(output_dir, 'final_model.h5'))
    return os.path.join(output_dir, 'best_model.h5'), os.path.join(output_dir, 'final_model.h5')


if __name__ == '__main__':
    # Example usage when running directly in Colab: set DATASET_DIR env var or edit below
    DATASET_DIR = os.environ.get('DATASET_DIR', '/content/Dataset')
    print('Dataset dir:', DATASET_DIR)
    best, final = train(DATASET_DIR)
    print('Saved models:', best, final)
