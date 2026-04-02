import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# -------------------------
# Paths
# -------------------------
train_dir = "dataset/train"  # FER2013 train folder
test_dir = "dataset/test"    # FER2013 test folder

img_size = 48   # Grayscale 48x48
batch_size = 32
epochs = 50     # You can increase

# -------------------------
# Data Generators
# -------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size,img_size),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size,img_size),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_size,img_size),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

# -------------------------
# Build MobileNetV2 for grayscale
# -------------------------
input_shape = (img_size,img_size,1)

base_model = tf.keras.applications.MobileNetV2(
    weights=None,  # can't use ImageNet because it's 3 channels
    include_top=False,
    input_shape=input_shape
)

base_model.trainable = True

model = Sequential([
    Input(shape=input_shape),
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation="relu"),
    Dropout(0.3),
    Dense(train_generator.num_classes, activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------
# Class weights
# -------------------------
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))

# -------------------------
# Train
# -------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    class_weight=class_weights
)

# -------------------------
# Save model and labels
# -------------------------
model.save("fer_MobileNetV2_gray.keras")

labels = list(train_generator.class_indices.keys())
with open("labels.json","w") as f:
    json.dump(labels,f)

print(f"✅ Model saved as 'fer_MobileNetV2_gray.keras'")
print(f"✅ Labels saved as 'labels.json'")

# -------------------------
# Evaluate
# -------------------------
loss, acc = model.evaluate(test_generator)
print(f"🎯 Test Accuracy: {acc*100:.2f}%")

# -------------------------
# Plot Accuracy / Loss
# -------------------------
plt.figure(figsize=(8,6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(8,6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
