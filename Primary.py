import os
import time
import shutil
import tempfile
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import psutil
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score


# ⏲️ Monitor system usage
def print_system_usage(phase):
    memory = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=1)
    print(f"[{phase}] CPU Usage: {cpu:.2f}% | RAM Usage: {memory.percent:.2f}% ({memory.used / 1e9:.2f} GB used)")


# 🛠️ Create temporary folder with only 'Healthy' and 'Unhealthy' classes
def create_binary_temp_dataset(original_path):
    temp_dir = tempfile.mkdtemp()
    healthy_dir = Path(temp_dir) / "Healthy"
    unhealthy_dir = Path(temp_dir) / "Unhealthy"
    healthy_dir.mkdir(parents=True, exist_ok=True)
    unhealthy_dir.mkdir(parents=True, exist_ok=True)

    for class_dir in Path(original_path).iterdir():
        if not class_dir.is_dir():
            continue
        target_dir = healthy_dir if class_dir.name.lower() == "healthy" else unhealthy_dir
        for img_file in class_dir.glob("*.*"):
            shutil.copy(img_file, target_dir / img_file.name)
    return temp_dir


# 🧪 Load and preprocess binary-class image data
def load_data_from_folder(path, target_size=(96, 96), batch_size=16):
    binary_path = create_binary_temp_dataset(path)
    datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v3.preprocess_input)
    return datagen.flow_from_directory(
        binary_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )


# 🧠 Build lightweight MobileNetV3 model
def build_model():
    base = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(96, 96, 3))
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base.input, outputs=out)
    model.compile(optimizer=Adam(0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# 📁 Set dataset paths
train_path = os.path.join("Potato", "Train")
val_path = os.path.join("Potato", "Val")
test_path = os.path.join("Potato", "Test")

# 📦 Load data
print("📦 Loading data...")
train_gen = load_data_from_folder(train_path)
val_gen = load_data_from_folder(val_path)
test_gen = load_data_from_folder(test_path)
print_system_usage("After loading data")

# ✅ Inspect a sample batch
x_batch, y_batch = next(train_gen)
print("Min pixel value:", np.min(x_batch))
print("Max pixel value:", np.max(x_batch))
print(f"🔍 Sample image shape: {x_batch[0].shape}")

# 🧠 Build and train model
print("\n🚀 Training model...")
start_time = time.time()
model = build_model()
model.fit(train_gen, validation_data=val_gen, epochs=5)
training_duration = time.time() - start_time
print(f"⏱️ Training completed in {training_duration:.2f} seconds.")
print_system_usage("After training")

# 🧪 Evaluate on test data
print("\n🎯 Test Set Results:")
test_gen.reset()
y_true = test_gen.classes
y_pred_probs = model.predict(test_gen, verbose=0)
y_pred = (y_pred_probs > 0.5).astype(int).reshape(-1)

print("Accuracy:", accuracy_score(y_true, y_pred))
print_system_usage("After prediction (Test)")
print(classification_report(y_true, y_pred, target_names=["Healthy", "Unhealthy"]))

# 📊 Confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy", "Unhealthy"])
disp.plot()

# 💾 Save model
model.save("mobilenetv3_leaf_model.h5")
print("\n💾 Model saved as 'mobilenetv3_leaf_model.h5'")

# 🛸 Convert to TFLite (for Raspberry Pi deployment)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("mobilenetv3_leaf_model.tflite", "wb") as f:
    f.write(tflite_model)
print("✨ TFLite model saved as 'mobilenetv3_leaf_model.tflite'")


# 📸 Show sample predictions
matplotlib.rcParams['font.family'] = 'Arial'

def show_predictions(generator, model, class_names=["Healthy", "Unhealthy"], num_samples=8):
    print("\n🖼️ Showing sample predictions:")
    generator.reset()
    x, y_true = next(generator)
    y_pred_probs = model.predict(x)
    y_pred = (y_pred_probs > 0.5).astype(int).reshape(-1)

    plt.figure(figsize=(15, 6))
    for i in range(min(num_samples, len(x))):
        plt.subplot(2, 4, i + 1)
        # Convert from [-1, 1] to [0, 1] and clip
        img = np.clip((x[i] + 1) / 2.0, 0, 1)
        plt.imshow(img)
        plt.axis('off')

        true_label = class_names[int(y_true[i])]
        pred_label = class_names[int(y_pred[i])]

        # Color prediction in red if wrong
        color = 'green' if true_label == pred_label else 'red'
        plt.title(f"Pred: {pred_label}\nTrue: {true_label}", color=color)

    plt.tight_layout()
    plt.show()

# 🖼️ Call the function to display
show_predictions(test_gen, model)