import os
import time
import shutil
import tempfile
import collections
import numpy as np
import psutil
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight


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
def load_data_from_folder(path, target_size=(128, 128), batch_size=8, augment=False):
    binary_path = create_binary_temp_dataset(path)
    if augment:
        datagen = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.mobilenet_v3.preprocess_input,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.3,
            horizontal_flip=True,
            brightness_range=[0.8, 1.4],
            fill_mode='nearest',
        )
    else:
        datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v3.preprocess_input)

    return datagen.flow_from_directory(
        binary_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )


# 🧠 Build MobileNetV3 model
def build_model():
    base = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    base.trainable = False  # Freeze the base model to prevent training during initial training phase

    x = base.output
    x = GlobalMaxPooling2D()(x)  # Apply GlobalMaxPooling2D to the feature map from MobileNetV3
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base.input, outputs=out)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model



# 📁 Dataset paths
train_path = os.path.join("Potato", "Train")
val_path = os.path.join("Potato", "Val")
test_path = os.path.join("Potato", "Test")

# 📦 Load datasets
print("📦 Loading data...")
train_gen = load_data_from_folder(train_path, augment=True)
val_gen = load_data_from_folder(val_path)
test_gen = load_data_from_folder(test_path)


def show_class_distribution(generator, name):
    labels = generator.classes
    counter = collections.Counter(labels)
    label_map = {v: k for k, v in generator.class_indices.items()}
    print(f"\n📊 Class distribution in {name}:")
    for class_id, count in counter.items():
        print(f" - {label_map[class_id]}: {count} images")


show_class_distribution(train_gen, "Train Set")
show_class_distribution(val_gen, "Validation Set")
show_class_distribution(test_gen, "Test Set")

print_system_usage("After loading data")

# 📉 Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# ⚖️ Compute class weights
classes = np.array([0, 1])
weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_gen.classes)
class_weight = dict(zip(classes, weights))


# 🚀 Train the model with k-fold cross-validation
def cross_validate(model, train_gen, val_gen, n_splits=5):
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
    fold_no = 1
    for train_idx, val_idx in kfold.split(train_gen.filenames, train_gen.classes):
        print(f"\n📊 Training fold {fold_no}/{n_splits}...")

        train_gen_subset = train_gen.__getitem__(train_idx)  # Use data for training
        val_gen_subset = val_gen.__getitem__(val_idx)  # Use data for validation

        history = model.fit(
            train_gen_subset,
            validation_data=val_gen_subset,
            epochs=20,
            callbacks=[early_stop, reduce_lr, checkpoint],
            class_weight=class_weight
        )
        fold_no += 1
    return model


# 📦 Build and Train the Model
print("\n🚀 Training model...")
start_time = time.time()
model = build_model()

model = cross_validate(model, train_gen, val_gen)
training_duration = time.time() - start_time
print(f"⏱️ Training completed in {training_duration:.2f} seconds.")
print_system_usage("After training")

# 📉 Accuracy and loss plots
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('📊 Accuracy vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
plt.title('📉 Loss vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 🧪 Evaluate on test data
print("\n🎯 Test Set Results:")
test_gen.reset()
y_true = test_gen.classes

print("\n🕒 Starting test-time inference with Test-Time Augmentation...")
test_start_time = time.time()
y_pred_probs = model.predict(test_gen, verbose=0)
test_duration = time.time() - test_start_time
print(f"🕒 Test-time inference completed in {test_duration:.2f} seconds.")

y_pred = (y_pred_probs > 0.5).astype(int).reshape(-1)

print("Accuracy:", accuracy_score(y_true, y_pred))
print_system_usage("After prediction (Test)")
print(classification_report(y_true, y_pred, target_names=["Healthy", "Unhealthy"]))

# 📊 Confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy", "Unhealthy"])
disp.plot()
plt.show()

# 💾 Save models
model.save("mobilenetv3_leaf_model.h5")
print("\n💾 Model saved as 'mobilenetv3_leaf_model.h5'")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("mobilenetv3_leaf_model.tflite", "wb") as f:
    f.write(tflite_model)
print("✨ TFLite model saved as 'mobilenetv3_leaf_model.tflite'")

# 📸 Sample predictions
matplotlib.rcParams['font.family'] = 'Arial'


def show_predictions(generator, model, class_names=["Healthy", "Unhealthy"], num_samples=8):
    print("\n🗈️ Showing sample predictions:")
    generator.reset()
    x, y_true = next(generator)
    y_pred_probs = model.predict(x)
    y_pred = (y_pred_probs > 0.5).astype(int).reshape(-1)

    plt.figure(figsize=(15, 6))
    for i in range(min(num_samples, len(x))):
        plt.subplot(2, 4, i + 1)
        img = (x[i] + 1) / 2.0  # Inverse of MobileNetV3 preprocess
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.axis('off')

        true_label = class_names[int(y_true[i])]
        pred_label = class_names[int(y_pred[i])]
        color = 'green' if true_label == pred_label else 'red'
        plt.title(f"Pred: {pred_label}\nTrue: {true_label}", color=color)

    plt.tight_layout()
    plt.show()


# ✅ Show some predictions
show_predictions(test_gen, model)
