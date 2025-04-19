import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Lambda
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# 🔧 Set paths
train_dir = os.path.join("Potato", "Train")
val_dir = os.path.join("Potato", "Val")
test_dir = os.path.join("Potato", "Test")

# 📦 Load data
print("📦 Loading data...")
img_size = 224
batch_size = 32

# Data augmentation for training (only augment healthy class if needed)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary'
)

val_gen = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary'
)

test_gen = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Print class distribution
def print_class_distribution(generator, name):
    labels = generator.classes
    healthy = np.sum(labels == 0)
    unhealthy = np.sum(labels == 1)
    print(f"\n📊 Class distribution in {name} Set:\n - Healthy: {healthy} images\n - Unhealthy: {unhealthy} images")

print_class_distribution(train_gen, "Train")
print_class_distribution(val_gen, "Validation")
print_class_distribution(test_gen, "Test")

# ⚙ Build model using EfficientNet-Lite0 from TF Hub
def build_model():
    input_tensor = Input(shape=(img_size, img_size, 3))

    # Load TF Hub model
    hub_layer = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2",
        trainable=False
    )

    # Fix KerasTensor issue using Lambda
    def apply_hub(x):
        return hub_layer(x)

    x = Lambda(apply_hub)(input_tensor)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_tensor, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# 🧠 Train the model
print("\n🚀 Training model...")
model = build_model()

# Compute class weights (to handle imbalance)
class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(train_gen.classes),
                                     y=train_gen.classes)
class_weights_dict = dict(enumerate(class_weights))

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    class_weight=class_weights_dict
)

# 📈 Plot accuracy and loss
def plot_metrics(history):
    plt.figure(figsize=(12, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_metrics(history)

# 🎯 Evaluate on test set
print("\n🎯 Evaluating model on test data...")
test_gen.reset()
y_true = test_gen.classes
y_pred_probs = model.predict(test_gen, verbose=0)
y_pred = (y_pred_probs > 0.5).astype("int32").flatten()

# 🧾 Classification Report
print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Healthy', 'Unhealthy']))

# 🔍 Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("\n🧩 Confusion Matrix:")
print(cm)
