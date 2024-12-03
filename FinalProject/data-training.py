import tensorflow as tf
from tensorflow.keras import layers, models, Input
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Dataset paths
dataset_path = "/Users/med/Library/CloudStorage/OneDrive-PrincetonUniversity/Fall 2024 - Princeton University/MAE 345 - Introduction to Robotics/F2024-main/Dataset"

# Image size
image_size = (300, 300)

# Load datasets
train_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    labels='inferred',
    label_mode='int',
    image_size=image_size,
    batch_size=16,
    validation_split=0.2,
    subset="training",
    seed=123
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    labels='inferred',
    label_mode='int',
    image_size=image_size,
    batch_size=16,
    validation_split=0.2,
    subset="validation",
    seed=123
)

# Normalize datasets
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

# Enhanced data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2)
])

# Apply data augmentation to the training dataset
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))

# Compute class weights
train_labels = np.concatenate([y.numpy() for x, y in train_dataset])
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# Prefetch datasets for performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

# Define the enhanced model architecture
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(300, 300, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # Freeze the base model

model = tf.keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=15,
    class_weight=class_weights,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="obstacle_model_300x300.keras",
            monitor='val_loss',
            save_best_only=True
        )
    ]
)

# Save the trained model
model.save("obstacle_model_300x300.keras")
print("Model training complete and saved as obstacle_model_300x300.keras")
