# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import backend as K
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(f"Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}")
print(f"Testing data shape: {x_test.shape}, Testing labels shape: {y_test.shape}")

# Normalize the images
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode the labels
num_classes = 10
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

print(f"y_train_cat shape: {y_train_cat.shape}")
print(f"y_test_cat shape: {y_test_cat.shape}")

# Split training data into training and validation sets
x_train_aug, x_val, y_train_aug, y_val = train_test_split(
    x_train, y_train_cat, test_size=0.2, random_state=42
)

print(f"Augmented training data shape: {x_train_aug.shape}")
print(f"Validation data shape: {x_val.shape}")

# **a) Data Augmentation**

# Define data augmentation parameters
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)

# Validation data generator (no augmentation)
val_datagen = ImageDataGenerator()

batch_size = 64

# Create data generators
train_generator = train_datagen.flow(
    x_train_aug, y_train_aug,
    batch_size=batch_size
)

val_generator = val_datagen.flow(
    x_val, y_val,
    batch_size=batch_size
)

# Function to visualize augmented images
def visualize_augmented_images(generator, num_images=9):
    augmented_images, _ = next(generator)
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Visualize augmented images
visualize_augmented_images(train_generator)

# **b) Transfer Learning**

# Define input shape
input_shape = (32, 32, 3)

# Load the ResNet50 model without the top layers and with ImageNet weights
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

# Freeze the base model
base_model.trainable = False

# Build the custom model on top of the base model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(
    optimizer=optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# **Training Phase 1: Train the top layers**

epochs_initial = 10

history_initial = model.fit(
    train_generator,
    epochs=epochs_initial,
    validation_data=val_generator,
    steps_per_epoch=len(x_train_aug) // batch_size,
    validation_steps=len(x_val) // batch_size
)

# **Fine-Tuning Phase**

# Unfreeze the last 50 layers of the base model
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 50

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Recompile the model with a lower learning rate
model.compile(
    optimizer=optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

epochs_fine = 10

history_fine = model.fit(
    train_generator,
    epochs=epochs_fine,
    validation_data=val_generator,
    steps_per_epoch=len(x_train_aug) // batch_size,
    validation_steps=len(x_val) // batch_size
)

# **c) Evaluation**

# For demonstration, we'll use the fine-tuned model
best_model = model

# Ensure y_test_cat is defined
# (This should already be defined earlier in the script)
# If not, uncomment the following line:
# y_test_cat = to_categorical(y_test, num_classes)

# Evaluate on the test set
test_loss, test_acc = best_model.evaluate(x_test, y_test_cat, verbose=2)
print(f'Test accuracy: {test_acc:.4f}')
print(f'Test loss: {test_loss:.4f}')

# **Compute AUC/ROC Curves**

# Binarize the test labels for ROC AUC
y_test_bin = label_binarize(y_test.flatten(), classes=range(num_classes))

# Predict probabilities
y_pred_prob = best_model.predict(x_test, batch_size=64, verbose=1)

# Verify shapes
print(f"y_test_bin shape: {y_test_bin.shape}")    # Should be (10000, 10)
print(f"y_pred_prob shape: {y_pred_prob.shape}")  # Should be (10000, 10)

# Compute ROC AUC for each class
roc_auc = roc_auc_score(y_test_bin, y_pred_prob, average=None, multi_class='ovr')
print(f'ROC AUC scores for each class: {roc_auc}')

# Plot ROC Curve for one class as an example
class_idx = 0  # Change this to plot other classes
fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx], y_pred_prob[:, class_idx])

plt.figure()
plt.plot(fpr, tpr, label=f'Class {class_idx} ROC curve (AUC = {roc_auc[class_idx]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve for Class {class_idx}')
plt.legend(loc='lower right')
plt.show()

# **Plot Confusion Matrix**

# Predict classes
y_pred_classes = np.argmax(best_model.predict(x_test, batch_size=64, verbose=1), axis=1)
y_true = np.argmax(y_test_cat, axis=1)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Plot confusion matrix using seaborn heatmap for better visualization
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(num_classes), 
            yticklabels=range(num_classes))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# **Compare with Baseline Model**

# **Baseline Model: Simple CNN without Data Augmentation and Transfer Learning**

def create_baseline_model(input_shape, num_classes):
    baseline = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return baseline

baseline_model = create_baseline_model(input_shape, num_classes)

# Compile the baseline model
baseline_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

baseline_model.summary()

# Train the baseline model without data augmentation
epochs_baseline = 20

history_baseline = baseline_model.fit(
    x_train_aug, y_train_aug,
    epochs=epochs_baseline,
    validation_data=(x_val, y_val),
    batch_size=64,
    verbose=2
)

# Evaluate the baseline model
baseline_loss, baseline_acc = baseline_model.evaluate(x_test, y_test_cat, verbose=2)
print(f'Baseline Test accuracy: {baseline_acc:.4f}')
print(f'Baseline Test loss: {baseline_loss:.4f}')

# **Plot Training Curves for Comparison**

# Plot training and validation accuracy
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(history_initial.history['accuracy'] + history_fine.history['accuracy'], label='Transfer Learning Accuracy')
plt.plot(history_baseline.history['accuracy'], label='Baseline Accuracy')
plt.title('Training Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training and validation loss
plt.subplot(1,2,2)
plt.plot(history_initial.history['loss'] + history_fine.history['loss'], label='Transfer Learning Loss')
plt.plot(history_baseline.history['loss'], label='Baseline Loss')
plt.title('Training Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# **Summary of Results**

print(f"\nFinal Test Accuracy with Transfer Learning: {test_acc:.4f}")
print(f"Final Test Accuracy without Transfer Learning (Baseline): {baseline_acc:.4f}")
