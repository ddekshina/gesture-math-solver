import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

def create_model(num_classes=14):
    """Create a CNN model for gesture recognition"""
    model = Sequential([
        # First convolution block
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Second convolution block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Fully connected layers
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_training_history(history):
    """Plot training and validation metrics"""
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def main():
    # Setup data paths
    dataset_path = "dataset"
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset directory '{dataset_path}' not found.")
        print("Please run collect_gestures.py to collect data first.")
        return
    
    # Check for classes
    classes = sorted(os.listdir(dataset_path))
    valid_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', 'x', 'divide']
    found_classes = [c for c in classes if c in valid_classes]
    
    if len(found_classes) < 2:
        print(f"Error: Not enough classes found in '{dataset_path}'.")
        print(f"Found classes: {found_classes}")
        print("Please collect data for at least 2 classes.")
        return
    
    print(f"Found {len(found_classes)} classes: {found_classes}")
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1,
        fill_mode='constant',
        cval=1  # White background
    )
    
    # Setup data generators
    batch_size = 32
    img_size = 28
    color_mode = 'grayscale'
    
    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(img_size, img_size),
        color_mode=color_mode,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    validation_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(img_size, img_size),
        color_mode=color_mode,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # Get class indices and create label mapping
    class_indices = train_generator.class_indices
    label_map = {v: k for k, v in class_indices.items()}
    
    print(f"Class indices: {class_indices}")
    print(f"Label map: {label_map}")
    
    # Save the label map for inference
    np.save('label_map.npy', label_map)
    
    # Create model
    num_classes = len(found_classes)
    model = create_model(num_classes)
    
    # Print model summary
    model.summary()
    
    # Setup callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
    ]
    
    # Train the model
    epochs = 50
    print(f"Training model for up to {epochs} epochs...")
    
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=callbacks
    )
    
    # Save the model
    model.save('gesture_model.h5')
    print("Model saved as 'gesture_model.h5'")
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(validation_generator)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()