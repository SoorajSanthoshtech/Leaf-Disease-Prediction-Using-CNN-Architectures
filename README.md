# Mango Leaf Disease Classification

This project classifies mango leaf diseases using transfer learning with various deep learning architectures (VGG19, DenseNet121, InceptionV3, and Xception). It utilizes a dataset of mango leaf images and trains CNN models to predict the disease class. The code is structured for use in Google Colab, leveraging the Kaggle API to download the dataset.

## Project Structure

- **Dataset**: Mango leaf disease images, organized in folders by class.
- **Model Architectures**: VGG19, DenseNet121, InceptionV3, and Xception pre-trained on ImageNet, used as base models.
- **Training and Evaluation**: The models are trained on augmented data with accuracy metrics tracked and compared.

## Requirements

- Python 3.x
- TensorFlow and Keras
- Kaggle API
- Matplotlib
- Numpy
- ImageDataGenerator from Keras for data preprocessing and augmentation

## Code Walkthrough

### 1. Dataset Download

The code downloads the mango leaf disease dataset directly from Kaggle using the Kaggle API:

```python
!kaggle datasets download -d aryashah2k/mango-leaf-disease-dataset -p /content/mango_leaf_disease --unzip
```

Make sure to set up your Kaggle API key in Colab by following the [Kaggle API instructions](https://www.kaggle.com/docs/api).

### 2. Import Libraries

The required libraries are imported, including TensorFlow and Keras models for transfer learning:

```python
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19, DenseNet121, InceptionV3, Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
```

### 3. Define Data Generators

To augment the dataset and improve model generalization, `ImageDataGenerator` is used:

```python
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)
```

This generator applies several transformations, including:
- **Rescaling**: Normalizes pixel values to a range of 0-1.
- **Augmentation**: Random rotations, shifts, zooms, and flips to increase dataset variety.

The `get_data_generators()` function defines the training, validation, and test generators with an 80/20 train-validation split.

### 4. Model Building

A function, `build_model()`, creates and compiles a model by adding custom layers to a pre-trained base model:

```python
def build_model(base_model, train_generator):
    base_model.trainable = False  # Freezing base model layers

    # Custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(train_generator.num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
```

Each CNN architecture (e.g., VGG19, DenseNet121) is passed as the `base_model`. A dropout layer helps prevent overfitting, and a dense layer with softmax activation classifies images into disease categories.

### 5. Model Training and Evaluation

The `train_and_evaluate_model()` function trains the model and evaluates it on the test set, with early stopping and learning rate reduction:

```python
def train_and_evaluate_model(model, train_generator, val_generator, test_generator, epochs=10):
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=[early_stopping, reduce_lr]
    )
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_accuracy:.2f}")
```

- **Callbacks**:
  - `EarlyStopping`: Stops training if the validation accuracy does not improve for 5 epochs.
  - `ReduceLROnPlateau`: Reduces the learning rate if validation loss does not improve, helping the model fine-tune better.

A sample prediction is performed after testing to check the model’s output against the actual class.

### 6. Model Training Loop

The code iterates over the four pre-trained models, training and evaluating each one:

```python
models = {
    "VGG19": (VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3)), (224, 224)),
    "DenseNet121": (DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3)), (224, 224)),
    "InceptionV3": (InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3)), (299, 299)),
    "Xception": (Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3)), (299, 299))
}

histories = {}
test_accuracies = {}

for model_name, (base_model, target_size) in models.items():
    print(f"Training {model_name} model...")
    train_generator, val_generator, test_generator = get_data_generators(target_size)
    model = build_model(base_model, train_generator)
    histories[model_name], test_accuracies[model_name] = train_and_evaluate_model(model, train_generator, val_generator, test_generator)
```

Each model’s training and validation accuracy is stored in `histories`, and test accuracy is saved in `test_accuracies`.

### 7. Results Visualization

After training all models, training, validation, and test accuracies are plotted to compare performances:

```python
plt.figure(figsize=(12, 8))

for model_name, history in histories.items():
    plt.plot(history.history['accuracy'], label=f'{model_name} Training Accuracy')
    plt.plot(history.history['val_accuracy'], label=f'{model_name} Validation Accuracy')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Comparison Across Models')
plt.legend(loc='upper left')
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(test_accuracies.keys(), test_accuracies.values(), color='skyblue')
plt.xlabel('Model')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy Comparison Across Models')
plt.show()
```

- **Accuracy Plot**: Shows training and validation accuracy for each model.
- **Bar Plot**: Compares the final test accuracy across all models.

## Results and Insights

The performance comparison allows selecting the best model based on test accuracy. Fine-tuning and hyperparameter adjustments can be explored to further improve results.

## Conclusion

This project demonstrates a comparative analysis of different CNN architectures for mango leaf disease classification. By using data augmentation, transfer learning, and performance comparison, it provides insights into model effectiveness for plant disease classification tasks.
