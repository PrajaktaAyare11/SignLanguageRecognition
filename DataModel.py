import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image size & batch settings
img_size = (200, 200)  
batch_size = 16  

# Dataset path
train_dir = r"D:\NLP Mini Project\DATAPICS"

#  Improved Image Augmentation (No Rotation or Flipping)
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    validation_split=0.2,  # 20% validation split
    brightness_range=[0.8, 1.2],  # Adjust lighting
    width_shift_range=0.1,  # Allow small left/right shifts
    height_shift_range=0.1,  # Allow small up/down shifts
    zoom_range=0.1  # Allow slight zoom-in/out
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    color_mode="grayscale",  # Since your dataset is grayscale
    class_mode="categorical",
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="categorical",
    subset='validation'
)

# Lightweight CNN Model for Faster Training
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(200, 200, 1)),  # 1 channel (grayscale)
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(27, activation='softmax')  # 27 classes (A-Z + space)
])

# Show Model Summary
model.summary()

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#  Train Model with Early Stopping to Prevent Overfitting
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.fit(train_generator, validation_data=val_generator, epochs=10, callbacks=[early_stopping])  

# Save Model
model.save(r"D:\NLP Mini Project\Model\keras_model12.h5")
