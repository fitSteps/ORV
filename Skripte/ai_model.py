import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from PIL import Image
from tensorflow.keras.utils import Sequence

# Define custom data loader
class CustomDataLoader(Sequence):
    def __init__(self, image_paths, labels, batch_size):
        self.image_paths = np.array(image_paths)
        self.labels = np.array(labels)
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, len(self.image_paths))
        batch_image_paths = self.image_paths[start:end]
        batch_labels = self.labels[start:end]
        
        batch_images = [np.array(Image.open(path).resize((224, 224))) / 255.0 for path in batch_image_paths]
        return np.array(batch_images), np.array(batch_labels)

# Load the VGG16 model without the top layer (classifier)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False  # Freeze all the layers

# Add custom layers on top of VGG16 for binary classification
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)  # Binary output

# Complete model setup
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Assuming folders 'me' and 'others' are directly under 'data/'
me_dir = 'Me\\frames'
others_dir = 'RandomPeople'
me_images = [os.path.join(me_dir, img) for img in os.listdir(me_dir)]
others_images = [os.path.join(others_dir, img) for img in os.listdir(others_dir)]

# Labels: 1 for 'me', 0 for 'others'
image_paths = me_images + others_images
labels = [1] * len(me_images) + [0] * len(others_images)

# Shuffle data (important for training)
indices = np.arange(len(image_paths))
np.random.shuffle(indices)
image_paths = np.array(image_paths)[indices]
labels = np.array(labels)[indices]

# Data loader setup
batch_size = 32
data_loader = CustomDataLoader(image_paths, labels, batch_size)

# Train the model
model.fit(data_loader, epochs=10)

# Save the trained model
model.save('face_verification_model.h5')
