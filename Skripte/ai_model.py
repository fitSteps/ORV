import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import LearningRateScheduler
from functions import augm_horizontal_flip, augm_adjust_brightness, augm_random_crop, augm_adjust_contrast, video_to_images
from PIL import ImageFile
import argparse
import subprocess


# Setup command line argument parsing
parser = argparse.ArgumentParser(description='Process the MQTT message for the AI model.')
parser.add_argument('mqtt_message', type=str, help='MQTT message payload')
args = parser.parse_args()


ImageFile.LOAD_TRUNCATED_IMAGES = True

class CustomDataLoader(Sequence):
    def __init__(self, image_paths, labels, batch_size, augment=False):
        self.image_paths = np.array(image_paths)
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.augment = augment

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, len(self.image_paths))
        batch_image_paths = self.image_paths[start:end]
        batch_labels = self.labels[start:end]
        
        batch_images = []
        for path in batch_image_paths:
            image = np.array(Image.open(path).resize((224, 224)))  # Resize the image first

            if self.augment:
                image = augm_horizontal_flip(image)
                image = augm_adjust_brightness(image, 25)
                image = augm_adjust_contrast(image, 0.15)
                try:
                    image = augm_random_crop(image, 1.05)  # Ensure image can be cropped
                except ValueError:
                    pass  # Skip cropping if the random size is invalid
            image = image / 255.0  # Normalize to [0, 1]
            batch_images.append(image)

        return np.array(batch_images), np.array(batch_labels)

# Load the VGG16 model without the top layer (classifier)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Set the last five convolutional layers to be trainable
for layer in base_model.layers[:-4]:
    layer.trainable = False
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Add custom layers on top of VGG16 for binary classification
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)  # Binary output

# Complete model setup
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(lr=0.000001), loss='binary_crossentropy', metrics=['accuracy'])


me_dir = '/frames'
video_path = f'/app/videos/{args.mqtt_message}.mp4'
video_to_images(video_path, me_dir, frame_rate=1,max_frames=200000)

others_dir = '/usr/src/orv/scraped_images'

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
train_data_loader = CustomDataLoader(image_paths, labels, batch_size, augment=True)

# Define the learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 7:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_scheduler = LearningRateScheduler(scheduler)

# Train the model with the learning rate scheduler
model.fit(train_data_loader, epochs=15, callbacks=[lr_scheduler])
output_folder='/ai_models'
# Save the trained model
if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")
model.save('/ai_models/{args.mqtt_message}.h5')

