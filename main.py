import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import splitfolders
import os

if not os.path.exists('flower_data'):
    splitfolders.ratio('flowers',output='flower_data',seed=888,ratio=(0.8,0.2))

train_format = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_format = ImageDataGenerator(
    rescale=1./255,
)
train_data = train_format.flow_from_directory(
    './flower_data/train',
    target_size= (150,150),
    class_mode='categorical',
    batch_size= 16
)
val_data = val_format.flow_from_directory(
    './flower_data/val',
    target_size= (150,150),
    class_mode='categorical',
    batch_size= 16
)
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3) ,activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(512, activation = 'relu'),
    Dropout(0.5),
    Dense(5,activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy','precision','recall'])
result = model.fit(train_data,epochs=9,validation_data=val_data)
model.save('flower.h5')

fig, axes = plt.subplots(2,2, figsize=(12   ,4))

axes[0,0].plot(result.history['loss'], label = 'Training loss')
axes[0,0].plot(result.history['val_loss'], label = 'Validation loss')
axes[0,0].set_title('Loss')
axes[0,0].legend()

axes[0,1].plot(result.history['accuracy'], label = 'Training accuracy')
axes[0,1].plot(result.history['val_accuracy'], label = 'Validation accuracy')
axes[0,1].set_title('Accuracy')
axes[0,1].legend()

axes[1,0].plot(result.history['precision'], label = 'Training precision')
axes[1,0].plot(result.history['val_precision'], label = 'Validation precision')
axes[1,0].set_title('PRecision')
axes[1,0].legend()

axes[1,1].plot(result.history['recall'], label = 'Training recall')
axes[1,1].plot(result.history['val_recall'], label = 'Validation recall')
axes[1,1].set_title('Recall')
axes[1,1].legend()
plt.show()
