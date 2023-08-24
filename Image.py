from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

model = Sequential()
# < ----------------------------- PART 1 --------------------------------------------->
model.add(Conv2D(filters=512, kernel_size=(3,3), activation='elu', padding='same', kernel_initializer='he_normal', input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(Conv2D(filters=256, kernel_size=(3,3), activation='elu', padding='same', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2))) # Second max pooling layer
model.add(Dropout(0.25))

# < ----------------------------- PART 2 --------------------------------------------->

model.add(Conv2D(filters=512, kernel_size=(3,3), activation='elu', padding='same', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Conv2D(filters=256, kernel_size=(3,3), activation='elu', padding='same', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2))) # Second max pooling layer
model.add(Dropout(0.25))

# < ----------------------------- PART 3 --------------------------------------------->

model.add(Conv2D(filters=512, kernel_size=(3,3), activation='elu', padding='same', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Conv2D(filters=256, kernel_size=(3,3), activation='elu', padding='same', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2))) # First max pooling layer
model.add(Dropout(0.25))


# < ----------------------------- PART 4 --------------------------------------------->

model.add(Conv2D(filters=512, kernel_size=(3,3), activation='elu', padding='same', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Conv2D(filters=256, kernel_size=(3,3), activation='elu', padding='same', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.25))


# < ----------------------------- PART 5 --------------------------------------------->

model.add(Conv2D(filters=512, kernel_size=(3,3), activation='elu', padding='same', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Conv2D(filters=256, kernel_size=(3,3), activation='elu', padding='same', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

# < ----------------------------- PART 6 --------------------------------------------->

model.add(Conv2D(filters=512, kernel_size=(3,3), activation='elu', padding='same', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Conv2D(filters=256, kernel_size=(3,3), activation='elu', padding='same', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2))) # Second max pooling layer
model.add(Dropout(0.25))


# < ----------------------------- END --------------------------------------------->

model.add(Flatten(name='flatten'))
model.add(Dense(256, activation='elu', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


LAST EPOCHS
Epoch 64/100
449/449 [==============================] - 55s 123ms/step - loss: 0.7537 - accuracy: 0.7202 - val_loss: 0.8919 - val_accuracy: 0.6803 - lr: 1.0000e-05
Epoch 65/100
449/449 [==============================] - 55s 122ms/step - loss: 0.7566 - accuracy: 0.7197 - val_loss: 0.8919 - val_accuracy: 0.6810 - lr: 1.0000e-05
Epoch 66/100
449/449 [==============================] - ETA: 0s - loss: 0.7539 - accuracy: 0.7206Restoring model weights from the end of the best epoch: 56.
449/449 [==============================] - 56s 125ms/step - loss: 0.7539 - accuracy: 0.7206 - val_loss: 0.8928 - val_accuracy: 0.6814 - lr: 1.0000e-05
Epoch 66: early stopping
