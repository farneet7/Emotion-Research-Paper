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



# 4th November (4 pm) - Achieved maximum 82.92% accuracy

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()

model.add(
        Conv2D(
            filters=512,
            kernel_size=(5,5),
            input_shape=(48, 48, 1),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_1'
        )
    )
model.add(BatchNormalization(name='batchnorm_1'))
model.add(
        Conv2D(
            filters=256,
            kernel_size=(5,5),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_2'
        )
    )
model.add(BatchNormalization(name='batchnorm_2'))
    
model.add(MaxPooling2D(pool_size=(2,2), name='maxpool2d_1'))
model.add(Dropout(0.25, name='dropout_1'))

model.add(
        Conv2D(
            filters=128,
            kernel_size=(3,3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_3'
        )
    )
model.add(BatchNormalization(name='batchnorm_3'))
model.add(
        Conv2D(
            filters=128,
            kernel_size=(3,3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_4'
        )
    )
model.add(BatchNormalization(name='batchnorm_4'))
    
model.add(MaxPooling2D(pool_size=(2,2), name='maxpool2d_2'))
model.add(Dropout(0.25, name='dropout_2'))

model.add(
        Conv2D(
            filters=256,
            kernel_size=(3,3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_5'
        )
    )
model.add(BatchNormalization(name='batchnorm_5'))
model.add(
        Conv2D(
            filters=512,
            kernel_size=(3,3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_6'
        )
    )
model.add(BatchNormalization(name='batchnorm_6'))
    
model.add(MaxPooling2D(pool_size=(2,2), name='maxpool2d_3'))
model.add(Dropout(0.25, name='dropout_3'))

model.add(Flatten(name='flatten'))
        
model.add(
        Dense(
            256,
            activation='elu',
            kernel_initializer='he_normal',
            name='dense_1'
        )
    )
model.add(BatchNormalization(name='batchnorm_7'))
    
model.add(Dropout(0.25, name='dropout_4'))
    
model.add(
        Dense(
            7,
            activation='softmax',
            name='out_layer'
        )
    )
    
model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
model.summary()

Epoch 47/60
192/192 [==============================] - 52s 272ms/step - loss: 0.3441 - accuracy: 0.8791 - val_loss: 0.5502 - val_accuracy: 0.8289 - lr: 5.0000e-04
Epoch 48/60
192/192 [==============================] - 52s 271ms/step - loss: 0.3358 - accuracy: 0.8782 - val_loss: 0.5544 - val_accuracy: 0.8292 - lr: 5.0000e-04
Epoch 49/60
192/192 [==============================] - 53s 274ms/step - loss: 0.3273 - accuracy: 0.8835 - val_loss: 0.5967 - val_accuracy: 0.8246 - lr: 5.0000e-04
Epoch 50/60
192/192 [==============================] - 53s 274ms/step - loss: 0.3046 - accuracy: 0.8894 - val_loss: 0.6177 - val_accuracy: 0.8113 - lr: 5.0000e-04
Epoch 51/60
192/192 [==============================] - 53s 276ms/step - loss: 0.3018 - accuracy: 0.8927 - val_loss: 0.6137 - val_accuracy: 0.8123 - lr: 5.0000e-04
Epoch 52/60
192/192 [==============================] - 54s 279ms/step - loss: 0.2980 - accuracy: 0.8923 - val_loss: 0.6151 - val_accuracy: 0.8227 - lr: 5.0000e-04
Epoch 53/60
192/192 [==============================] - 54s 279ms/step - loss: 0.3014 - accuracy: 0.8937 - val_loss: 0.6431 - val_accuracy: 0.8207 - lr: 5.0000e-04
Epoch 54/60
192/192 [==============================] - 53s 274ms/step - loss: 0.2964 - accuracy: 0.8920 - val_loss: 0.7213 - val_accuracy: 0.7904 - lr: 5.0000e-04
Epoch 55/60
192/192 [==============================] - ETA: 0s - loss: 0.2894 - accuracy: 0.8942
Epoch 55: ReduceLROnPlateau reducing learning rate to 0.0002500000118743628.
192/192 [==============================] - 53s 275ms/step - loss: 0.2894 - accuracy: 0.8942 - val_loss: 0.6689 - val_accuracy: 0.8080 - lr: 5.0000e-04
Epoch 56/60
192/192 [==============================] - 52s 271ms/step - loss: 0.2601 - accuracy: 0.9066 - val_loss: 0.6149 - val_accuracy: 0.8155 - lr: 2.5000e-04
Epoch 57/60
192/192 [==============================] - 52s 272ms/step - loss: 0.2507 - accuracy: 0.9102 - val_loss: 0.6446 - val_accuracy: 0.8162 - lr: 2.5000e-04
Epoch 58/60
192/192 [==============================] - 53s 275ms/step - loss: 0.2545 - accuracy: 0.9082 - val_loss: 0.6354 - val_accuracy: 0.8116 - lr: 2.5000e-04
Epoch 59/60
192/192 [==============================] - ETA: 0s - loss: 0.2365 - accuracy: 0.9157Restoring model weights from the end of the best epoch: 48.
192/192 [==============================] - 52s 272ms/step - loss: 0.2365 - accuracy: 0.9157 - val_loss: 0.6424 - val_accuracy: 0.8158 - lr: 2.5000e-04
Epoch 59: early stopping
