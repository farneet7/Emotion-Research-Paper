import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, LSTM, Bidirectional, 
                                     Conv2D, MaxPooling2D, Flatten, concatenate, MultiHeadAttention)

from tensorflow.keras.layers import Dropout

# CNN for Melspectrogram
input_mel = Input(shape=(mel_height, mel_width, 1))  # Adjust shape accordingly
x = Conv2D(256, (3, 3), activation='relu')(input_mel)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(256, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)  # Dropout layer added here
cnn_output = Dense(128, activation='relu')(x)

# Bidirectional LSTM for sequential features
input_seq = Input(shape=(timesteps, features))  # Adjust shape accordingly
y = Bidirectional(LSTM(256, return_sequences=True))(input_seq)
y = MultiHeadAttention(num_heads=2, key_dim=128)(y, y)  # Attention layer added here
y = Dropout(0.5)(y)
y = Bidirectional(LSTM(256, return_sequences=True))(y)
y = Bidirectional(LSTM(128))(y)
y = Dropout(0.5)(y)  # Dropout layer added here
lstm_output = Dense(128, activation='relu')(y)

# Combining both models
combined = concatenate([cnn_output, lstm_output])
z = Dense(128, activation='relu')(combined)
z = Dropout(0.5)(z)  # Dropout layer added here
z = Dense(num_classes, activation='softmax')(z)  # Adjust num_classes for your data

model = Model(inputs=[input_mel, input_seq], outputs=z)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


135/135 [==============================] - 3s 19ms/step - loss: 0.9640 - accuracy: 0.6392
Test Accuracy: 63.91608119010925%


                precision    recall  f1-score   support

           0       0.73      0.78      0.75       688
           2       0.61      0.54      0.57       688
           3       0.64      0.51      0.57       688
           4       0.59      0.54      0.56       686
           5       0.58      0.63      0.60       612
           6       0.63      0.75      0.68       692
           7       0.77      0.88      0.82       236

    accuracy                           0.64      4290
   macro avg       0.65      0.66      0.65      4290
weighted avg       0.64      0.64      0.64      4290
