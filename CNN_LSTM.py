# <------------------------------------------------ TRIAL 1 ------------------------------------------------------------>
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

TEST RESULTS
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





# <------------------------------------------------ TRIAL 2 ------------------------------------------------------------>


# CNN for Melspectrogram
input_mel = Input(shape=(mel_height, mel_width, 1))  # Adjust shape accordingly
x = Conv2D(512, (3, 3), activation='relu')(input_mel)
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
attention_output = MultiHeadAttention(num_heads=2, key_dim=128)(y, y)
# Adding residual connection
y = add([y, attention_output])
y = Dropout(0.5)(y)

y = Bidirectional(LSTM(256, return_sequences=True))(y)
attention_output = MultiHeadAttention(num_heads=2, key_dim=128)(y, y)
# Adding residual connection
y = add([y, attention_output])
y = Dropout(0.5)(y)

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

TEST RESULTS
135/135 [==============================] - 5s 30ms/step - loss: 0.9829 - accuracy: 0.6413
Test Accuracy: 64.1258716583252%

CLASSIFICATION REPORT
                precision   recall  f1-score   support

           0       0.76      0.77      0.76       688
           2       0.61      0.54      0.57       688
           3       0.62      0.55      0.58       688
           4       0.60      0.57      0.58       686
           5       0.57      0.62      0.59       612
           6       0.62      0.73      0.67       692
           7       0.85      0.83      0.84       236

    accuracy                           0.64      4290
   macro avg       0.66      0.66      0.66      4290
weighted avg       0.64      0.64      0.64      4290


# <------------------------------------------------ TRIAL 3 ------------------------------------------------------------>

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, LSTM, Bidirectional, 
                                     Conv2D, MaxPooling2D, Flatten, concatenate, MultiHeadAttention)
from tensorflow.keras.layers import add
from tensorflow.keras.layers import Dropout

# CNN for Melspectrogram
input_mel = Input(shape=(mel_height, mel_width, 1))  # Adjust shape accordingly
x = Conv2D(512, (3, 3), activation='relu')(input_mel)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(256, (3, 3), activation='relu')(x)
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
attention_output = MultiHeadAttention(num_heads=2, key_dim=64)(y, y)
# Adding residual connection
y = add([y, attention_output])
y = Dropout(0.5)(y)

y = Bidirectional(LSTM(256, return_sequences=True))(y)
attention_output = MultiHeadAttention(num_heads=2, key_dim=64)(y, y)
# Adding residual connection
y = add([y, attention_output])
y = Dropout(0.5)(y)

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

TEST RESULTS
135/135 [==============================] - 4s 28ms/step - loss: 0.9310 - accuracy: 0.6632
Test Accuracy: 66.31701588630676%

CLASSIFICATION REPORT
                precision   recall  f1-score   support

           0       0.75      0.80      0.77       688
           2       0.66      0.61      0.63       688
           3       0.65      0.53      0.58       688
           4       0.61      0.59      0.60       686
           5       0.62      0.67      0.64       612
           6       0.64      0.70      0.67       692
           7       0.78      0.90      0.84       236

    accuracy                           0.66      4290
   macro avg       0.67      0.69      0.68      4290
weighted avg       0.66      0.66      0.66      4290

# <------------------------------------------------ TRIAL 4 ------------------------------------------------------------>

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, LSTM, Bidirectional, 
                                     Conv2D, MaxPooling2D, Flatten, concatenate, MultiHeadAttention)
from tensorflow.keras.layers import add
from tensorflow.keras.layers import Dropout

# CNN for Melspectrogram
input_mel = Input(shape=(mel_height, mel_width, 1))  # Adjust shape accordingly
x = Conv2D(512, (3, 3), activation='relu')(input_mel)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(256, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(256, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)  # Dropout layer added here
cnn_output = Dense(128, activation='relu')(x)

# Bidirectional LSTM for sequential features
input_seq = Input(shape=(timesteps, features))  # Adjust shape accordingly

y = Bidirectional(LSTM(512, return_sequences=True))(input_seq)
attention_output = MultiHeadAttention(num_heads=2, key_dim=64)(y, y)
# Adding residual connection
y = add([y, attention_output])
y = Dropout(0.5)(y)

y = Bidirectional(LSTM(256, return_sequences=True))(y)
attention_output = MultiHeadAttention(num_heads=2, key_dim=64)(y, y)
# Adding residual connection
y = add([y, attention_output])
y = Dropout(0.5)(y)

y = Bidirectional(LSTM(256, return_sequences=True))(input_seq)
attention_output = MultiHeadAttention(num_heads=2, key_dim=64)(y, y)
# Adding residual connection
y = add([y, attention_output])
y = Dropout(0.5)(y)

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


Epoch 16/30
256/256 [==============================] - ETA: 0s - loss: 0.5846 - accuracy: 0.7809Restoring model weights from the end of the best epoch: 11.
256/256 [==============================] - 49s 193ms/step - loss: 0.5846 - accuracy: 0.7809 - val_loss: 0.9229 - val_accuracy: 0.6760 - lr: 1.0000e-05
Epoch 16: early stopping


TEST RESULTS
135/135 [==============================] - 4s 30ms/step - loss: 0.9007 - accuracy: 0.6727
Test Accuracy: 67.27272868156433%

CLASSIFICATION REPORT
                precision   recall  f1-score   support

           0       0.75      0.78      0.76       688
           2       0.64      0.61      0.62       688
           3       0.65      0.54      0.59       688
           4       0.66      0.60      0.63       686
           5       0.65      0.69      0.67       612
           6       0.65      0.75      0.70       692
           7       0.73      0.89      0.80       236

    accuracy                           0.67      4290
   macro avg       0.68      0.69      0.68      4290
weighted avg       0.67      0.67      0.67      4290
