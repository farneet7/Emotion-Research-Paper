### ATTEMPT 1
# Define model architecture Bidirectional LSTM
model = Sequential()

# Input shape for the LSTM
input_shape=(DemoT.shape[1], DemoT.shape[2] if len(DemoT.shape) > 2 else 1)

# Adding Bidirectional LSTM
model.add(Bidirectional(LSTM(256, return_sequences=True), input_shape=input_shape))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(labels_encoded_train.shape[1], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
early_stop = EarlyStopping(monitor='val_loss', patience=5)
callbacks_list = [reduce_lr, early_stop]
# Train your model
history = model.fit(DemoT, labels_encoded_train, epochs=15, batch_size=64, validation_data=(DemoV, labels_encoded_valid), callbacks=callbacks_list)


### FEATURES USED: ZCR, CHROMA VECTOR, MELSPECTROGRAM, RMS, ENERGY, ENTROPY, SPECTRAL ENTROPY
69/69 [==============================] - 2s 28ms/step - loss: 1.0790 - accuracy: 0.6003
Test Accuracy: 60.03%
Test Loss: 1.0790

69/69 [==============================] - 2s 32ms/step
Predicted  angry  calm  disgust  fear  happy  neutral  sad  surprised
Actual                                                               
angry        264     1       10    18     37       11    3          2
calm           0    23        2     0      0        0    9          0
disgust       34     3      126    19     45       63   51          5
fear          19     1        6   195     34       25   60          6
happy         55     1       16    47    180       29    9         10
neutral        1     3       20    18     30      194   38          3
sad            1     5       10    33      5       50  236          5
surprised      5     0        4     6      7        0    0         96



CLASSIFICATION REPORT
                 precision  recall   f1-score   support

       angry       0.70      0.76      0.73       346
        calm       0.62      0.68      0.65        34
     disgust       0.65      0.36      0.47       346
        fear       0.58      0.56      0.57       346
       happy       0.53      0.52      0.53       347
     neutral       0.52      0.63      0.57       307
         sad       0.58      0.68      0.63       345
   surprised       0.76      0.81      0.78       118

    accuracy                           0.60      2189
   macro avg       0.62      0.63      0.62      2189
weighted avg       0.60      0.60      0.59      2189




# ATTEMPT 2 - USED ATTENTION
from keras import initializers, regularizers, constraints, backend as K
from keras.layers import Layer

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dropout, Dense, Flatten

# Your provided code
model = Sequential()

# Input shape for the LSTM
input_shape=(DemoT.shape[1], DemoT.shape[2] if len(DemoT.shape) > 2 else 1)

# Adding Bidirectional LSTM
model.add(Bidirectional(LSTM(256, return_sequences=True), input_shape=input_shape))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Dropout(0.5))

# Adding Attention layer
model.add(Attention(DemoT.shape[1])) # Assuming DemoT.shape[1] is your sequence length

# Continue with the rest of your model
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(labels_encoded_train.shape[1], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

#FEATURES USED: ZCR, CHROMA VECTOR, MELSPECTROGRAM, RMS, ENERGY
69/69 [==============================] - 2s 26ms/step - loss: 1.0714 - accuracy: 0.6067
Test Accuracy: 60.67%
Test Loss: 1.0714

69/69 [==============================] - 6s 25ms/step
Predicted  angry  calm  disgust  fear  happy  neutral  sad  surprised
Actual                                                               
angry        269     0       12    17     29       14    2          3
calm           0    26        2     0      0        1    5          0
disgust       35     4      129    23     34       70   44          7
fear          19     2        8   196     33       27   56          5
happy         53     0       15    48    181       35    8          7
neutral        3     4       18    24     23      200   34          1
sad            2     2       14    37      8       45  234          3
surprised      4     0        7     6      6        0    2         93


                precision    recall  f1-score   support

       angry       0.70      0.78      0.74       346
        calm       0.68      0.76      0.72        34
     disgust       0.63      0.37      0.47       346
        fear       0.56      0.57      0.56       346
       happy       0.58      0.52      0.55       347
     neutral       0.51      0.65      0.57       307
         sad       0.61      0.68      0.64       345
   surprised       0.78      0.79      0.78       118

    accuracy                           0.61      2189
   macro avg       0.63      0.64      0.63      2189
weighted avg       0.61      0.61      0.60      2189
