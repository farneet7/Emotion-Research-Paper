### Case 1
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


### CASE 1: WHEN MELSPECTROGRAM IS CONSIDERED
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
