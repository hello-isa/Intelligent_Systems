import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation

model = tf.keras.models.Sequential([
    Dense(5, input_shape=(3,), activation='relu'),  # Corrected input_shape
    Dense(3, activation='softmax')
])

# Compile the model (optional, depending on your next steps)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model (optional, for checking the model architecture)
model.summary()
