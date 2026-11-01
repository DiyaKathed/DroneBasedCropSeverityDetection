import tensorflow as tf

model = tf.keras.models.load_model(
    "models/plantdoc_disease_model.keras"
)

print("Model loaded successfully!")
model.summary()