import json
import numpy as np
import tensorflow as tf

with open('labels.json') as f:
    labels = json.load(f)

model = tf.keras.applications.ResNet50()
image_path = 'user_photo.jpg'
image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = model.predict(input_arr)
label = labels[str(predictions[0].argmax())]
print(f"Ini adalah gambar {label}")