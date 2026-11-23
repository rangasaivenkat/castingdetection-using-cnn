import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras

IMAGE_SIZE = (224, 224)
MODEL_PATH = "models/casting_cnn.keras"


def predict_image(image_path):
    model = keras.models.load_model(MODEL_PATH)

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
    )
    image = tf.keras.utils.img_to_array(image)
    image = np.expand_dims(image, axis=0)

    probability = float(model.predict(image, verbose=0)[0][0])

    if probability >= 0.5:
        prediction = "Defective"
    else:
        prediction = "OK"

    print("Prediction:", prediction)
    print(f"Defect probability: {probability:.2%}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py path/to/image.jpg")
        sys.exit(1)

    predict_image(sys.argv[1])
