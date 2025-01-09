from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import json
from io import BytesIO

app = Flask(__name__)
CORS(app)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="breed_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load class labels
with open("breed_names.json", "r") as file:
    class_labels = json.load(file)

# Preprocess the uploaded image
def preprocess_image(img, target_size=(260, 260)):
    img = img.resize(target_size)  # Resize image to match model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)  # Normalize to [0, 1] if necessary
    return img_array

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    # Read and preprocess the image
    file = request.files["file"]
    img = image.load_img(BytesIO(file.read()), target_size=(260, 260))
    img_array = preprocess_image(img)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Run inference
    interpreter.invoke()

    # Get output predictions
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]  # Get probabilities for each class
    top_index = np.argmax(predictions)  # Index of the top class

    # Response
    top_result = {
        "breed": class_labels[top_index],
        "confidence": round(float(predictions[top_index]), 3),
    }
    return jsonify(top_result)

if __name__ == "__main__":
    app.run(debug=True)
