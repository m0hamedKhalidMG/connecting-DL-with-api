





"""
Created on Mon April 29 21:36:16 2024

@author: mohamed khalid
"""
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the pre-trained model
loaded_model = load_model('D:/datase/dataset/your_model_file.h5')
class_labels = ['DFU', 'Wound']  

def preprocess_image(image):
    img = image.resize((150, 150))  # Resize the image to match model input size
    img_array = np.array(img)  # Convert PIL image to numpy array
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        try:
            image = Image.open(io.BytesIO(file.read()))
            img_array = preprocess_image(image)

            # Make predictions using the loaded model
            predictions = loaded_model.predict(img_array)

            # Get the predicted class index
            predicted_class_index = np.argmax(predictions)
            predicted_class_label = class_labels[predicted_class_index]

            # Get class probabilities
            class_probabilities = predictions[0].tolist()

            return jsonify({
                'predicted_class': predicted_class_label,
                'class_probabilities': dict(zip(class_labels, class_probabilities))
            })
        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
