import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

# Path to the saved model
MODEL_PATH = 'model_vgg.h5'

# Load your trained model
model = load_model(MODEL_PATH)

# Define class names
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

def preprocess_image(image_path):
    """Preprocess the image by resizing and normalizing it."""
    img = image.load_img(image_path, target_size=(256, 256))  # Adjusted target_size
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0
    return img

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction."""
    if request.method == 'POST':
        # Get the file from the POST request
        f = request.files['file']
        
        # Ensure the uploads directory exists
        uploads_dir = os.path.join('uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        
        # Save the file to the uploads directory
        file_path = os.path.join(uploads_dir, secure_filename(f.filename))
        f.save(file_path)
        
        
        # Preprocess the image and predict
        img = preprocess_image(file_path)
        preds = model.predict(img)
        predicted_class = class_names[np.argmax(preds)]
        
        # Log the raw predictions and predicted class
        print("Predictions:", preds)
        print("Predicted Class:", predicted_class)
        
        # Return the result as JSON
        return jsonify({'class': predicted_class})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)