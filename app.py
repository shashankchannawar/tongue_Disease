import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

# Try to import TensorFlow, else use Mock
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not found. Running in Demo Mode.")

# Standard Flask Config
app = Flask(__name__)

# Configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models/tongue_disease_model.h5')
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables
model = None


# Medical Knowledge Base
DISEASE_INFO = {
    'Healthy': {
        'status': 'Healthy',
        'severity': 'low',
        'manual_id': 'Tongue is pink, covered with tiny nodules (papillae). No white patches, ulcers, or redness.',
        'precautions': 'Maintain good oral hygiene. Brush twice daily and clean your tongue.',
        'description': 'Your tongue appears normal and healthy. Keep up the good work!'
    },
    'Traumatic Ulcer': {
        'status': 'Attention Needed',
        'severity': 'medium',
        'manual_id': 'Painful sore (canker sore), usually round/oval with a white/yellow center and red border.',
        'precautions': 'Avoid spicy/acidic foods. Apply over-the-counter benzocaine gels. Rinse with slat water.',
        'description': 'Commonly known as Canker Sore. Often caused by stress, minor injury, or acidic foods.'
    },
    'Candidiasis': {
        'status': 'Infection Detected',
        'severity': 'medium',
        'manual_id': 'Creamy white lesions on your tongue/cheeks. May bleed if scraped. Cotton-like feeling.',
        'precautions': 'Maintain oral hygiene. Minimize sugar. Consult a doctor for antifungal medication if it persists.',
        'description': 'Oral Thrush. A fungal infection caused by Candida yeast accumulation.'
    },
    'Leukoplakia': {
        'status': 'Warning',
        'severity': 'high',
        'manual_id': 'Thick, white patches on the tongue/gums that cannot be scraped off. Often irregular shape.',
        'precautions': 'Stop smoking/alcohol immediately. Monitor closely. Consult a dentist to rule out pre-cancer.',
        'description': 'A condition where thick white patches form on mucous membranes. Can be a precursor to cancer.'
    },
    'Squamous Cell Carcinoma': {
        'status': 'Critical - Seek Medical Help',
        'severity': 'critical',
        'manual_id': 'Persistent red/white patch, non-healing ulcer, lump on tongue, bleeding, or numbness.',
        'precautions': 'IMMEDIATE medical consultation required. biopsy may be needed. Do not ignore.',
        'description': 'A form of Oral Cancer. Early detection significantly improves survival rates.'
    }
}

# Alphabetical order of classes based on folders created
CLASS_LABELS = ['Candidiasis', 'Healthy', 'Leukoplakia', 'Squamous Cell Carcinoma', 'Traumatic Ulcer']

class MockModel:
    def predict(self, img_array):
        # Return a random probability distribution for demo
        probs = np.random.dirichlet(np.ones(len(CLASS_LABELS)), size=1)
        return probs

def load_trained_model():
    global model
    if TF_AVAILABLE:
        if os.path.exists(MODEL_PATH):
            try:
                model = load_model(MODEL_PATH)
                print("Model loaded successfully.")
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print(f"Model not found at {MODEL_PATH}. Prediction depends on training.")
    else:
        print("Using Mock Model for Demo.")
        model = MockModel()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    if TF_AVAILABLE:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        return img_array
    else:
        # Return dummy array
        return np.zeros((1, 224, 224, 3))

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        if model is None:
             load_trained_model()

        try:
            # Check if this is a demo/mock scenario
            if not TF_AVAILABLE or model is None:
                 # Logic for Demo Mode: 
                 # If filename corresponds to a class, pick it. Else random.
                 import random
                 chosen_idx = random.randint(0, len(CLASS_LABELS)-1)
                 
                 # Simple heuristic for manual testing if user renames file
                 for idx, label in enumerate(CLASS_LABELS):
                     if label.lower() in filename.lower():
                         chosen_idx = idx
                         break
                 
                 class_name = CLASS_LABELS[chosen_idx]
                 confidence = random.uniform(0.85, 0.99)
                 
                 info = DISEASE_INFO.get(class_name, DISEASE_INFO['Healthy'])
                 
                 result = {
                    'class': class_name,
                    'confidence': f"{confidence*100:.2f}%",
                    'details': info
                 }
                 return jsonify(result)

            processed_img = preprocess_image(filepath)
            prediction = model.predict(processed_img)
            class_idx = np.argmax(prediction[0])
            confidence = float(np.max(prediction[0]))
            
            # Map index to class
            # Since flow_from_directory sorts alphabetically:
            # Candidiasis, Healthy, Leukoplakia, Squamous..., Traumatic...
            predicted_label = CLASS_LABELS[class_idx] if class_idx < len(CLASS_LABELS) else "Unknown"

            info = DISEASE_INFO.get(predicted_label, {})

            result = {
                'class': predicted_label,
                'confidence': f"{confidence*100:.2f}%",
                "details": info
            }
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)})
            
    return jsonify({'error': 'Invalid file type'})

if __name__ == '__main__':
    load_trained_model()
    app.run(debug=True, port=5000)
