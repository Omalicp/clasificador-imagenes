from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# Cargar el modelo pre-entrenado
model = tf.keras.models.load_model('best_model.h5')

# Lista de clases, actualizar según el modelo
class_names = ['bolso', 'calzado', 'pantalon - short', 'polo']

def predict_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))  # Ajustar el tamaño según el modelo
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)
    return predicted_class, confidence

# Crear la aplicación Flask
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            image_path = os.path.join('uploads', file.filename)
            file.save(image_path)
            predicted_class, confidence = predict_image(image_path)
            return render_template('index.html', prediction=predicted_class, confidence=confidence)
    return render_template('index.html')

if __name__ == '__main__':
    # Crear la carpeta 'uploads' si no existe
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
