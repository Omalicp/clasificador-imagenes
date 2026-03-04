import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk # type: ignore
import numpy as np # type: ignore
import tensorflow as tf # type: ignore

# Cargar el modelo pre-entrenado
model = tf.keras.models.load_model('best_model.h5')

# Lista de clases, actualizar según el modelo
class_names = ['Clase_1', 'Clase_2', 'Clase_3']

def predict_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))  # Ajustar el tamaño según el modelo
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)
    return predicted_class, confidence

def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        predicted_class, confidence = predict_image(file_path)
        result_label.config(text=f'Predicción: {predicted_class}\nConfianza: {confidence:.2f}')
        img = Image.open(file_path)
        img = img.resize((250, 250))
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img

# Crear la interfaz gráfica
root = tk.Tk()
root.title('Clasificador de Imágenes')
root.geometry('400x500')

btn = tk.Button(root, text='Cargar Imagen', command=open_image)
btn.pack(pady=20)

panel = tk.Label(root)
panel.pack()

result_label = tk.Label(root, text='Predicción: ', font=('Helvetica', 14))
result_label.pack(pady=20)

root.mainloop()
