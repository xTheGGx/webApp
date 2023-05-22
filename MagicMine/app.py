import os
import csv

from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from datetime import datetime

ALLOWED_EXTENSIONS = set(['csv'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_location = os.path.join('input', filename)
            file.save(save_location)

            # Aquí puedes realizar el procesamiento del archivo si es necesario
            # y guardar el archivo de salida en la carpeta "output"
            output_file = "archivo_procesado.csv"  # ¡Reemplaza esto con tu propio archivo de salida!

            return redirect(url_for('download', filename=output_file))
        
    return render_template('upload.html')

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory('output', filename)

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
