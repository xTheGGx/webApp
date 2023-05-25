import os
import csv
import io
import pandas as pd                 # Para la manipulación y análisis de los datos
import numpy as np                  # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt     # Para la generación de gráficas a partir de los datos
from sklearn.preprocessing import StandardScaler, MinMaxScaler  
from apyori import apriori
from flask import Flask, request, render_template, redirect, url_for, send_from_directory,send_file
from werkzeug.utils import secure_filename
from datetime import datetime
from scipy.spatial.distance import cdist

app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            file.save(os.path.join('input', filename))
            return render_template('upload.html', success_message='Archivo subido correctamente.')
        else:
            error_message = 'Error: El archivo debe tener extensión .csv.'
            return render_template('upload.html', error_message=error_message)
    return render_template('upload.html')

@app.route('/menu')
def menu():
    return render_template('menu.html')

@app.route('/apriori')
def aprioriAlg():
    input_folder = 'input'
    files = os.listdir(input_folder)
    csv_files = [file for file in files if file.endswith('.csv')]
    return render_template('apriori.html', csv_files=csv_files)

@app.route('/apriori/process', methods=['POST'])
def aprioriProcess():
    input_folder = 'input'
    selected_file = request.form['file']
    df = pd.read_csv(os.path.join(input_folder, selected_file), header = None)
    
    # Obtener los valores ingresados por el usuario
    support = float(request.form['support'])
    confidence = float(request.form['confidence'])
    lift = float(request.form['lift'])
    # Realiza el procesamiento deseado con el DataFrame df
    #Se incluyen todas las transacciones en una sola lista
    Transacciones = df.values.reshape(-1).tolist() #-1 significa 'dimensión desconocida'
    #Se crea una matriz (dataframe) usando la lista y se incluye una columna 'Frecuencia'
    Lista = pd.DataFrame(Transacciones)
    Lista['Frecuencia'] = 1
    #Se agrupa los elementos
    Lista = Lista.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=True) #Conteo
    Lista['Porcentaje'] = (Lista['Frecuencia'] / Lista['Frecuencia'].sum()) #Porcentaje
    Lista = Lista.rename(columns={0 : 'Item'})

    #Se muestra la lista
    Lista_html = Lista.to_html()

    # Se genera un gráfico de barras
    plt.figure(figsize=(8,20), dpi=300)
    plt.ylabel('Item')
    plt.xlabel('Frecuencia')
    plt.barh(Lista['Item'], width=Lista['Frecuencia'], color='blue')
    plt.savefig('static/chart.png', bbox_inches='tight')
    #Se crea una lista de listas a partir del dataframe y se remueven los 'NaN'
    #level=0 especifica desde el primer índice
    TransaccionesLista = df.stack().groupby(level=0).apply(list).tolist()

    ReglasC1 = apriori(TransaccionesLista,
                       min_support=support,
                       min_confidence=confidence,
                       min_lift=lift)
    
    Resultados = list(ReglasC1)
    
    formatted_results = []
    for item in Resultados:
        Emparejar = item[0]
        items = [x for x in Emparejar]
        regla = ', '.join(items)
        soporte = item[1]
        confianza = item[2][0][2]
        elevacion = item[2][0][3]
        formatted_results.append((regla, soporte, confianza, elevacion))

    return render_template('aprioriProcess.html', dataframe=Lista_html, enumerated_rules=formatted_results )

@app.route('/metricas')
def metricas():
    input_folder = 'input'
    files = os.listdir(input_folder)
    csv_files = [file for file in files if file.endswith('.csv')]
    return render_template('metricas.html', csv_files=csv_files)

@app.route('/metricas/process', methods=['POST'])
def metricasProcess():
    selected_file = request.form['file']
    option = request.form['option']

    if option == "Euclidiana":
        return redirect(url_for('metricasEuclidiana', selected_file=selected_file))
    elif option == "Chebyshev":
        return redirect(url_for('metricasChebyshev', selected_file=selected_file))
    elif option == "Manhattan":
        return redirect(url_for('metricasManhattan', selected_file=selected_file))
    elif option == "Minkowski":
        return redirect(url_for('metricasMinkowski', selected_file=selected_file))


#Euclidiana, Chebyshev, Manhattan, Minkowski

@app.route("/metricas/euclidian")
def metricasEuclidiana():
    input_folder = 'input'
    selected_file = request.args.get('selected_file')
    df = pd.read_csv(os.path.join(input_folder, selected_file))
    print(df)
    if selected_file is None:
        # Manejar la situación si la clave "file" no está presente en la solicitud POST
        return "Error: No se proporcionó el archivo seleccionado"
    

    #Logica del algoritmo
    #Estandarizando matriz
    estandarizar = StandardScaler()                               # Se instancia el objeto StandardScaler o MinMaxScaler 

    # Estandarizar matriz
    MEstandarizada = estandarizar.fit_transform(df)    
    euclidianSample = pd.DataFrame(MEstandarizada)
    print(MEstandarizada)
    print(euclidianSample)
    euclidianSample_html = euclidianSample.to_html()
    DstEuclidiana = cdist(MEstandarizada, MEstandarizada, metric='euclidean')
    MEuclidiana = pd.DataFrame(DstEuclidiana)
    MEuclidiana_html = MEuclidiana.to_html()
    return render_template("metricasEuclidiana.html",euclidianSample=euclidianSample_html,MEuclidiana=MEuclidiana_html)

@app.route("/metricas/chebyshev")
def metricasChebyshev():
    input_folder = 'input'
    selected_file = request.form['file']
    df = pd.read_csv(os.path.join(input_folder, selected_file), header = None)
    #Logica 
    return render_template("metricasChebyshev.html")

@app.route("/metricas/manhattan")
def metricasManhattan():
    input_folder = 'input'
    selected_file = request.form['file']
    df = pd.read_csv(os.path.join(input_folder, selected_file), header = None)
    #Logica 
    return render_template("metricasManhattan.html")

@app.route("/metricas/minkowsky")
def metricasMinkowsky():
    input_folder = 'input'
    selected_file = request.form['file']
    df = pd.read_csv(os.path.join(input_folder, selected_file), header = None)
    #Logica 
    return render_template("metricasMinkowsky.html")

@app.route('/clustering')
def clustering():
    return render_template('clustering.html')


if __name__ == '__main__':
    app.run(debug=True)
