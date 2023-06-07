import io
import os
import pandas as pd
import matplotlib as plt
import seaborn as sns 
import numpy as np
import base64
import scipy.cluster.hierarchy as shc

from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from multiprocessing import current_process
from flask import request
from sklearn.preprocessing import StandardScaler  
from scipy.spatial.distance import cdist
from scipy.spatial import distance
current_dir = os.path.dirname(os.path.abspath(__file__))

class Clustering:
        
    def __init__(self, fileName):
        self.file = os.path.join(current_dir, f'../../../input/{fileName}')
        self.df = pd.read_csv(self.file)
    def createCorrMatrix(self):
        
        
        corrMatrix =  self.df.corr(method='pearson')
         
        return corrMatrix
    
    def createHeatMap(self,df):
        buffer = io.BytesIO()
        plt.figure(figsize = (14,7) )
        MatrizInf = np.triu(df)
        sns.heatmap(df, cmap = 'RdBu_r', annot = True, mask = MatrizInf)
        plt.show()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        # Codificar la imagen en base64
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        return image_base64
    
    def createMVar(self):
        print(self.df)
        nombres_columnas = self.df.columns.to_numpy()
        MatrizHipoteca = np.array(self.df[['ingresos', 'gastos_comunes', 'pago_coche', 'gastos_otros', 'ahorros', 'vivienda', 'estado_civil', 'hijos', 'trabajo']])
        

        return MatrizHipoteca
    
    #Recibe variables con menor dependencia (apoyandose del mapa de calor)
    def MEstandarizada(self,MatrizVariables):
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        # Crear una instancia de StandardScaler
        estandarizar = StandardScaler()
        # Ajustar y transformar la matriz de variables
        MEstandarizada = estandarizar.fit_transform(MatrizVariables)
        # Crear un DataFrame con la matriz estandarizada
        pd.DataFrame(MEstandarizada)
        return MEstandarizada
    
    def hierarchyClusterPNG(self,MEstandarizada):
        plt.figure(figsize=(10,7))
        plt.title("Pacientes con cáncer de mama")  # Corregido el nombre de la función
        plt.xlabel("Observaciones")
        plt.ylabel("Distancia")

        Arbol = shc.dendrogram(shc.linkage(MEstandarizada, method='complete', metric='euclidean'))

        # Guardar la figura en un buffer de memoria
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        
        # Codificar la imagen en base64
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return image_base64
    
    def hierarchyClusterArray(self,MEstandarizada):
        MJerarquico = AgglomerativeClustering(n_clusters=4, linkage='complete', affinity='euclidean')
        MJerarquico.fit_predict(MEstandarizada)
        MJerarquico.labels_
        return MJerarquico
    
    def centroids(self,BCancer):
        CentroidesH = BCancer.groupby(['clusterH'])[['Texture', 'Area', 'Smoothness', 'Compactness', 'Symmetry', 'FractalDimension']].mean()
        return CentroidesH
   
    def dispersionGraph(self, MEstandarizada,MJerarquico):
        plt.figure(figsize = (10,7))
        plt.scatter(MEstandarizada[:,0], MEstandarizada[:,1], c=MJerarquico.labels_)
        plt.grid()

        # Guardar la figura en un buffer de memoria
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        
        # Codificar la imagen en base64
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return image_base64