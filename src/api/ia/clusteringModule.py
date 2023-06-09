import io
import os
import pandas as pd
import matplotlib as plt
import seaborn as sns 
import numpy as np
import base64
import scipy.cluster.hierarchy as shc
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.metrics import pairwise_distances_argmin_min
from multiprocessing import current_process
from flask import request
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.spatial.distance import cdist
from scipy.spatial import distance


current_dir = os.path.dirname(os.path.abspath(__file__))

class Clustering:
        
    def __init__(self, fileName):
        self.file = os.path.join(current_dir, f'../../../input/{fileName}')
        self.df = pd.read_csv(self.file)
   
    def createCorrMatrix(self, dfNum):     
        corrMatrix =  dfNum.corr(method='pearson')
        return corrMatrix
    
    def delNoNums(self):
        DFNum = pd.DataFrame()
        for columna in self.df.columns:
            if self.df[columna].dtype in ['int64','float64']:
                DFNum[columna] = self.df[columna]
        return (DFNum)

    def createHeatMap(self, CorrMatrix):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        chart_path = os.path.normpath(os.path.join(current_dir, '../../static/img/charts/hMap.png'))
        plt.figure(figsize=(10, 8))
        MatrizInf = np.triu(CorrMatrix)
        sns.heatmap(CorrMatrix, cmap='RdBu_r', annot=True, mask=MatrizInf)
        plt.savefig(chart_path, bbox_inches='tight')
        return chart_path
    
    def getColumns(self):
        vars = list(self.df.columns.values)
        return vars

    def createMVar(self,var):
        MVar = np.array(self.df[var])
        return MVar
    
    #Recibe variables con menor dependencia (apoyandose del mapa de calor)
    def MEstandarizada(self,MatrizVariables):
        # Crear una instancia de StandardScaler
        MatrizVariables
        estandarizar = StandardScaler()
        # Ajustar y transformar la matriz de variables
        MEstandarizada = estandarizar.fit_transform(MatrizVariables)
        # Crear un DataFrame con la matriz estandarizada
      
        return MEstandarizada
    
    def PartitionalClusterPNG(self,MEstandarizada,metric):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        chart_path = os.path.normpath(os.path.join(current_dir, '../../static/img/charts/pCluster.png'))
        plt.figure(figsize=(10,7))
        plt.title("Cluster Particional")  # Corregido el nombre de la función
        plt.xlabel("Observaciones")
        plt.ylabel("Distancia")
        Arbol = shc.dendrogram(shc.linkage(MEstandarizada,method='complete',metric=metric))
        plt.savefig(chart_path, bbox_inches='tight')
        return chart_path
    
    def hierarchyClusterArray(self,MEstandarizada,metric):
        MJerarquico = AgglomerativeClustering(n_clusters=4, linkage='complete', affinity=metric)
        MJerarquico.fit_predict(MEstandarizada)
        
        return MJerarquico
    
    def centroidsP(self,var):
        CentroidesP = self.df.groupby(['clusterP'])[var].mean()
        return CentroidesP
    
    def centroidsH(self,var):
        CentroidesH = self.df.groupby(['clusterH'])[var].mean()
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
 
    def elbowMethod(self, MEstandarizada, i):
        SSE = []
        for i in range(2, 12):
            km = KMeans(n_clusters=i, random_state=0)
            km.fit(MEstandarizada)
            SSE.append(km.inertia_)

        # Graficar SSE en función de k
        plt.figure(figsize=(10, 7))
        plt.plot(range(2, 12), SSE, marker='o')
        plt.xlabel('Cantidad de clusters *k*')
        plt.ylabel('SSE')
        plt.title('Elbow Method')
        
        # Encontrar el punto de inflexión
        kl = KneeLocator(range(2, 12), SSE, curve="convex", direction="decreasing")
        kl.elbow

        # Obtener la ruta completa de la carpeta "static"
        static_folder = os.path.join(os.getcwd(), 'static')
        
        # Guardar la gráfica como imagen en la carpeta "static"
        image_path = os.path.join(static_folder, 'knee_plot.png')
        plt.savefig(image_path)
        
        # Retornar el nombre del archivo de la gráfica
        return 'knee_plot.png'


