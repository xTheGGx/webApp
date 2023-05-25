import os
import pandas as pd

from multiprocessing import current_process
from flask import request
from sklearn.preprocessing import StandardScaler  
from scipy.spatial.distance import cdist
from scipy.spatial import distance
current_dir = os.path.dirname(os.path.abspath(__file__))

class Metricas:
        
    def __init__(self, fileName):
        self.file = os.path.join(current_dir, f'../../input/{fileName}')

    def createME(self):
       
        df = pd.read_csv(self.file)
        #Logica del algoritmo
        #Estandarizando matriz
        estandarizar = StandardScaler()                               # Se instancia el objeto StandardScaler o MinMaxScaler 

        # Estandarizar matriz
        MEstandarizada = estandarizar.fit_transform(df)    
        euclidianSample = pd.DataFrame(MEstandarizada)
        
        return euclidianSample
   
    def createMEuclidian(self, MEstandarizada):
       
        DstEuclidiana = cdist(MEstandarizada, MEstandarizada, metric='euclidean')
        MEuclidiana = pd.DataFrame(DstEuclidiana)
        
        return MEuclidiana
    def partialEuclidianDistance(self,MEuclidiana,limInferior, limSuperior):
        #Matriz de distancias de una parte del total de objetos

        DstEuclidiana = cdist(MEuclidiana[limInferior:limSuperior], MEuclidiana[limInferior:limSuperior], metric='euclidean')
        MEuclidiana = pd.DataFrame(DstEuclidiana)
        return MEuclidiana

    def distEuclidian(self, MEuclidiana, distA, distB):
        #Distancia entre dos objetos
        Objeto1 = MEuclidiana[distA]
        Objeto2 = MEuclidiana[distB]
        dstEuclidiana = distance.euclidean(Objeto1,Objeto2)
        return dstEuclidiana