import os
import pandas as pd

from multiprocessing import current_process
from flask import request
from sklearn.preprocessing import StandardScaler  
from scipy.spatial.distance import cdist
from scipy.spatial import distance
current_dir = os.path.dirname(os.path.abspath(__file__))

class Clustering:
        
    def __init__(self, fileName):
        self.file = os.path.join(current_dir, f'../../input/{fileName}')

    def createCorrMatrix(self):
        
        df = pd.read_csv(self.file)
        corrMatrix =  df.corr(method='pearson')
         
        return