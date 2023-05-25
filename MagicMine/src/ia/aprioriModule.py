import os
import pandas as pd                 # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt     # Para la generación de gráficas a partir de los datos
import pandas as pd            # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt     # Para la generación de gráficas a partir de los datos
from apyori import apriori


current_dir = os.path.dirname(os.path.abspath(__file__))

class Apriori:
    def __init__(self, fileName):
        self.file = os.path.join(current_dir, f'../../input/{fileName}')

    def createFrecuencyTable(self):
        df = pd.read_csv(self.file)
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
        self.createFrecuencyPlot(Lista)
        #Se muestra la lista
        return Lista
    
    def createFrecuencyPlot(self, Lista):
         # Se genera un gráfico de barras
        current_dir = os.path.dirname(os.path.abspath(__file__))
        chart_path = os.path.join(current_dir, '../../static/chart.png')
        plt.figure(figsize=(8,20), dpi=300)
        plt.ylabel('Item')
        plt.xlabel('Frecuencia')
        plt.barh(Lista['Item'], width=Lista['Frecuencia'], color='blue')
        plt.savefig(chart_path, bbox_inches='tight')

    def apriori(self, support, confidence, lift):
        df = pd.read_csv(self.file)
        #Se crea una lista de listas a partir del dataframe y se remueven los 'NaN'
        #level=0 especifica desde el primer índice
        TransaccionesLista = df.stack().groupby(level=0).apply(list).tolist()
        
        ReglasC1 = apriori(TransaccionesLista,
                        min_support=support,
                        min_confidence=confidence,
                        min_lift=lift)
        
        results = list(ReglasC1)
        data = []
        for relation_record in results:
            items = ', '.join(str(item) for item in relation_record.items)
            support = relation_record.support
            confidence = next(iter(relation_record.ordered_statistics)).confidence
            lift = next(iter(relation_record.ordered_statistics)).lift

            data.append({'Rule': items, 'Support': support, 'Confidence': confidence, 'Lift': lift})

        df = pd.DataFrame(data)
        return df