import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import seaborn as sns

class generate_clusters:
    ''' 
    Clase para generar y visualizar muestras de distribuciones normales multivariadas, 
    exponenciales o uniformes en n dimensiones. 
    '''

    def __init__(self):
        """
        Inicializa la clase.
        """
        self.dist = {}           # Diccionario con etiquetas
        self.dist_no_labels = []  # Lista sin etiquetas
        self.df = None  # DataFrame para los datos con etiquetas

    @classmethod
    def from_multivariate_normal(cls, means, covariances, sample_sizes):
        """
        Genera puntos utilizando la distribución normal multivariada.
        
        Parámetros:
        - means: Diccionario de medias, donde cada media es un vector de n dimensiones.
        - covariances: Diccionario de matrices de covarianza, cada una correspondiente a las medias.
        - sample_sizes: Diccionario de enteros, cada uno correspondiente al número de puntos a generar para cada media.
        
        Retorna:
        - Una instancia de la clase `generate_clusters` con las muestras generadas.
        """
        try:
            instance = cls()
            data = []

            for key in means.keys():
                mean = means[key]
                covariance = covariances[key]
                num_samples = int(sample_sizes[key])
                samples = np.random.multivariate_normal(mean=mean, cov=covariance, size=num_samples)
                instance.dist[key] = samples
                instance.dist_no_labels.append(samples)
                labeled_samples = np.hstack([samples, np.full((num_samples, 1), key)])
                data.append(labeled_samples)

            instance.dist_no_labels = np.vstack(instance.dist_no_labels)
            data = np.vstack(data)
            columns = [f'Dim_{j+1}' for j in range(data.shape[1] - 1)] + ['Group']
            instance.df = pd.DataFrame(data, columns=columns)
            
            return instance
        except Exception as e:
            print(f"Error generating samples: {e}")
            return None


    def plot_2d(self, labels=False):
        """
        Genera un gráfico interactivo en 2D de las muestras.

        Parámetros:
        - labels: Booleano. Si es True, los puntos se colorean según el grupo. Si es False, no se usan etiquetas.
        """
        try:
            dims = self.dist_no_labels.shape[1]

            fig = go.Figure()

            if labels:
                for key in self.dist.keys():
                    samples = self.dist[key]
                    fig.add_trace(go.Scatter(
                        x=samples[:, 0],
                        y=samples[:, 1],
                        mode='markers',
                        marker=dict(size=4),
                        name=key
                    ))
            else:
                fig.add_trace(go.Scatter(
                    x=self.dist_no_labels[:, 0],
                    y=self.dist_no_labels[:, 1],
                    mode='markers',
                    marker=dict(size=4),
                    name='Scatter'
                ))

            buttons = []
            for i in range(dims):
                for j in range(dims):
                    if i != j:
                        if labels:
                            button = {
                                'args': [{'x': [self.dist[key][:, i] for key in self.dist.keys()], 
                                        'y': [self.dist[key][:, j] for key in self.dist.keys()]}],
                                'label': f'Dimension {i+1} vs Dimension {j+1}',
                                'method': 'update'
                            }
                        else:
                            button = {
                                'args': [{'x': [self.dist_no_labels[:, i]], 
                                        'y': [self.dist_no_labels[:, j]]}],
                                'label': f'Dimension {i+1} vs Dimension {j+1}',
                                'method': 'update'
                            }
                        buttons.append(button)

            fig.update_layout(
                title='Interactive 2D Scatter Plot of Multivariate Normal Distributions',
                xaxis_title=f'Dimension 1',
                yaxis_title=f'Dimension 2',
                updatemenus=[{
                    'buttons': buttons,
                    'direction': 'down',
                    'showactive': True,
                }],
            )

            fig.show()
        except Exception as e:
            print(f"Error generating interactive 2D scatter plot: {e}")

    def plot_3d(self, labels=False):
        """
        Genera un gráfico interactivo en 3D de las muestras.

        Parámetros:
        - labels: Booleano. Si es True, los puntos se colorean según el grupo. Si es False, no se usan etiquetas.
        """
        try:
            dims = self.dist_no_labels.shape[1]

            fig = go.Figure()

            if labels:
                for key in self.dist.keys():
                    samples = self.dist[key]
                    fig.add_trace(go.Scatter3d(
                        x=samples[:, 0],
                        y=samples[:, 1],
                        z=samples[:, 2],
                        mode='markers',
                        marker=dict(size=4),
                        name=key
                    ))
            else:
                fig.add_trace(go.Scatter3d(
                    x=self.dist_no_labels[:, 0],
                    y=self.dist_no_labels[:, 1],
                    z=self.dist_no_labels[:, 2],
                    mode='markers',
                    marker=dict(size=4),
                    name='Scatter'
                ))

            buttons = []
            for i in range(dims):
                for j in range(i+1, dims):
                    for k in range(j+1, dims):
                        if labels:
                            button = {
                                'args': [{'x': [self.dist[key][:, i] for key in self.dist.keys()], 
                                        'y': [self.dist[key][:, j] for key in self.dist.keys()], 
                                        'z': [self.dist[key][:, k] for key in self.dist.keys()]}],
                                'label': f'Dim {i+1}, Dim {j+1}, Dim {k+1}',
                                'method': 'update'
                            }
                        else:
                            button = {
                                'args': [{'x': [self.dist_no_labels[:, i]], 
                                        'y': [self.dist_no_labels[:, j]], 
                                        'z': [self.dist_no_labels[:, k]]}],
                                'label': f'Dim {i+1}, Dim {j+1}, Dim {k+1}',
                                'method': 'update'
                            }
                        buttons.append(button)

            fig.update_layout(
                title='Interactive 3D Scatter Plot of Multivariate Normal Distributions',
                scene=dict(
                    xaxis_title='Dimension 1',
                    yaxis_title='Dimension 2',
                    zaxis_title='Dimension 3'
                ),
                updatemenus=[{
                    'buttons': buttons,
                    'direction': 'down',
                    'showactive': True,
                }],
            )

            fig.show()
        except Exception as e:
            print(f"Error generating interactive 3D scatter plot: {e}")

    def pairplot(self, labels=False, palette='summer'):
        """
        Genera un gráfico de pares (pairplot) de todas las combinaciones posibles de las dimensiones.

        Parámetros:
        - labels: Booleano. Si es True, los puntos se colorean según el grupo. Si es False, no se usan etiquetas.
        - palette: Paleta de colores utilizada por Seaborn.
        """
        try:
            if labels:
                pair = self.to_df(labels=True) 
                sns.pairplot(pair, palette=palette, hue='Group')
                plt.show()
            else:
                pair = self.to_df(labels=False)
                sns.pairplot(pair, palette=palette)
                plt.show()
        except Exception as e:
            print(f"Error generating pairplot: {e}")

    def to_df(self, labels=False):
        """
        Convierte las muestras generadas en un DataFrame de pandas.

        Parámetros:
        - labels: Booleano. Si es True, se incluye una columna con los grupos.

        Retorna:
        - Un DataFrame con las muestras generadas.
        """
        try:
            df_copy = self.df.copy()
            df_copy.replace([np.inf, -np.inf], np.nan)  # Reemplaza valores infinitos con NaN

            for col in df_copy.columns: 
                df_copy[col] = df_copy[col].astype(float)  # Convierte las columnas a float

            if labels:
                return df_copy
            else:
                return df_copy.drop(columns=['Group'])
        except Exception as e:
            print(f"Error generating DataFrame: {e}")
            return None
        
    @property
    def get_points(self):
        """Devuelve el número de puntos."""
        if self.df is not None:
            return print(self.df.shape[0])
        return 0

    @property
    def get_dimensions(self):
        """Devuelve la dimensión de los puntos."""
        if self.df is not None:
            return print(self.df.shape[1]-1)
        return 0
