import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class NeuralNetwork:
    def __init__(self, layers:np.array):
        """
        Inicializa la red neuronal y genera su estructura.
        :param layers: Un iterador donde cada elemento es el número de neuronas en esa capa.
        """
        self.graph = nx.DiGraph()  # Grafo dirigido
        self.edges = []
        self.layers = layers
        self._generate()  # Llama al método privado _generate

    def _generate(self):
        """
        Genera la estructura de la red neuronal.
        """
        current_node = 0
        for i, layer_size in enumerate(self.layers):
            for j in range(layer_size):
                self.graph.add_node(current_node + j, subset=i)
            
            if i < len(self.layers) - 1:
                layer1_size = self.layers[i]
                layer2_size = self.layers[i + 1]
                
                for j in range(layer1_size):
                    for k in range(layer2_size):
                        self.edges.append((current_node + j, current_node + layer1_size + k))
            
            current_node += layer_size
        
        self.graph.add_edges_from(self.edges)
    
    def plot(self, palette:str='summer'):
        """
        Visualiza la estructura de la red neuronal.
        :param palette: Paleta de colores para las conexiones entre capas.
        """
        cmap = plt.cm.get_cmap(palette, len(set(nx.get_node_attributes(self.graph, 'subset').values())) - 1)
        
        pos = nx.multipartite_layout(self.graph, subset_key="subset")
        plt.figure(figsize=(10, 8))
        nx.draw_networkx_nodes(self.graph, pos, node_color='white', edgecolors='black', linewidths=1)
        
        current_edge = 0
        for i in range(len(set(nx.get_node_attributes(self.graph, 'subset').values())) - 1):
            layer1_size = sum(1 for _, attr in self.graph.nodes(data=True) if attr['subset'] == i)
            layer2_size = sum(1 for _, attr in self.graph.nodes(data=True) if attr['subset'] == i + 1)
            color = cmap(i / (len(set(nx.get_node_attributes(self.graph, 'subset').values())) - 1))
            edges_to_draw = self.edges[current_edge:current_edge + layer1_size * layer2_size]
            nx.draw_networkx_edges(self.graph, pos, edgelist=edges_to_draw, arrows=True, arrowstyle='->', edge_color=color)
            current_edge += layer1_size * layer2_size
        
        plt.show()

