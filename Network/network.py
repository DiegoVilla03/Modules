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
        self.activation_function = None

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
    
    
    def plot(self, palette: str = 'summer', edge_label_pos: float = 0.8):
        """
        Visualiza la estructura de la red neuronal, mostrando pesos en las aristas y umbrales en los nodos.
        :param palette: Paleta de colores para las conexiones entre capas.
        :param edge_label_pos: Posición de las etiquetas de las aristas a lo largo de las aristas.
        """
        cmap = plt.cm.get_cmap(palette, len(set(nx.get_node_attributes(self.graph, 'subset').values())) - 1)
        
        pos = nx.multipartite_layout(self.graph, subset_key="subset")
        plt.figure(figsize=(8, 8))
        
        nx.draw_networkx_nodes(self.graph, pos, node_color='white', edgecolors='black', linewidths=1, node_size=800)
        
        current_edge = 0
        for i in range(len(set(nx.get_node_attributes(self.graph, 'subset').values())) - 1):
            layer1_size = sum(1 for _, attr in self.graph.nodes(data=True) if attr['subset'] == i)
            layer2_size = sum(1 for _, attr in self.graph.nodes(data=True) if attr['subset'] == i + 1)
            color = cmap(i / (len(set(nx.get_node_attributes(self.graph, 'subset').values())) - 1))
            edges_to_draw = self.edges[current_edge:current_edge + layer1_size * layer2_size]
            nx.draw_networkx_edges(self.graph, pos, edgelist=edges_to_draw, arrows=True, arrowstyle='->', edge_color=color)
            
            # Asegurarse de que todas las aristas tengan el atributo 'weight'
            edge_labels = {}
            for u, v in edges_to_draw:
                weight = self.graph[u][v].get("weight", 0.0)  # Usar 0.0 como valor por defecto
                edge_labels[(u, v)] = f'{weight:.2f}'
            
            # Cambiar la posición de las etiquetas de las aristas usando edge_label_pos
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, label_pos=edge_label_pos)
            
            current_edge += layer1_size * layer2_size
        
        threshold_labels = {node: f'{self.graph.nodes[node].get("threshold", 0):.2f}' for node in self.graph.nodes}
        nx.draw_networkx_labels(self.graph, pos, labels=threshold_labels, font_color='black', font_size=10)
        
        plt.show()

    def weights(self, weights=None):
        """
        Asigna pesos a las aristas de la red neuronal.
        :param weights: Puede ser un diccionario donde las claves son tuplas (nodo1, nodo2) y los valores son los pesos,
        o una lista de pesos que se asignarán a las aristas en orden. Si no se proporciona, se generan pesos aleatorios.
        """
        # Revisar todas las aristas en la gráfica
        all_edges = list(self.graph.edges)

        if isinstance(weights, dict):
            # Usar el diccionario de pesos directamente
            weights_dict = weights
        else:
            # Generar pesos aleatorios si no se proporciona ninguno
            weights_dict = {edge: np.random.uniform(-1, 1) for edge in all_edges}
        
        nx.set_edge_attributes(self.graph, weights_dict, "weight")

    def thresholds(self, thresholds=None):
        """
        Asigna umbrales a las neuronas en las capas ocultas y finales.
        :param thresholds: Puede ser un diccionario donde las claves son los nodos y los valores son los umbrales,
        o una lista de umbrales que se asignarán a los nodos en orden. Si no se proporciona, se generan umbrales aleatorios.
        """
        hidden_and_output_nodes = [node for node in self.graph.nodes if self.graph.nodes[node]['subset'] > 0]

        if isinstance(thresholds, dict):
            # Usar el diccionario de umbrales directamente
            thresholds_dict = thresholds
        else:
            # Generar umbrales aleatorios si no se proporciona ninguno
            thresholds_dict = {node: np.random.uniform(-1, 1) for node in hidden_and_output_nodes}
        
        nx.set_node_attributes(self.graph, thresholds_dict, "threshold")

    
    def set_activation_function(self, activation_function):
        """
        Guarda la función de activación para las neuronas.
        :param activation_function: Una función que toma una entrada y devuelve una salida (e.g., función sigmoide).
        """
        self.activation_function = activation_function

class logic_and(NeuralNetwork):
    
    def __init__(self) -> None:
        super().__init__(np.array([2, 1])) 
        weights = {(0, 2): 1, (1, 2): 1}  
        self.weights(weights) 

        thresholds = {2: 2}
        self.thresholds(thresholds) 
        self.set_activation_function(lambda x: 1 if x >= 0 else 0)
        

class logic_or(NeuralNetwork):
    
    def __init__(self):
        super().__init__(np.array([2,1]))
        weights = {(0,2): 1, (1,2): 1}
        self.weights(weights)
        
        thresholds = {2:1}
        self.thresholds(thresholds)
        self.set_activation_function(lambda x:1 if x >= 0 else 0)
        

class logic_not(NeuralNetwork):
    
    def __init__(self):
        super().__init__(np.array([1,1]))
        weights = {(0,1): -1}
        self.weights(weights)
        
        thresholds = {1:-0.5}
        self.thresholds(thresholds)
        self.set_activation_function(lambda x:1 if x >= 0 else 0)