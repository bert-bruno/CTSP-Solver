# %%
# --- Imports --- #
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Set, Optional
import random
import time
from dataclasses import dataclass
import heapq
import sys
from pathlib import Path
import os

import matplotlib.pyplot as plt

from src.cluster_handling import *

# %%
# --- Load Data --- #
# Get the project root directory 
project_root = Path().resolve()
sys.path.append(str(project_root / 'src'))

# %%
# --- Path --- #
data_path = Path().resolve().joinpath('data','CTSP_instances')
ctsp = list(data_path.glob("*.csv"))
# --- Load Data --- #
ctsp_instances = load_data(ctsp)

# %%
@dataclass
class Solution:
    """Classe para armazenar informações da solução"""
    route: List[int]
    cost: float
    
    def __lt__(self, other):
        """Necessário para operações com heapq"""
        return self.cost < other.cost

class HybridGRASP_TabuCTSP:
    def __init__(self, 
                 df: pd.DataFrame,
                 max_time: float = 300.0,  # 5 minutos por padrão
                 grasp_alpha: float = 0.2,
                 grasp_iterations: int = 20,
                 elite_pool_size: int = 10,
                 tabu_tenure: Tuple[int, int] = (20, 30)):
        """
        Inicializa o solver híbrido GRASP-Tabu Search para CTSP.
        
        Args:
            df: DataFrame contendo:
                - Coluna 'cluster' indicando a qual cluster cada cidade pertence
                - Matriz de distâncias (demais colunas)
            max_time: Tempo máximo de execução em segundos
            grasp_alpha: Parâmetro GRASP para tamanho da RCL
            grasp_iterations: Número de iterações GRASP para soluções iniciais
            elite_pool_size: Tamanho do pool de soluções elite
            tabu_tenure: Intervalo para tenure tabu aleatório (min, max)
        """
        # Extrai os clusters do DataFrame
        cluster_series = df['cluster']
        unique_clusters = sorted(cluster_series.unique())
        
        self.clusters = []
        for cluster_id in unique_clusters:
            # Obtém os índices das cidades neste cluster
            cluster_cities = cluster_series[cluster_series == cluster_id].index.tolist()
            self.clusters.append(cluster_cities)
        
        # Extrai a matriz de distâncias
        distance_cols = [col for col in df.columns if col != 'cluster']
        self.distances = df[distance_cols].values
        
        # Configura os parâmetros do algoritmo
        self.max_time = max_time
        self.grasp_alpha = grasp_alpha
        self.grasp_iterations = grasp_iterations
        self.elite_pool_size = elite_pool_size
        self.tabu_tenure = tabu_tenure
        
        # Inicializa estruturas de dados de suporte
        self.n_cities = len(self.distances)
        self.n_clusters = len(self.clusters)
        
        # Cria lookup de pertencimento aos clusters
        self.city_to_cluster = {}
        for cluster_idx, cluster in enumerate(self.clusters):
            for city in cluster:
                self.city_to_cluster[city] = cluster_idx
        
        # Inicializa pool de soluções elite usando min heap
        self.elite_solutions = []
        
    def calculate_cost(self, route: List[int]) -> float:
        """Calcula a distância total de uma rota."""
        return sum(self.distances[route[i]][route[i + 1]] 
                  for i in range(len(route) - 1))

    def check_contiguous_clusters(self, route: List[int]) -> bool:
        """Verifica se os clusters são visitados de forma contígua."""
        route = route[:-1]  # Remove última cidade (igual à primeira)
        current_cluster = self.city_to_cluster[route[0]]
        cluster_cities = set()
        
        for city in route:
            cluster = self.city_to_cluster[city]
            if cluster != current_cluster:
                # Verifica se o cluster anterior foi completamente visitado
                if cluster_cities != set(self.clusters[current_cluster]):
                    return False
                current_cluster = cluster
                cluster_cities = {city}
            else:
                cluster_cities.add(city)
                
        return cluster_cities == set(self.clusters[current_cluster])

    def construct_grasp_solution(self) -> Solution:
        """
        Constrói uma única solução usando GRASP.
        Retorna um objeto Solution com rota e custo.
        """
        # Inicializa rota vazia
        route = []
        available_clusters = list(range(self.n_clusters))
        
        while available_clusters:
            if not route:
                # Começa com cluster aleatório
                cluster_idx = random.choice(available_clusters)
                cluster = self.clusters[cluster_idx]
                # Adiciona permutação aleatória das cidades do cluster
                route.extend(random.sample(cluster, len(cluster)))
                available_clusters.remove(cluster_idx)
            else:
                # Constrói lista de candidatos para próximo cluster
                candidates = []
                last_city = route[-1]
                
                for cluster_idx in available_clusters:
                    cluster = self.clusters[cluster_idx]
                    # Calcula custo de conexão com cada cidade no cluster
                    for city in cluster:
                        cost = self.distances[last_city][city]
                        candidates.append((cost, cluster_idx, city))
                
                candidates.sort()  # Ordena por custo
                
                # Cria RCL baseada no parâmetro alpha
                min_cost = candidates[0][0]
                max_cost = candidates[-1][0]
                threshold = min_cost + self.grasp_alpha * (max_cost - min_cost)
                rcl = [c for c in candidates if c[0] <= threshold]
                
                # Seleciona cluster e cidade inicial da RCL
                chosen = random.choice(rcl)
                cluster_idx = chosen[1]
                start_city = chosen[2]
                
                # Obtém cidades restantes do cluster escolhido
                cluster = self.clusters[cluster_idx]
                remaining_cities = [c for c in cluster if c != start_city]
                
                # Adiciona cidades do cluster à rota
                route.append(start_city)
                route.extend(random.sample(remaining_cities, len(remaining_cities)))
                available_clusters.remove(cluster_idx)
        
        # Fecha o tour
        route.append(route[0])
        cost = self.calculate_cost(route)
        
        return Solution(route=route, cost=cost)

    def generate_initial_solutions(self) -> None:
        """
        Gera pool inicial de soluções elite usando GRASP.
        Atualiza self.elite_solutions.
        """
        for _ in range(self.grasp_iterations):
            solution = self.construct_grasp_solution()
            
            if len(self.elite_solutions) < self.elite_pool_size:
                heapq.heappush(self.elite_solutions, solution)
            else:
                heapq.heappushpop(self.elite_solutions, solution)

    def apply_tabu_search(self, initial_solution: Solution, 
                         end_time: float) -> Solution:
        """
        Aplica Busca Tabu para melhorar uma solução até o limite de tempo.
        
        Args:
            initial_solution: Solução inicial
            end_time: Tempo quando a busca deve parar
            
        Returns:
            Melhor solução encontrada
        """
        current = initial_solution
        best = initial_solution
        
        # Inicializa lista tabu como um conjunto de movimentos proibidos (pares de cidades)
        tabu_list: Set[Tuple[int, int]] = set()
        
        while time.time() < end_time:
            # Gera vizinhança usando movimentos 4-opt*
            improved = False
            best_neighbor = None
            best_delta = 0
            
            # Amostra posições para três cadeias
            route = current.route
            n = len(route)
            
            # Tenta uma amostra de possíveis movimentos 4-opt*
            for _ in range(100):  # Limita número de movimentos a verificar
                # Seleciona posições aleatoriamente para as cadeias
                i = random.randint(1, n-3)
                j = random.randint(i+1, n-2)
                k = random.randint(j+1, n-1)
                
                # Verifica se o movimento quebraria a contiguidade do cluster
                if (self.city_to_cluster[route[i]] != self.city_to_cluster[route[i-1]] or
                    self.city_to_cluster[route[j]] != self.city_to_cluster[route[j-1]] or
                    self.city_to_cluster[route[k]] != self.city_to_cluster[route[k-1]]):
                    continue
                
                # Calcula mudança no custo
                current_cost = (self.distances[route[i-1]][route[i]] +
                              self.distances[route[j-1]][route[j]] +
                              self.distances[route[k-1]][route[k]])
                
                new_cost = (self.distances[route[i-1]][route[j]] +
                          self.distances[route[j-1]][route[k]] +
                          self.distances[route[k-1]][route[i]])
                
                delta = new_cost - current_cost
                
                # Verifica se movimento é não-tabu ou satisfaz critério de aspiração
                move_tuple = (route[i-1], route[j])
                is_tabu = move_tuple in tabu_list
                
                if (not is_tabu or current.cost + delta < best.cost) and \
                   (best_neighbor is None or delta < best_delta):
                    # Cria nova solução
                    new_route = (route[:i] + 
                               route[j:k] + 
                               route[i:j] + 
                               route[k:])
                    
                    # Verifica se clusters permanecem contíguos
                    if self.check_contiguous_clusters(new_route):
                        improved = True
                        best_neighbor = new_route
                        best_delta = delta
            
            if improved:
                # Atualiza solução atual
                current = Solution(route=best_neighbor,
                                cost=current.cost + best_delta)
                
                # Atualiza melhor solução se melhorou
                if current.cost < best.cost:
                    best = current
                
                # Atualiza lista tabu
                tabu_tenure = random.randint(*self.tabu_tenure)
                if len(tabu_list) >= tabu_tenure:
                    tabu_list.pop()
                tabu_list.add((best_neighbor[-2], best_neighbor[-1]))
            
            else:
                # Se não encontrou melhoria, perturba solução
                # trocando dois clusters aleatórios
                route = current.route[:-1]  # Remove última cidade
                cluster_indices = list(range(self.n_clusters))
                c1, c2 = random.sample(cluster_indices, 2)
                
                # Encontra fronteiras dos clusters
                boundaries = []
                current_cluster = self.city_to_cluster[route[0]]
                start = 0
                
                for i, city in enumerate(route):
                    if self.city_to_cluster[city] != current_cluster:
                        boundaries.append((start, i))
                        start = i
                        current_cluster = self.city_to_cluster[city]
                boundaries.append((start, len(route)))
                
                # Troca clusters
                new_route = route[:boundaries[c1][0]]
                new_route.extend(route[boundaries[c2][0]:boundaries[c2][1]])
                new_route.extend(route[boundaries[c1][1]:boundaries[c2][0]])
                new_route.extend(route[boundaries[c1][0]:boundaries[c1][1]])
                new_route.extend(route[boundaries[c2][1]:])
                new_route.append(new_route[0])  # Fecha tour
                
                current = Solution(route=new_route,
                                cost=self.calculate_cost(new_route))
        
        return best

    def solve(self) -> Tuple[List[int], float]:
        """
        Procedimento principal de resolução combinando GRASP e Busca Tabu.
        
        Returns:
            Tupla contendo:
            - Melhor rota encontrada
            - Custo total da rota
        """
        start_time = time.time()
        end_time = start_time + self.max_time
        
        # Gera soluções iniciais usando GRASP
        self.generate_initial_solutions()
        
        # Aplica Busca Tabu para cada solução elite
        best_solution = None
        
        while self.elite_solutions and time.time() < end_time:
            initial_solution = heapq.heappop(self.elite_solutions)
            improved_solution = self.apply_tabu_search(initial_solution, end_time)
            
            if best_solution is None or improved_solution.cost < best_solution.cost:
                best_solution = improved_solution
        
        if best_solution is None:
            raise RuntimeError("Nenhuma solução encontrada dentro do limite de tempo")
            
        return best_solution.route, best_solution.cost

# %%
# --- Visualize Route --- #
def visualize_route(df: pd.DataFrame, 
                   solution: List[int],
                   title: str = "CTSP Route Visualization",
                   figsize: tuple = (12, 8),
                   save_path: Optional[str] = None,
                   dpi: int = 300) -> None:
    """
    Visualize a CTSP route with cities colored by cluster and optionally save to file.
    
    Args:
        df: DataFrame containing:
            - X: X coordinates
            - Y: Y coordinates
            - cluster: cluster assignments
        solution: List of city indices in the order they are visited
        title: Title for the plot
        figsize: Figure size as (width, height)
        save_path: Path where to save the PNG file. If None, display only
        dpi: Dots per inch for saved figure (higher means better quality)
    """
    # Create figure and axis with specified size
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    # Get unique clusters for color assignment
    unique_clusters = sorted(df['cluster'].unique())
    
    # Create a color map for clusters using a qualitative colormap
    # Set3 is good for up to 12 distinct colors
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
    cluster_colors = dict(zip(unique_clusters, colors))
    
    # Plot each city, colored by its cluster
    for cluster in unique_clusters:
        mask = df['cluster'] == cluster
        ax.scatter(df.loc[mask, 'X'], 
                  df.loc[mask, 'Y'],
                  c=[cluster_colors[cluster]], 
                  s=100,
                  label=f'Cluster {cluster}')
    
    # Plot the route connections
    for i in range(len(solution)-1):
        city1, city2 = solution[i], solution[i+1]
        x1, y1 = df.loc[city1, 'X'], df.loc[city1, 'Y']
        x2, y2 = df.loc[city2, 'X'], df.loc[city2, 'Y']
        
        # Calculate arrow properties based on figure size
        fig_width = fig.get_size_inches()[0]
        head_width = (ax.get_xlim()[1] - ax.get_xlim()[0]) / (fig_width * 5)
        head_length = head_width * 2
        
        # Draw arrow to show direction
        ax.arrow(x1, y1, 
                x2-x1, y2-y1,
                head_width=head_width,
                head_length=head_length,
                length_includes_head=True,
                color='gray',
                alpha=0.6)
    
    # Customize the plot
    ax.set_title(title, pad=20)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Adjust plot limits to include arrows and legend
    plt.margins(0.1)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', 
                   exist_ok=True)
        
        # Save figure with high DPI for quality
        plt.savefig(save_path, 
                   dpi=dpi, 
                   bbox_inches='tight',
                   facecolor='white',
                   edgecolor='none')
        
        # Close the figure to free memory
        plt.close(fig)
    else:
        # If not saving, display the plot
        plt.show()
        plt.close(fig)

# %%
# Initialize empty DataFrame to store results
results = pd.DataFrame(columns=['Instance', 'Cost', 'Route'])

# Iterate through all instances
for key in ctsp_instances.keys():
    # Drop coordinate columns
    df = ctsp_instances[key].drop(columns=['X', 'Y'])

    # Create and run solver
    solver = HybridGRASP_TabuCTSP(
        df=df,
        max_time=2,  # seconds
        grasp_alpha=1,
        grasp_iterations=100,
        elite_pool_size=5,
        tabu_tenure=(10, 20)
    )

    # Get solution and cost
    solution, cost = solver.solve()

    # Create a new row of data
    new_row = pd.DataFrame({
        'Instance': [key],
        'Cost': [cost],
        'Route': [solution]
    })

    # Append the new row to results DataFrame
    results = pd.concat([results, new_row], ignore_index=True)
    
    # Save route visualization
    visualize_route(
        ctsp_instances[key][['X', 'Y', 'cluster']],
        solution,
        title=f'{key}',
        save_path=f"output_random_search/{key}_random_search_run.png"
    )

# --- Salva resultados em um csv --- #       
results.to_csv('output_random_search/results_random_search_run.csv', index=False)
# %%
