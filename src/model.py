
import random
import re

import numpy as np
import pandas as pd
from src.structure.paths import ProjectPathsStructure

from .utils import calculate_objective_function


class InstanceReader:
    
        @property
        def _get_instances_folder(self):
            ''' Gets the path to the instances folder.'''
            
            path = ProjectPathsStructure.Folders.CTSP_INSTANCES
            
            if not path.exists():
                raise FileNotFoundError(f"Folder {path} not found.")
            return path

        @staticmethod
        def _sort(item_list):
            def sort_items(item):
                parts = re.split(r'(\d+)', item)
                parts = [int(part) if part.isdigit() else part for part in parts]
                return parts
    
            return sorted(item_list, key=sort_items)
    
        @property
        def create_instances_path_map(self):
            ''' Creates a dictionary with all instances. '''

            folder_instances_path = self._get_instances_folder

            files = []
            for file in folder_instances_path.iterdir():
                if file.exists() and str(file).endswith('.csv'):
                    files.append(file.stem)

            instances_map = {}
            for file_name in self._sort(files):
                file_path = folder_instances_path / file_name
                instances_map[file_name] = file_path.with_suffix('.csv')
            return instances_map
        
        def read_instance(self, instance_name):
            ''' Reads the instances file (.csv). '''

            instance_path = self.create_instances_path_map[instance_name]
            return pd.read_csv(instance_path)
        
        @property
        def create_instances_map(self):
            ''' Reads all instances files (.csv). '''

            instances = {}
            for instance_name in self.create_instances_path_map.keys():
                instances[instance_name] = self.read_instance(instance_name).sort_index()
            return instances

class GRASP:
    """Solves the Traveling Salesman Problem (TSP) using the GRASP method."""

    def __init__(self, instance: pd.DataFrame | np.ndarray) -> None:
        self.instance = instance.to_numpy() if isinstance(instance, pd.DataFrame) else instance

    def create_min_max_distance_matrix(self):
        """ Creates the minimum and maximum distance matrix for each city. """

        n_cities = len(self.instance)
        for city in range(n_cities):
            cities_distance_vector = self.instance[city]
            non_zero_cities_distance_vector = cities_distance_vector[cities_distance_vector > 0]
            # ---
            min_distance = non_zero_cities_distance_vector.min()
            max_distance = non_zero_cities_distance_vector.max()
            yield (min_distance, max_distance)

    def create_restricted_candidate_list(self, L: float):
        """ Calcultes the Restricted Candidate List (RCL) for each city based on the minimum and maximum distances."""

        min_distances, max_distances = zip(*self.create_min_max_distance_matrix())
        # ---
        dict_rcl = {}
        for node, _ in enumerate(self.instance):
            min_distance, max_distance = min_distances[node], max_distances[node]
            # ---
            dict_rcl[f"node_{node}"] = {
                "rcl": "Random" if max_distance > L else "Greedy",
                "initial_min_distance": min_distance,
                "initial_max_distance": max_distance,
            }
        print(dict_rcl)
        return dict_rcl
    
    def calculate_L(self, alpha: float):
        """ Calculates the L parameter for the GRASP method. """

        distance_matrix_min_distance, distance_matrix_max_distance = self.instance.min(), self.instance.max()
        return (
            distance_matrix_min_distance + alpha * (distance_matrix_max_distance - distance_matrix_min_distance)
            )
    
    def identify_cities_with_the_smallest_distance(self) -> list:
        """ Identify the two cities with the smallest distance between them for using as the first part of the tour. """

        mask = self.instance > 0
        min_distance = np.min(self.instance[mask])
        min_distance_node = np.argwhere(self.instance == min_distance)
        return min_distance_node[0].tolist()

    def solve(self, alpha: float):
        """ Apply the GRASP method to the TSP problem. """

        # --- Initial Parameters --- #
        matrix, n_cities = self.instance, len(self.instance)
        L = self.calculate_L(alpha)
        RCL = self.create_restricted_candidate_list(L)
        available_cities = [_ for _ in range(n_cities)]
        # ---
        origin_city = None
        destination_city = None
        # ---
        visited_cities = []
        for node in range(n_cities):
            if node == 0:
                closest_cities = self.identify_cities_with_the_smallest_distance()
                origin_city, destination_city = closest_cities[0], closest_cities[1]
                visited_cities.append(origin_city)
                available_cities.remove(origin_city)
                # ---
                matrix[origin_city, :] = np.nan
                matrix[:, origin_city] = np.nan
            # ---
            elif node > 0:
                origin_city = destination_city
                # ---
                node_name = f"node_{origin_city}"
                origin_city_rcl = RCL[node_name]["rcl"]
                # ---
                if origin_city_rcl == "Greedy":
                    vector = np.array(matrix[origin_city, :])
                    mask = (vector != 0) & (~np.isnan(vector))
                    masked_vector = np.array(vector[mask])
                    # ---
                    if masked_vector.size != 0:
                        min_distance = np.min(masked_vector)
                        destination_city = int(np.argwhere(vector == min_distance)[0][0])
                        # ---
                        visited_cities.append(origin_city)
                        available_cities.remove(origin_city)
                        # ---
                        matrix[origin_city, :] = np.nan; matrix[:, origin_city] = np.nan
                    elif masked_vector.size == 0:
                        visited_cities.append(destination_city)
                        available_cities.remove(destination_city)
                # ---
                elif origin_city_rcl == "Random":
                    origin_city = destination_city
                    visited_cities.append(origin_city)
                    available_cities.remove(origin_city)
                    if len(available_cities) > 0:
                        destination_city = random.choice(available_cities)
                    elif len(available_cities) == 0:
                        destination_city = visited_cities[0]
                    # ---
                    matrix[origin_city, :] = np.nan
                    matrix[:, origin_city] = np.nan
        # ---
        initial_city = visited_cities[0] # identify the first city visited so that the salesman returns at the end of the tour
        S = visited_cities + [initial_city] # add the first city visited at the end of the tour
        OF = calculate_objective_function(S, self.instance) # calculate the OF (total distance) of the solution
        return S, OF, L, [v['rcl'] for k, v in RCL.items()]
        
        @classmethod
        def apply_grasp_with_local_search_method(cls, cities_distance_matrix: np.ndarray, alpha: float, time_criteria: int = None, printable_sequence: bool = False):
            """ Apply the GRASP method to the TSP problem. """
            obj = TravellingSalesmanProblem()
            # ---
            S, OF, L, RCL = cls.apply_grasp_method(cities_distance_matrix, alpha=alpha, printable_sequence=printable_sequence)
            # ---
            S, OF = obj.LocalSearch.apply_local_search(S, cities_distance_matrix, time_criteria=time_criteria, strategy='best_improvement')
            return S, OF, L, RCL

        @classmethod
        def run(cls, local_sarch_criteria=None, save_results=False, return_results=False, printable_sequence=False):
            """ Executes all the steps to solve the TSP problem with Random Search. """
            obj = TravellingSalesmanProblem() # create an instance of the class
            # ---
            instances = obj.create_instances_map # load all instances to process
            #--- Criteria: each alpha value is the iteration from 0 to 1, adding .01 in each iteration ---#
            start_point = .0
            lst_alphas = []
            for _ in range(100):
                start_point = round(start_point + .01, 2)
                lst_alphas.append(start_point)
            lst_alphas = list(reversed(lst_alphas))
            # ---
            results = {} # random search results
            for instance_name, instance_path in instances.items(): # iterate over each instance and calculate the best solution using Random Search method
                tsp_data = obj.read_tsp_file(instance_path) # read the instance file as DataFrame
                tsp_data.drop(columns=['X','Y'], inplace=True) # drop the columns 'X' and 'Y' (not necessary for the calculations)
                tsp_distance = tsp_data.to_numpy() # convert the DataFrame to a numpy array
                #--- GRASP Apply ---#
                s_best = None
                of_best = None
                l_best = None
                rcl_best = None
                for alpha in lst_alphas: # 
                    S, OF, L, RCL = obj.GRASP.apply_grasp_method(tsp_distance, alpha=alpha, printable_sequence=printable_sequence) # apply the GRASP Method
                    if of_best is None or of_best < of_best:
                        s_best = S
                        of_best = OF
                        l_best = L
                        rcl_best = RCL
                # --- GRASP with Local Search Apply --- #
                s_best, of_best = obj.LocalSearch.apply_local_search(initial_tour=s_best, cities_distance_matrix=tsp_distance, time_criteria=local_sarch_criteria, strategy='best_improvement')
                # --- Print GRASP/Local Search Results --- #
                print(f"Instance: {instance_name} | OF_best: {round(of_best, 4)}, S_best: {s_best}")
                # ---
                results |= {instance_name: {"best_OF": round(of_best, 4), "best_SOL": s_best, "L_parameter": l_best, "RCL": rcl_best}}
            # ---
            df = (
                    pd.DataFrame(results)
                        .T
                        .reset_index()
                        .rename({"index": "instance"}, axis=1)
                        .assign(method="GRASP")
                        [['instance', 'method', 'best_OF', 'best_SOL', 'L_parameter', 'RCL']]
                )
            if save_results:
                df.drop(columns=['L_parameter', 'RCL']).to_csv(f"results_grasp_with_local_search.csv", index=False)
            if return_results:
                return df

class TSPSolver:
    """Solves the Traveling Salesman Problem (TSP)."""
    
    def __init__(self, instance: pd.DataFrame, cluster: int) -> None:
        self.instance = instance
        self.cluster = cluster

    @property
    def _adjust_cluster(self) -> pd.DataFrame:
        """Adjusts the cluster of the instance."""
        instance = self.instance.drop(["X", "Y"], axis=1).copy()
        self.instance = instance[instance["cluster"] == self.cluster].drop("cluster", axis=1)

    @classmethod
    def _check_tsp_constraints(cls, tsp_complete_tour: list, available_cities: list) -> list:
        ''' Checks the constraints of the TSP problem. '''
        
        # Checks whether the salesman visited all cities or not.
        if not set(tsp_complete_tour) == set(available_cities):
            raise ValueError("Salesman didn't visit all cities yet.")
        
        # Checks whether the salesman visited each city only once.
        if not len(tsp_complete_tour[:-1]) == len(set(tsp_complete_tour)):
            raise ValueError("Salesman didn't visit each city once.")
        
        # Checks whether the salesman returned to origin city.
        if not tsp_complete_tour[0] == tsp_complete_tour[-1]:
            raise ValueError("The salesman didn't return to the origin city.")
        
        return tsp_complete_tour
    
    def t(self):
        self._adjust_cluster

        return self.instance

    
