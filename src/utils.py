import numpy as np


def calculate_objective_function(
    tour: list,
    cities_distance_matrix: np.ndarray,
    printable_sequence: bool = False,
) -> float:
    """ Calculate the Objective Function (OF) of a given tour. """
    n_cities = len(tour)
    # ---
    lst_tour_distances = []
    for travel in range(n_cities):
        current_city = tour[travel]
        next_city = tour[
            (travel + 1) % n_cities
        ]
        # ---
        distance_between_cities = cities_distance_matrix[current_city][
            next_city
        ]
        lst_tour_distances.append(distance_between_cities)
    return sum(lst_tour_distances)
