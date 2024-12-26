from typing import Tuple, List, Any

import pygame

import GeneticCode
from GeneticCode import produce_grid_fitness_values, calculate_grid_fitness_list, normalize_probabilities, \
    produce_offsprings_list, create_new_population_grids
from Visuals import display_grid, display_best_chromosome_in_all_generations, \
    display_generations_fitness_values_graphs, get_best_chromosome_values_in_generation_off_springs_list, \
    get_best_chromosome_in_all_generations


def generation_life_process(population_grids, number_of_communities, parents_configuration_size) -> tuple[
    list, Any]:
    """
    Simulates the life process of one generation, including fitness evaluation, normalization,
    offspring production, and finding the best offspring in the generation.

    Parameters:
        population_grids (list): The current population of grids (chromosomes).
        number_of_communities (int): Number of communities (chromosomes) in the population.
        parents_configuration_size (int): The configuration size of each parent's grid.

    Returns:
        tuple: Contains the following:
            - List of offspring grids for the next generation.
            - Fitness values of the current population.
            - Best offspring and its fitness values.
    """
    grid_fitness_values = produce_grid_fitness_values(population_grids, number_of_communities)
    # print("print 3 \n\n\n\n ", grid_fitness_values[0][0] , "\n\n\n\n 2 : " ,grid_fitness_values[1][0] , " \nEqual = ", grid_fitness_values[0][0] == grid_fitness_values[1][0])
    # print(" \nprint 4 => " , grid_fitness_values[0][0] == grid_fitness_values[1][0])
    grid_fitness_tuple = calculate_grid_fitness_list(grid_fitness_values)
    # print("print 2 \n\n\n\n ", grid_fitness_tuple[0][0] , "\n\n\n\n 2 : " ,grid_fitness_tuple[1][0] , " \nEqual = ", grid_fitness_tuple[0][0] == grid_fitness_tuple[1][0])
    # print("\n print 5 => " , grid_fitness_tuple[0][0] == grid_fitness_tuple[1][0])
    normalize_fitness_list = normalize_probabilities(grid_fitness_tuple)
    # print("print 1 \n\n\n\n ", normalize_fitness_list[0][0] , "\n\n\n\n 2 : " ,normalize_fitness_list[1][0] , " \nEqual = ", normalize_fitness_list[0][0] == normalize_fitness_list[1][0])
    # print("\n print 6 => ", normalize_fitness_list[0][0] == normalize_fitness_list[1][0])
    off_springs = produce_offsprings_list(normalize_fitness_list, number_of_communities, parents_configuration_size)
    # best_off_spring_with_values = get_best_chromosome_values_in_generation_off_springs_list(off_springs)
    return off_springs, grid_fitness_values


def run_simulation(number_of_mating_generations=10, number_of_chromosomes=5, configuration_size_of_each_grid=7):
    """
    Runs a genetic algorithm simulation for a specified number of generations.

    Parameters:
        number_of_mating_generations (int, optional): Number of generations to simulate. Defaults to 10.
        number_of_chromosomes (int, optional): Number of chromosomes (communities) in each generation. Defaults to 5.
        configuration_size_of_each_grid (int, optional): The size of each grid (chromosome) in the population. Defaults to 7.

    The simulation performs the following steps:
        1. Initializes the population with random grids.
        2. Simulates the life process for the specified number of generations.
        3. Tracks and displays the fitness values of all generations.
        4. Identifies and displays the best chromosome across all generations.

    Outputs:
        - Graphs showing the fitness values for each generation.
        - Visualization of the best chromosome in all generations.
    """
    off_springs = create_new_population_grids(number_of_chromosomes, configuration_size_of_each_grid)
    #best_off_springs_each_generation = list()
    # great_off_spring_values = get_best_chromosome_values_in_generation_off_springs_list(off_springs)
    #best_off_springs_each_generation.append(great_off_spring_values)

    fitness_values_list_across_all_generations = list()
    for generation in range(number_of_mating_generations):
        print(f'Current mating generation {generation}')
        # Display Generations metoshalaues and generation number
        off_springs, current_fitness_values_list = generation_life_process(off_springs,
                                                                                            number_of_chromosomes,
                                                                                            configuration_size_of_each_grid)
        fitness_values_list_across_all_generations.append(current_fitness_values_list)
        # best_off_springs_each_generation.append(best_off_spring)

    display_generations_fitness_values_graphs(fitness_values_list_across_all_generations)
    best_chromosome_last_generation, best_chromosome_lifetime_last_generation, best_chromosome_cells_number_last_generation  = get_best_chromosome_values_in_generation_off_springs_list(off_springs)
    display_grid(best_chromosome_last_generation, best_chromosome_lifetime_last_generation, best_chromosome_cells_number_last_generation)

    pygame.quit()
