import random
import numpy as np

from SimulationConstants import *
from SimulationConstants import GRID_WIDTH


def compute_grid_unique_integer(grid):
    """
    Encodes a binary grid (1s and 0s) into a unique integer.
    Assumes the grid dimensions are fixed.
    """
    binary_string = ''.join(str(cell) for row in grid for cell in row)
    return int(binary_string, 2)


# def count_alive_cells(grid):
#     # Initialize the counter
#     alive_count = 0
#
#     # Iterate through each row of the grid
#     for row in grid:
#         # Add the number of alive cells (1's) in this row to the counter
#         alive_count += sum(row)
#
#     return alive_count

def count_alive_cells(grid):
    """
    Counts the total number of alive cells (value 1) in a binary grid.

    Parameters:
        grid (list of int): The binary grid.

    Returns:
        int: The total number of alive cells.
    """
    return np.sum(grid)

# def count_alive_cells(grid, device='cuda'):
#     grid = torch.tensor(grid, dtype=torch.int32, device=device)
#     return int(torch.sum(grid).item())  # Convert back to a Python int


# Count live neighbors of a cell
# def count_alive_neighbors(grid, x, y):
#     alive_neighbors = 0
#     for i in range(-1, 2):
#         for j in range(-1, 2):
#             if i == 0 and j == 0:
#                 continue  # Skip the current cell itself (x,y)
#             neighbor_x_pos, neighbor_y_pos = x + i, y + j
#             if 0 <= neighbor_x_pos < GRID_WIDTH and 0 <= neighbor_y_pos < GRID_HEIGHT:
#                 alive_neighbors += grid[neighbor_y_pos][neighbor_x_pos]
#     return alive_neighbors


# Initialize grid with dead cells and a community given in the middle
def initialize_grid_with_community_grid(community):
    """
    Initializes a grid with a given community placed at its center.

    Parameters:
        a 2D array: A binary grid representing the community.

    Returns:
        a 2D array: A new grid with the community at the center.

    Raises:
        ValueError: If the community size exceeds the grid size.
    """
    community_gird_size = len(community)

    if community_gird_size > GRID_WIDTH or community_gird_size > GRID_HEIGHT:
        raise ValueError("Community grid too big to fit in current grid")
        # Calculate the starting indices to place the m*m matrix in the center

    output_matrix = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]

    start_row = (GRID_WIDTH - community_gird_size) // 2
    start_col = (GRID_HEIGHT - community_gird_size) // 2

    # Embed the input matrix into the center of the output matrix
    for i in range(community_gird_size):
        for j in range(community_gird_size):
            output_matrix[start_row + i][start_col + j] = community[i][j]

    return output_matrix


# # Update grid based on Conway's rules
# def update_grid(grid):
#     new_grid = [[0] * GRID_WIDTH for _ in range(GRID_HEIGHT)]
#     for y in range(GRID_HEIGHT):
#         for x in range(GRID_WIDTH):
#             neighbors = count_alive_neighbors(grid, x, y)
#             if grid[y][x] == 1:  # Alive cell
#                 if neighbors < 2 or neighbors > 3:
#                     new_grid[y][x] = 0  # Dies
#                 else:
#                     new_grid[y][x] = 1  # Stays alive
#             else:  # Dead cell
#                 if neighbors == 3:
#                     new_grid[y][x] = 1  # Becomes alive
#     return new_grid

def update_grid(grid):
    """
     Updates the grid based on Conway's Game of Life rules.

     Parameters:
         grid (2D array): The current binary grid.

     """
    grid = np.array(grid)
    padded_grid = np.pad(grid, pad_width=1, mode='constant', constant_values=0)
    neighbor_count = (
        padded_grid[:-2, :-2] + padded_grid[:-2, 1:-1] + padded_grid[:-2, 2:] +
        padded_grid[1:-1, :-2] +                     0 + padded_grid[1:-1, 2:] +
        padded_grid[2:, :-2] + padded_grid[2:, 1:-1] + padded_grid[2:, 2:]
    )

    # Apply the rules of the Game of Life
    new_grid = ((grid == 1) & ((neighbor_count == 2) | (neighbor_count == 3))) | ((grid == 0) & (neighbor_count == 3))
    return new_grid.astype(int).tolist()


# def update_grid(grid, device='cuda'):
#     # Convert the grid to a PyTorch tensor and move it to the GPU
#     grid = torch.tensor(grid, dtype=torch.int32, device=device)
#
#     # Pad the grid to handle edge neighbors
#     padded_grid = torch.nn.functional.pad(grid, (1, 1, 1, 1), mode='constant', value=0)
#
#     # Compute the number of alive neighbors using slicing
#     neighbor_count = (
#             padded_grid[:-2, :-2] + padded_grid[:-2, 1:-1] + padded_grid[:-2, 2:] +
#             padded_grid[1:-1, :-2] + 0 + padded_grid[1:-1, 2:] +
#             padded_grid[2:, :-2] + padded_grid[2:, 1:-1] + padded_grid[2:, 2:]
#     )
#
#     # Apply the Game of Life rules
#     alive = (grid == 1) & ((neighbor_count == 2) | (neighbor_count == 3))
#     born = (grid == 0) & (neighbor_count == 3)
#     new_grid = alive | born
#
#     # Convert the result back to CPU and a list (optional)
#     return new_grid.int().cpu().tolist()


def create_first_gen_array(rows=5, columns=5):
    """
    Creates a random binary grid of the specified dimensions.

    Parameters:
        rows (int): Number of rows in the grid.
        columns (int): Number of columns in the grid.

    Returns:
        a 2D array with the rows and columns specified
    """
    # Generate a 2D grid with random 0s and 1s
    return [[random.randint(0, 1) for _ in range(rows)] for _ in range(columns)]


def create_new_population_grids(number_of_communities, original_community_grid_size) -> list:
    """
    Creates a list of population grids with random communities embedded in the center.

    Parameters:
        number_of_communities (int): Number of grids to generate.
        original_community_grid_size (int): Size of the community to embed in each grid.

    Returns:
        list: A list of grids with embedded communities.
    """
    population_grids = list()
    for community in range(number_of_communities):  # Creating community grids inside each population grid
        community_array = create_first_gen_array(original_community_grid_size, original_community_grid_size)
        community_grid_in_population = initialize_grid_with_community_grid(community_array)
        population_grids.append(community_grid_in_population)
        # display_grid(community_grid_in_population)
    return population_grids


def produce_grid_fitness_values(population_grids, number_of_grids=5):
    """
    Evaluates the fitness values of each grid in the population.

    Parameters:
        population_grids (list): The list of population grids.
        number_of_grids (int): Number of grids to evaluate.

    Returns:
        list: Fitness values including generations and alive cells for each grid.
    """
    grid_fitness_values_list = list()
    for grid in range(number_of_grids):
        life_time_generations, alive_cells_across_all_history = calculate_grid_fitness_values(
            population_grids[grid])

        grid_fitness_values_list.append([population_grids[grid], life_time_generations, alive_cells_across_all_history])
    return grid_fitness_values_list


def calculate_grid_fitness_values(population_grid):
    """
    Calculates the fitness values for a single grid over generations.

    Parameters:
        population_grid (a 2D array): The grid to evaluate.

    Returns:
        tuple: (Number of generations survived, Total alive cells across history).
    """
    MAX_GRIDS_UNIQUE_VALUES_SAVED = 10
    MAX_GENERATIONS_FOR_TESTING = 3000
    generation_count = 0
    number_of_alive_cells_across_all_history = 0  # Note: We do not include the original alive cells of the population

    cells_alive = count_alive_cells(population_grid)
    if cells_alive == 0:
        return 0, 0

    previous_unique_grid_values = list()
    grid = population_grid
    previous_unique_grid_values.append(compute_grid_unique_integer(grid))

    while generation_count < MAX_GENERATIONS_FOR_TESTING:
        if len(previous_unique_grid_values) > MAX_GRIDS_UNIQUE_VALUES_SAVED:
            previous_unique_grid_values.clear()

        grid = update_grid(grid)
        current_unique_value = compute_grid_unique_integer(grid)

        if current_unique_value in previous_unique_grid_values:
            break  # Because already we have been in the same configuration in a previous generation

        cells_alive = count_alive_cells(grid)
        number_of_alive_cells_across_all_history += cells_alive
        generation_count += 1
        previous_unique_grid_values.append(current_unique_value)

    return generation_count, number_of_alive_cells_across_all_history


def calculate_grid_fitness_list(grid_fitness_values_list) -> list:
    """
    Calculate the fitness values for a list of community grids based on alive cells.

    Parameters:
        grid_fitness_values_list (list): A list where each entry represents a grid and contains:
            - COMMUNITY_GRID_INDEX (int): Index of the grid.
            - COMMUNITY_INDEX_GENERATION (int): Number of generations (unused in this calculation).
            - COMMUNITY_INDEX_CELLS (int): Number of alive cells across the grid's history.

    Returns:
        list: A list of tuples with the grid index and its fitness value.
    """
    # sum_of_all_communities_generations = 0
    sum_of_all_communities_alive_cells_across_history = 0
    COMMUNITY_GRID_INDEX, COMMUNITY_INDEX_GENERATION, COMMUNITY_INDEX_CELLS = 0, 1, 2

    for grid in grid_fitness_values_list:
        # sum_of_all_communities_generations += grid[COMMUNITY_INDEX_GENERATION]
        sum_of_all_communities_alive_cells_across_history += grid[COMMUNITY_INDEX_CELLS]

    fitness_values_list = list()
    if sum_of_all_communities_alive_cells_across_history == 0:
        for grid in grid_fitness_values_list:
            fitness_values_list.append((grid[COMMUNITY_GRID_INDEX], 0))
        return fitness_values_list

    for grid in grid_fitness_values_list:
        # current_community_generations = grid[COMMUNITY_INDEX_GENERATION]
        current_community_cells = grid[COMMUNITY_INDEX_CELLS]
        # print("Suka grid -> \n\n\n", grid)

        # current_community_generation_fittness = current_community_generations / sum_of_all_communities_generations
        current_community_cells_fittness =  current_community_cells

        # current_community_fittness = (current_community_generation_fittness + current_community_cells_fittness) / 2  # AVG
        current_community_fittness = current_community_cells_fittness
        fitness_values_list.append((grid[COMMUNITY_GRID_INDEX], current_community_fittness))

    return fitness_values_list


def normalize_probabilities(tuples_grid_fittness_list):
    """
    Normalize the probabilities in a list of tuples where probabilities are at index 1.

    Parameters:
        tuples_grid_fittness_list (list of tuples): A list of tuples, where each tuple contains a value
                                         and its corresponding tuple_grid_fittness (e.g., (value, tuple_grid_fittness)).

    Returns:
        list of tuples: A list of tuples with normalized probabilities.
    """
    # Extract the probabilities from the tuples
    total_sum = sum(tuple_index[1] for tuple_index in tuples_grid_fittness_list)

    if total_sum == 0:
        raise ValueError("Cannot normalize a list with a total sum of 0.")

    if not (0.99 <= total_sum <= 1.01):  # Allowing for floating-point tolerance
        return [(tuple_grid_fittness[0], tuple_grid_fittness[1] / total_sum) for tuple_grid_fittness in
                tuples_grid_fittness_list]

    return tuples_grid_fittness_list  # No need to normalise, they are already


def choose_n_based_on_probability(tuple_list, n):
    """
    Choose 'n' elements from a list of tuples based on the provided probabilities.

    Parameters:
        tuple_list (list of tuples): A list of tuples where each tuple contains:
            - tuple[0]: The value to be chosen.
            - tuple[1]: Probability of being chosen (must sum to 1).
        n (int): The number of elements to choose.

    Returns:
        list: A list of 'n' chosen values based on the probabilities.
    """
    VALUES_INDEX = 0
    PROBABILITIES_INDEX = 1
    # Extract values and probabilities
    values = list()
    probabilities = list()

    for tuple_index in range(len(tuple_list)):
        values.append(tuple_list[tuple_index][VALUES_INDEX])
        probabilities.append(tuple_list[tuple_index][PROBABILITIES_INDEX])
    # print("Definitely narcissism ", values[0] == values[1])
    print("Probabilities \n\n\n ", probabilities)
    if any(prob < 0 for prob in probabilities):
        raise ValueError("Probabilities must be non-negative.")
    if len(values) != len(probabilities):
        raise ValueError("The number of values must match the number of probabilities.")

    # Ensure probabilities sum to 1 (normalization in case of floating point errors)
    total_prob = sum(probabilities)
    if not (0.99 <= total_prob <= 1.01):  # Allowing for floating point tolerance
        probabilities = [prob / total_prob for prob in probabilities]

    # Use random.choices to pick 'n' elements based on the probabilities
    chosen_values = random.choices(values, weights=probabilities, k=n)

    return chosen_values


def mutate(number):
    """
    Mutate a given value by flipping its binary state.

    Parameters:
        number (int): The binary value (0 or 1) to mutate.

    Returns:
        int: The mutated binary value.
    """
    return 1 - number


def mate(parent1, parent2, parents_configuration_size):
    """
    Create offspring by mating two parents with a chance of mutation.

    Parameters:
        parent1 (list): The first parent's 2D grid.
        parent2 (list): The second parent's 2D grid.
        parents_configuration_size (int): The size of the parent configuration.

    Returns:
        list: A 2D grid representing the offspring.
    """
    # print("p1: \n\n", parent1)
    # print("p2: \n\n", parent2)
    MUTATION_PROBABILITY = 0.02
    offspring = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]

    start_row = (GRID_WIDTH - parents_configuration_size) // 2
    start_column = (GRID_HEIGHT - parents_configuration_size) // 2

    # Embed the input matrix into the center of the output matrix
    for i in range(start_row, parents_configuration_size + start_row):
        for j in range(start_column, parents_configuration_size + start_column):
            current_chosen_parent_matrix = random.choice([parent1, parent2])
            offspring[i][j] = current_chosen_parent_matrix[i][j]
            if random.random() < MUTATION_PROBABILITY:
                offspring[i][j] = mutate(offspring[i][j])

    return offspring


def mate_parents_in_pool(mating_pool, parents_configuration_size):
    """
    Mate consecutive parents from the mating pool. Parent[i] mates with parent[i+1].

    Parameters:
        mating_pool (list): A list of parents selected based on probabilities.
        :param parents_configuration_size: The configuration size of each parent first generation 2D array (community) in the grid
        :param mating_pool: a list of all parents (grid matrices 2D)

    Returns:
        list: A list of offspring produced by mating the parents in pairs.
    """
    offsprings = []
    # Iterate over the mating pool in steps of 2 to mate each consecutive pair
    for i in range(0, len(mating_pool) - 1, 2):
        parent1 = mating_pool[i]
        parent2 = mating_pool[i + 1]
        # if parent1 == parent2:
        #     print("SOMETHING IS SUS")

        child1 = mate(parent1, parent2, parents_configuration_size)

        offsprings.append(child1)

    return offsprings


def produce_offsprings_list(normalize_fitness_list, number_of_descendants_desired, parents_configuration_size) -> list:
    """
    Generate a list of offspring by mating selected parents.

    Parameters:
        normalize_fitness_list (list): A list of normalized fitness tuples.
        number_of_descendants_desired (int): The number of offspring to produce.
        parents_configuration_size (int): The size of the parent configuration.

    Returns:
        list: A list of offspring grids.
    """
    number_of_parents = number_of_descendants_desired * 2
    mating_pool = choose_n_based_on_probability(normalize_fitness_list, number_of_parents)
    off_springs = mate_parents_in_pool(mating_pool, parents_configuration_size)
    return off_springs
