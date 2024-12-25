import pygame
import matplotlib.pyplot as plt
import numpy as np
import GeneticCode
from SimulationConstants import *
import copy

def display_grid(original_grid, number_of_generations_alive, number_of_cells_created_in_all_of_history):
    """
    Displays and runs the simulation for Conway's Game of Life.

    Parameters:
        original_grid (list of lists): The initial state of the grid.
        number_of_generations_alive (int): Number of generations the best chromosome survived.
        number_of_cells_created_in_all_of_history (int): Total number of cells created by the chromosome.

    This function initializes a pygame window, handles user interaction, and updates the grid.
    """
    running = True
    pygame.init()
    if not pygame.font.get_init():
        pygame.font.init()

    # Set the screen dimensions
    screen_width = GRID_WIDTH * CELL_SIZE
    screen_height = GRID_HEIGHT * CELL_SIZE

    # Create the screen
    grid_screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Conway's Game of Life")
    grid = copy.deepcopy(original_grid)
    simulation_running = False
    generation = 0
    milliseconds_per_frame = MS_PER_FRAME_DEFAULT
    mouse_held = False  # Track if the mouse button is being held

    while running:
        grid_screen.fill(BLACK)  # Clear the screen
        title = "This Metoshalach lived for: " + str(number_of_generations_alive) + " ,and created " + str(
            number_of_cells_created_in_all_of_history) + " cells in it's lifetime"
        draw_grid(grid, grid_screen)  # Draw the grid
        display_title_with_background(grid_screen, title=title)
        menu_text = "PRESS 'r' for reset, PRESS DOWN ARROW to make it faster, UP ARROW for slower, and use mouse to click (WHEN PAUSED) to turn on and off cells"
        display_title_with_background(grid_screen, title=menu_text, font_size=16,  position=(2, 570), text_color=PINK, padding= 10)
        display_ms_per_frame(grid_screen, milliseconds_per_frame)
        draw_generation_count(generation, grid_screen)
        pygame.display.flip()  # Update the display

        keys = pygame.key.get_pressed()  # Checking pressed keys
        if keys[pygame.K_UP]:
            milliseconds_per_frame += 5
        elif keys[pygame.K_DOWN]:
            milliseconds_per_frame = max(1, milliseconds_per_frame - 5)  # 1 to ensure that we are not going negative

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:  # Check if a key was pressed
                if event.key == pygame.K_r:  # Press 'r' to reset the grid
                    grid = copy.deepcopy(original_grid)  # Reset the grid to the original
                    simulation_running = False  # Pause the simulation when resetting
                    generation = 0
                elif event.key == pygame.K_SPACE:  # Press space to toggle simulation
                    simulation_running = not simulation_running
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if not simulation_running:  # Only allow clicks when game is not running
                    mouse_held = True
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    cell_x = mouse_x // CELL_SIZE
                    cell_y = mouse_y // CELL_SIZE
                    if 0 <= cell_x < GRID_WIDTH and 0 <= cell_y < GRID_HEIGHT:
                        grid[cell_y][cell_x] = 1 - grid[cell_y][
                            cell_x]  # Switch the cell from dead to alive or alive to dead
            elif event.type == pygame.MOUSEBUTTONUP:  # Mouse button released
                mouse_held = False
            elif event.type == pygame.MOUSEMOTION and mouse_held and (not simulation_running):  # Mouse moved while button is held
                mouse_x, mouse_y = pygame.mouse.get_pos()
                cell_x = mouse_x // CELL_SIZE
                cell_y = mouse_y // CELL_SIZE
                if 0 <= cell_x < GRID_WIDTH and 0 <= cell_y < GRID_HEIGHT:
                    grid[cell_y][cell_x] = 1 if grid[cell_y][cell_x] == 0 else 0

        # Update the grid if simulation is running
        if simulation_running:
            grid = GeneticCode.update_grid(grid)  # Update the grid with your function
            generation += 1

        pygame.display.flip()  # Update the display
        pygame.time.wait(milliseconds_per_frame)
    pygame.quit()


# Draw the grid
def draw_grid(grid, grid_screen):
    """
    Draws the grid and its cells on the screen.

    Parameters:
        grid (list of lists): The grid representing the current simulation state.
        grid_screen (pygame.Surface): The pygame surface to draw the grid on.
    """
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            color = WHITE if grid[y][x] == 1 else BLACK
            pygame.draw.rect(grid_screen, color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Draw grid lines
    for x in range(0, WIDTH, CELL_SIZE):
        pygame.draw.line(grid_screen, GRAY, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, CELL_SIZE):
        pygame.draw.line(grid_screen, GRAY, (0, y), (WIDTH, y))


# Draw the current generation count
def draw_generation_count(generation, grid_screen, font_size=20, padding=10, text_color=GREEN,
                          background_color=(0, 0, 0), position=(100, 50)):
    """
    Displays the generation count on the screen.

    Parameters:
        generation (int): The current generation number.
        grid_screen (pygame.Surface): The pygame surface to draw on.
        font_size (int): Font size of the text.
        padding (int): Padding around the text.
        text_color (tuple): RGB color of the text.
        background_color (tuple): RGB color of the background rectangle.
        position (tuple): Position of the text on the screen.
    """
    font = pygame.font.Font(None, font_size)
    text = font.render(f"Generation: {generation}", True, text_color)

    # Define the position of the title and its background
    title_rect = text.get_rect()  # Get the size of the text
    title_rect.topleft = position  # Position of the title (top-left corner)

    # Create a white rectangle behind the text (padding around the text)
    background_rect = pygame.Rect(title_rect.x - padding, title_rect.y - padding,
                                  title_rect.width + padding * 2, title_rect.height + padding * 2)

    # Draw the white background (rectangle)
    pygame.draw.rect(grid_screen, background_color, background_rect)  # White color for the background

    # Blit (draw) the text on top of the rectangle
    grid_screen.blit(text, title_rect)


def display_title_with_background(grid_screen, title, font_size=20, padding=20, text_color=(255, 255, 255),
                                  background_color=(0, 0, 0), position=(10, 10)):
    """
        Displays a title with a colored background on the grid screen.

        :param grid_screen: The surface where the text and background will be drawn.
        :param title: The text to display.
        :param font_size: The size of the font for the text.
        :param padding: The padding around the text for the background.
        :param text_color: The color of the text (RGB tuple).
        :param background_color: The color of the background (RGB tuple).
        :param position: The (x, y) position to display the title.
        """
    # Define font and size
    font = pygame.font.Font(None, font_size)
    text = font.render(title, True, text_color)  # Render text (black color)

    # Define the position of the title and its background
    title_rect = text.get_rect()  # Get the size of the text
    title_rect.topleft = position  # Position of the title (top-left corner)

    # Create a white rectangle behind the text (padding around the text)
    background_rect = pygame.Rect(title_rect.x - padding, title_rect.y - padding,
                                  title_rect.width + padding * 2, title_rect.height + padding * 2)

    # Draw the white background (rectangle)
    pygame.draw.rect(grid_screen, background_color, background_rect)  # White color for the background

    # Blit (draw) the text on top of the rectangle
    grid_screen.blit(text, title_rect)


def display_ms_per_frame(grid_screen, milliseconds_per_frame, font_size=20, padding=10, background_color=(0, 0, 0),
                         position=(600, 50)):
    """
        Displays the frame time (milliseconds per frame) on the grid screen with color-coded text.

        :param grid_screen: The surface where the text will be displayed.
        :param milliseconds_per_frame: The frame time in milliseconds.
        :param font_size: The size of the font for the text.
        :param padding: The padding around the text for the background.
        :param background_color: The color of the background (RGB tuple).
        :param position: The (x, y) position to display the text.
        """
    color = WHITE
    if milliseconds_per_frame < 15:
        color = RED
    elif 15 < milliseconds_per_frame < 35:
        color = GREEN
    elif milliseconds_per_frame > 35:
        color = CYAN
    title = f'Milliseconds Per Frame = {milliseconds_per_frame}'
    display_title_with_background(grid_screen=grid_screen, title=title, text_color=color, position=position,
                                  font_size=font_size, padding=padding, background_color=background_color)


def get_best_chromosome_values_in_generation_off_springs_list(off_springs):
    """
        Finds the best chromosome in a list of offspring by evaluating fitness values.

        :param off_springs: A list of offspring chromosomes to evaluate.
        :return: The best chromosome, its generation lifetime, and its cells across history.
        """
    best_grid_index = 0
    best_chromosome_generation_life_time, best_chromosome_cells_across_history = GeneticCode.calculate_grid_fitness_values(
        off_springs[best_grid_index])
    best_fittness = best_chromosome_cells_across_history

    for grid_index in range(1, len(off_springs)):
        current_chromosome_generation_life_time, current_chromosome_cells_across_history = GeneticCode.calculate_grid_fitness_values(
            off_springs[grid_index])
        current_fittness = current_chromosome_cells_across_history

        if current_fittness > best_fittness:
            best_fittness = current_fittness
            best_chromosome_generation_life_time, best_chromosome_cells_across_history = current_chromosome_generation_life_time, current_chromosome_cells_across_history
            best_grid_index = grid_index
    return off_springs[best_grid_index], best_chromosome_generation_life_time, best_chromosome_cells_across_history


def get_best_chromosome_in_all_generations(best_off_springs_across_generations):
    """
        Finds the best chromosome across all generations.

        :param best_off_springs_across_generations: A list of the best chromosomes from each generation.
        :return: The best chromosome, its generation index, lifetime, and cell count.
        """
    GRID_INDEX, LIFE_TIME_INDEX, CELLS_ALIVE_INDEX = 0, 1, 2
    best_chromosome_generation = 0
    best_chromosome_grid = best_off_springs_across_generations[best_chromosome_generation][GRID_INDEX]
    best_chromosome_life_time = best_off_springs_across_generations[best_chromosome_generation][LIFE_TIME_INDEX]
    best_chromosome_cells_alive = best_off_springs_across_generations[best_chromosome_generation][CELLS_ALIVE_INDEX]

    for generation in range(1, len(best_off_springs_across_generations)):
        current_great_chromosome_in_off_spring = best_off_springs_across_generations[generation][GRID_INDEX]
        current_great_chromosome_life_time = best_off_springs_across_generations[generation][LIFE_TIME_INDEX]
        current_great_alive_cells_count = best_off_springs_across_generations[generation][CELLS_ALIVE_INDEX]

        if current_great_alive_cells_count > best_chromosome_cells_alive:
            best_chromosome_grid = current_great_chromosome_in_off_spring
            best_chromosome_generation = generation
            best_chromosome_life_time = current_great_chromosome_life_time
            best_chromosome_cells_alive = current_great_alive_cells_count

    return best_chromosome_grid, best_chromosome_generation, best_chromosome_life_time, best_chromosome_cells_alive


def display_best_chromosome_in_all_generations(off_springs_across_all_generations):
    """
        Displays information about the best chromosome across all generations.

        :param off_springs_across_all_generations: A list of best chromosomes from all generations.
        """
    best_chromosome, best_chromosomes_generation, best_chromosomes_generation_count, best_chromosomes_cells_alive_count = get_best_chromosome_in_all_generations(
        off_springs_across_all_generations)
    print(f'THE BEST OF THE BEST CHROMOSOME WAS IN GENERATION: {best_chromosomes_generation} \n')
    print(f'He lived a long life of: {best_chromosomes_generation_count} generations\n')
    print(f'And created {best_chromosomes_cells_alive_count} cells alive in his life_time \n')
    display_grid(best_chromosome, best_chromosomes_generation_count, best_chromosomes_cells_alive_count)


def display_generations_fitness_values_graphs(fitness_values_list_across_all_generations):
    """
     Displays graphs showing the fitness values of generations.

     :param fitness_values_list_across_all_generations: A list of fitness values across all generations.
     """
    LIFE_TIME_INDEX = 1
    ALIVE_CELLS_INDEX = 2
    # Prepare data for the graphs
    generations = np.arange(1, len(fitness_values_list_across_all_generations) + 1)

    # Extract the relevant data
    chromosomes_life_time_across_generations = [
        [chromosome[LIFE_TIME_INDEX] for chromosome in generation] for generation in
        fitness_values_list_across_all_generations
    ]
    chromosome_alive_cells_across_generations = [
        [chromosome[ALIVE_CELLS_INDEX] for chromosome in generation] for generation in
        fitness_values_list_across_all_generations
    ]
    # Call the two plotting functions
    plot_life_time_graph_of_each_generation(generations, chromosomes_life_time_across_generations)
    plot_alive_cells_all_across_history_for_each_generation(generations, chromosome_alive_cells_across_generations)


def plot_life_time_graph_of_each_generation(generations, chromosomes_life_time_across_generations):
    """
     Plots the lifetime statistics of chromosomes for each generation.

     :param generations: Array of generation numbers.
     :param chromosomes_life_time_across_generations: List of life times for each generation.
     """
    # Compute Best, Worst, and Average for each generation
    best_life_time = [max(gen) for gen in chromosomes_life_time_across_generations]
    worst_life_time = [min(gen) for gen in chromosomes_life_time_across_generations]
    avg_life_time = [np.mean(gen) for gen in chromosomes_life_time_across_generations]

    # Create the figure
    # plt.figure(figsize=(12, 6))
    plt.bar(generations, best_life_time, color='blue', alpha=0.7, label='Best')
    plt.bar(generations, worst_life_time, color='black', alpha=0.5, label='Worst')
    plt.plot(generations, avg_life_time, 'r-o', label='Average')

    plt.title('Life Time Generations')
    plt.xlabel('Generations')
    plt.ylabel('Lifetime')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def plot_alive_cells_all_across_history_for_each_generation(generations, chromosome_alive_cells_across_generations):
    """
    Plots the alive cell statistics for chromosomes across all generations.

    :param generations: Array of generation numbers.
    :param chromosome_alive_cells_across_generations: List of alive cell counts for each generation.
    """
    # Compute Best, Worst, and Average for each generation
    best_alive_cells = [max(gen) for gen in chromosome_alive_cells_across_generations]
    worst_alive_cells = [min(gen) for gen in chromosome_alive_cells_across_generations]
    avg_alive_cells = [np.mean(gen) for gen in chromosome_alive_cells_across_generations]

    # Create the figure
    # plt.figure(figsize=(12, 6))
    plt.bar(generations, best_alive_cells, color='blue', alpha=0.7, label='Best')
    plt.bar(generations, worst_alive_cells, color='black', alpha=0.5, label='Worst')
    plt.plot(generations, avg_alive_cells, 'r-o', label='Average')

    plt.title('Alive Cells Across All History')
    plt.xlabel('Generations')
    plt.ylabel('Alive Cells')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
