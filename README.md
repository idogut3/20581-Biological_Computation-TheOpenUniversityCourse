# 20581-Biological_Computation-TheOpenUniversityCourse

## Genetic Algorithm in the game of life 🧬🧬🧬 (MMN12): 
### Task:
Create an genetic algorithm that finds the best Methuselahs in the "Game of life" (The 'best' starting configurations that live the longest and create as many cells alive throughout their existence (until they become either static or repeat themselves):

![](https://github.com/idogut3/20581-Biological_Computation-TheOpenUniversityCourse/blob/main/images_and_gifs/mmn12/GeneticEvolution_of_configuration1.gif)
![](https://github.com/idogut3/20581-Biological_Computation-TheOpenUniversityCourse/blob/main/images_and_gifs/mmn12/GeneticEvolution_of_configuration2.gif)

## The Genetic Algorithm structure I created (type- Roulette Selection):
### 1. Start with a random population of n chromosomes (n grids which represent each starting configuration of a different Methuselah).
### 2. Compute the fitness of each chromosome in the population using the fittness function.
#### In our case I defined the fitness function of each Methuselah i to be:
#### $$fitness(i) = \sum_{generation}^\text{(all generations)} cellsAlive(generation)$$
### 3. Calculate the probabilities of choosing the i'th configuration to be a parent (of another descendent configuration), using a normalisation function
#### In our case I defined the probability of grid(i) to become a parent as follows:    
#### $$P(i) = \frac{fitness(i)}{\sum_{k \, \text{(all solutions k)}} fitness(k)}$$

### 4. Choose 2n parents using the probabilities we calculated.
### 5. Mate the parents in the pool repeat until n offspring are created in the new population.
### 6. The new population becomes the current population forming the next generation.
### 7. Repeat k times (until, asked for or reached a convergence).

## Questions you might be thinking about:
### 1. How did you implement the cellsAlive() function? How did you figure out when to stop? How did you define whats a repetitive state (and follows from that how did you define a static state?)? 🦠🦠🦠
#### Well, the cellsAlive() function is a tricky function, you need to count the current cells alive in the grid but also have to stop when you reach a repetitive state. 
<p> 
The first idea I thought about is to just keep a list of my last n grids representing my previous n-genertaions though that is an expensive approach in terms of Space complexity and Time Complexity assuming I would need a list of n-previous grids the Space complexity would be $$Θ(n³)$$ and in terms of Time complexity, any new generation I would need to check the current grid is equal to any of my last n grids, causing me to need a $$Θ(n³)$$ function that checks each new generation that it was not previously equal to any of my last n-generations. The expensiveness of the algorithm caused me to recoil from using it, so I tried to come up with a better idea.
</p>
---
<p>
The Second idea I had was, to distinguish each repetitive state, by defining a variable that would indicate if there was a 'big' change, in the amount of cells in the previous n generations, and if there weren't any then it would declare the state as repetitive and exit the aliveCells function $$Θ(1)$$ in space complexity and time complexity . This unfortunately was insufficient in terms of distinguishing the repetitive states and the ones that are not and caused false-positives (states were declared as repetitive when they were in fact not).
</p>
---
<p>
The last idea I had was the best one that worked perfectly, for anyone not familiar with hash algorithms, here is a short description of them:
Hash algorithms are mathematical functions that take input data of any size and produce a fixed-size string of characters, typically a sequence of numbers and letters. This output, known as a hash or digest, is unique to the input data, meaning even small changes to the input will produce a significantly different hash. Hash algorithms are widely used in cryptography, data integrity checks, and password storage due to their ability to create a secure, irreversible representation of data.
</p>
<p>
Based on that idea in mind, I used a list that contained the previous-n hashes calculated by my hash function (which later on you can read what I used) of the each previous generation-grid.
Continuously, each generation, I would calculate my current generation-grid hash, and check if the hash is the same as any other hash I stored in the list, later on (if it wasn't in the list, i.e. not repetitive) add it to the previous previous-n hashes and count the current alive cells. This solution is sufficient with $$Θ(n)$$ space complexity and time complexity.
</p>
<p>
I thought about many hash functions that might work, such as: SHA256 or maybe even the checksum function (like in linux) - but eventually I decided to implement a different one, a one that has it's collision number at the lowest (A case were 2 different imputs create the same hash value).
</p>
Here it is:

```python
def compute_grid_unique_integer(grid):
    """
    Encodes a binary grid (1s and 0s) into a unique integer.
    Assumes the grid dimensions are fixed.
    """
    binary_string = ''.join(str(cell) for row in grid for cell in row)
    return int(binary_string, 2)
```

### 2. How is the mating Process of each grid looks like?  🦠👩‍⚕🧬🩺💉
<p>
Each grid-offspring is created using two parents. Each parent contributes equally to the offspring, with both having an equal chance of influencing the child's traits. This ensures that the genetic diversity from both parents is preserved and balanced, giving both parents an equal opportunity to pass on their beneficial characteristics to the next generation.
</p>
<p>
The process as I chose to implement it is as follows: 
1. Iterate through only the relevent cells (meaning we produce a "starting configuration" of the offspring, the same size of the parents).
2. For each cell position in the grid (i,j) we choose a parent randomly.
3. We insert the chosen parent at indexes (i,j) to the same cell position in the offsprings grid. offspring[i][j] = chosen_parent[i][j] .
4. At the end we introduce a mutation 🦠🦠🦠 with a certain probability, where a cell can switch between alive and dead states.
</p>
For those interested in the code here it is:

```python

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


def mutate(number):
    """
    Mutate a given value by flipping its binary state.

    Parameters:
        number (int): The binary value (0 or 1) to mutate.

    Returns:
        int: The mutated binary value.
    """
    return 1 - number
```

#### Results:
##### 30 Generations
![](https://github.com/idogut3/20581-Biological_Computation-TheOpenUniversityCourse/blob/main/images_and_gifs/mmn12/aliveCells30.png)
![](https://github.com/idogut3/20581-Biological_Computation-TheOpenUniversityCourse/blob/main/images_and_gifs/mmn12/lifetime30.png)

##### 100 Generations
![](https://github.com/idogut3/20581-Biological_Computation-TheOpenUniversityCourse/blob/main/images_and_gifs/mmn12/aliveCells100.png)
![](https://github.com/idogut3/20581-Biological_Computation-TheOpenUniversityCourse/blob/main/images_and_gifs/mmn12/lifetime100.png)


