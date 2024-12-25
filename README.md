# 20581-Biological_Computation-TheOpenUniversityCourse

# Genetic Algorithm in the game of life 🧬🧬🧬 (MMN12): 
### Task:
#### Create an genetic algorithm that finds the best Methuselahs in the "Game of life" (The 'best' starting configurations that live the longest and create as many cells alive throughout their existence (until they become either static or repeat themselves):
![](https://github.com/idogut3/20581-Biological_Computation-TheOpenUniversityCourse/blob/main/images_and_gifs/GeneticEvolution_of_configuration1.gif)
![](https://github.com/idogut3/20581-Biological_Computation-TheOpenUniversityCourse/blob/main/images_and_gifs/GeneticEvolution_of_configuration2.gif)

### The Genetic Algorithm structure I created (type- Roulette Selection):
#### 1. Start with a random population of n chromosomes (n grids which represent each starting configuration of a different Methuselah).
#### 2. Compute the fitness of each chromosome in the population using the fittness function.
#### 3. Calculate the probabilities of choosing the i'th configuration to be a parent (of another descendent configuration).
#### 4. Choose 2n parents using the probabilities we calculated.
#### 5. Mate the parents in the pool repeat until n offspring are created in the new population.
#### 6. The new population becomes the current population forming the next generation.


$$P(i) = \frac{fitness(i)}{\sum_{k \, \text{(all solutions k)}} fitness(k)}$$
