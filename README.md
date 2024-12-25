# 20581-Biological_Computation-TheOpenUniversityCourse

## Genetic Algorithm in the game of life 🧬🧬🧬 (MMN12): 
### Task:
#### Create an genetic algorithm that finds the best Methuselahs in the "Game of life" (The 'best' starting configurations that live the longest and create as many cells alive throughout their existence (until they become either static or repeat themselves):
![](https://github.com/idogut3/20581-Biological_Computation-TheOpenUniversityCourse/blob/main/images_and_gifs/GeneticEvolution_of_configuration1.gif)
![](https://github.com/idogut3/20581-Biological_Computation-TheOpenUniversityCourse/blob/main/images_and_gifs/GeneticEvolution_of_configuration2.gif)

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
### 1. How did you implement the cellsAlive() function? How did you figure out when to stop? How did you define whats a repetitive state (and follows from that how did you define a static state?)?
#### Well, the cellsAlive() function is a tricky function, you need to count the current cells alive in the grid but also have to stop when you reach a repetitive state.
<p> 
The first idea I though about is to just keep a list of my last n grids representing my previous n-genertaions though that is an expensive approach in terms of Space complexity and Time Complexity assuming I would need a list of n-previous grids the Space complexity would be **Θ(n³)** and in terms of Time complexity, any new generation I would need to check the current grid is equal to any of my last n grids, causing me to need a **Θ(n³)** function that checks each new generation that it was not previously equal to any of my last n-generations.
#### That caused me to thing 
</p>
---
<p>
The Second idea I had was, to distinguish each repetitive state, by defining a variable that would indicate if there was a 'big' change, in the amount of cells in the previous n generations, and if there weren't any then it would declare the state as repetitive and exit the aliveCells function **Θ(1)** in space complexity and time complexity . This unfortunately was insufficient in terms of distinguishing the repetitive states and the ones that are not and caused false-positives (states were declared as repetitive when they were in fact not.
</p>
---
<p>
The last idea I had was the best one that worked perfectly, for anyone not familiar with hash algorithms, here is a short description of them:
Hash algorithms are mathematical functions that take input data of any size and produce a fixed-size string of characters, typically a sequence of numbers and letters. This output, known as a hash or digest, is unique to the input data, meaning even small changes to the input will produce a significantly different hash. Hash algorithms are widely used in cryptography, data integrity checks, and password storage due to their ability to create a secure, irreversible representation of data.
</p>
<p>
Based on that idea in mind, I used a list that contained the previous-n hashes calculated by my hash function (which I will later explain what I used) of the each previous generation-grid.
Continuously, each generation, I would calculate my current generation-grid hash, and check if the hash is the same as any other hash I stored in the list, later on (if it wasn't in the list, i.e. not repetitive) add it to the previous previous-n hashes and count the current alive cells. This solution is sufficient with **Θ(n)** space complexity and time complexity.
</p>

### 2. How is the mating Process of each grid looks like?

