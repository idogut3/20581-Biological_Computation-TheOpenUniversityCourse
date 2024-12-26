import LifeGameSimulation

if __name__ == '__main__':
     generations = int(input("Please write how many mating generations do u want it to be: (INT PLEASE)\n"))
     number_of_chromosomes = int(input("Please write how many chromosomes in each generation do u want: (INT PLEASE)\n"))
     configuration_size_of_each_grid = int(input("Please write the configuration size of each grid when starting example 7 will do 7X7: (INT PLEASE)\n"))
     LifeGameSimulation.run_simulation(generations,number_of_chromosomes, configuration_size_of_each_grid )

