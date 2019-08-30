# Snake
A snake game built with python that also demonstrates use of machine learning through feed-forward neural nets and genetic algorithms that train a computer to play the game.

# Feed-forward neural net
The neural network has an input layer of 12 nodes, 2 hidden layers of 16 nodes, and an output layer of 4 nodes. It is trained on data generated from a human-written algorithm that plays enough games to generate a sufficiently large set of data for the network.

# Genetic algorithm
A work in progress! The genetic algorithm still has issues as it worsens after improving in the first few generations and then oscillates between improvement and decline.

Regardless, the genetic algorithm is the largest portion of the program, and was the most time consuming to write. After creating a variably sized initial population of neural nets (usually around 100), each is run through a fitness function and the top performers are bred to create the next generation. There are four mechanisms through which offspring are produced:

1. Mutation - weights in the neural net are adjusted according to some predefined value (for example, multiplying the weight by a value between .5 and 1.5). The number of weights adjusted correspond to the mutation rate, which stays close to 1%.

2. Midpoint Crossover - Two parents are bred together by rather taking whole layers from each parent to generate a child or by taking the first half of one parent's layer and the second half of the other parent's layer to generate the child's layer.

3. Elite - a parent is kept in the next generation unchanged.

4. Uniform Crossover - The form of breeding that makes least sense in hindsight. Every other weight from two parents are bred into one child (think: the odd weights from one parent and the even ones from the other).

# Other functionality
Of course, users can also play the game! A GUI is available, created with pygame. The GUI also gives options for visualizing performance of the feed-foward neural network and user-written algorithm.
