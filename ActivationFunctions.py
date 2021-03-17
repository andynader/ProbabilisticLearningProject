#!/usr/bin/env python
# coding: utf-8

# In[97]:

from tensorflow.keras.activations import elu, exponential, gelu, linear, relu, selu, sigmoid, softmax, softplus, swish, \
    tanh
from tensorflow.math import add, atan, cos, erf, maximum, minimum, sin, sqrt, subtract
from deap.gp import PrimitiveSet, PrimitiveTree, genGrow, genFull, compile, cxOnePoint, mutShrink, staticLimit
from deap.algorithms import eaSimple
from deap import creator, base, tools
from copy import deepcopy
from collections import Counter
import tensorflow as tf
import numpy as np
import multiprocessing
import operator
from scoop import futures

np.set_printoptions(precision=2)
from time import sleep


# # We define the Evolutionary Algorithm class, which encapsulates everything that has to do with the evolution. This class has access to a LangevinEBM object, which implements a function that trains an EBM using Langevin Dynamics and the other bells and whistles found in the OpenAI paper, and then returns a scalar score that is directly proportional to the fitness of an individual. 

# In[100]:


class EvolutionaryAlgorithm:

    def __init__(self, base_act_functions, base_operations, min_depth, max_depth, pop_size):

        # We first initialize the pset that contains our base 
        # building blocks.

        self.base_act_functions = base_act_functions
        self.base_operations = base_operations
        self.pset = self.initialize_pset(base_act_functions, base_operations)
        self.pop_size = pop_size

        # We specify that we are dealing with a maximization problem, and 
        # that our individual is a PrimitiveTree. We add a reference 
        # to the pset in "Individual", since it is used by some DEAP 
        # GP operators when modifying an individual.
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset)

        self.toolbox = base.Toolbox()

        # Our toolbox contains an "expr" function that calls genGrow with 
        # the pset as a parameter. genGrow also takes as parameter the 
        # minimum and maximum depth of the tree. We do not fix them
        # here, and we leave them as variables which can be changed when 
        # creating a population with the create_generation() function

        self.toolbox.register("expr", genGrow, pset=self.pset, min_=min_depth, max_=max_depth)

        # The "individual" function calls "expr()" on a "creator.Individual" object
        # and returns it. In other words, it simply creates a new Primitive tree 
        # from our pset
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)

        # We register the variation operators and a simple EA algorithm.

        self.toolbox.register("evaluate", self.get_ebm_fitness)
        self.toolbox.register("select", tools.selRoulette)
        self.toolbox.register("mate", cxOnePoint)
        self.toolbox.register("mutate", mutShrink)

        # We decorate the mating and mutation operator with a limit on the maximum depth.

        self.toolbox.decorate("mate", staticLimit(key=operator.attrgetter("height"), max_value=5))
        self.toolbox.decorate("mutate", staticLimit(key=operator.attrgetter("height"), max_value=5))

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        self.mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        self.mstats.register("avg", np.mean)
        self.mstats.register("std", np.std)
        self.mstats.register("min", np.min)
        self.mstats.register("max", np.max)

        # We also add a Hall of fame object, and we keep the 10 best individuals in it.

        self.hof = tools.HallOfFame(10)

        # Set DEAP to evolve functions in parallel 
        self.toolbox.register("map", futures.map)

    def initialize_pset(self, base_act_functions, base_operations):

        # Our desired functions are unary and we specify this when creating the pset
        pset = PrimitiveSet("main", 1)

        # Activation functions in a neural network have 
        # arity 1.
        for func in base_act_functions:
            pset.addPrimitive(func, 1)

        # The operations that we are considering, such as 
        # maximum, add or subtract, have arity 2.
        for op in base_operations:
            pset.addPrimitive(op, 2)

        pset.renameArguments(ARG0="x")

        return pset

    def adjust_activation_tree(self, activation_tree):
        copied_activation_tree = deepcopy(activation_tree)
        bias_pset = PrimitiveSet(name='bias_set', arity=1)
        for func in self.base_act_functions:
            bias_pset.addTerminal(terminal=func, name='terminal_' + func.__name__)
        for func in self.base_act_functions:
            bias_pset.addPrimitive(primitive=func, arity=1, name=func.__name__)
        primitives_dictionary = {}
        for bias_primitive in bias_pset.primitives[object]:
            for primitive in self.pset.primitives[object]:
                if bias_primitive.name == primitive.name:
                    primitives_dictionary[bias_primitive.name] = primitive
        tree_elements = np.vectorize(lambda x: x.name)
        num_terminals = Counter(tree_elements(copied_activation_tree))['ARG0']
        starting_index = 0
        two_literals = False
        for i in range(num_terminals):
            done = False
            while not done:
                bias_activation = PrimitiveTree(genFull(pset=bias_pset, min_=1, max_=1))
                if 'ARG0' not in tree_elements(bias_activation):
                    done = True
            for i in range(starting_index, len(copied_activation_tree)):
                if i < len(copied_activation_tree) - 1:
                    condition = copied_activation_tree[i].name == 'ARG0' and (
                                copied_activation_tree[i + 1].name == 'ARG0' or two_literals)
                else:
                    condition = copied_activation_tree[i].name == 'ARG0' and two_literals
                if condition:
                    if i < len(copied_activation_tree) - 1:
                        two_literals = copied_activation_tree[i + 1].name == 'ARG0'
                    replace_by_activation = np.random.choice([True, False], p=[0.8, 0.2])
                    if replace_by_activation:
                        starting_index = i + 2
                        activation_primitive = primitives_dictionary[bias_activation[0].name]
                        copied_activation_tree.insert(i, activation_primitive)
                    else:
                        starting_index = i + 1
                    break
        return copied_activation_tree

    def create_generation(self):
        generation = [self.toolbox.individual() for i in range(self.pop_size)]
        generation = [self.adjust_activation_tree(x) for x in generation]
        return generation

    # This will be replaced with a proper EBM training function later on. Make sure to take care of nan case 

    def get_ebm_fitness(self, individual):

        # Choose some very bad value in case of an error, not infinity to prevent DEAP from encountering an error
        sleep(0.05)
        f = compile(individual, self.pset)
        const = tf.constant([1.0, 2.0, 3.0])
        fitness = np.around(tf.reduce_sum(f(const)).numpy(), 2)
        if np.isnan(fitness) or np.isinf(fitness):
            fitness = -10 ** 3
        return (fitness,)

    def evolve_functions(self, num_generations):
        pop = self.create_generation()
        pop, log = eaSimple(pop, self.toolbox, ngen=num_generations, cxpb=0.8, mutpb=0.02, stats=self.mstats,
                            halloffame=self.hof, verbose=True)
        return pop

    def get_hof_inds(self):
        return ['Function:{}\tFitness:{}'.format(str(mvp), mvp.fitness.values[0]) for mvp in self.hof]


# In[101]:


base_functions = [elu, gelu, linear, relu, selu, sigmoid, softplus, swish, tanh, atan, cos, erf, sin, sqrt]
base_operations = [maximum, minimum, add, subtract]

# In[102]:


evo = EvolutionaryAlgorithm(base_functions, base_operations, min_depth=1, max_depth=3, pop_size=10)

# In[103]:

if __name__ == "__main__":
    pop = evo.evolve_functions(num_generations=10)
    print("\n\n")

    hof_info = evo.get_hof_inds()
    for mvp_info in hof_info:
        print(mvp_info)

# In[ ]:
