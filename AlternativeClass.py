#!/usr/bin/env python
# coding: utf-8

# In[1]:

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
import operator
from pathos.multiprocessing import ProcessPool

np.set_printoptions(precision=2)
from time import sleep


# In[4]:


class EvolutionaryAlgorithm:
    def __init__(self, base_act_functions, base_operations, min_depth, max_depth, pop_size, n_parallel_nodes):

        # We first initialize the pset that contains our base 
        # building blocks.

        self.base_act_functions = base_act_functions
        self.base_operations = base_operations
        self.pset = self.initialize_pset(base_act_functions, base_operations)
        self.pop_size = pop_size
        self.min_depth = min_depth
        self.max_depth = max_depth

        # We specify that we are dealing with a maximization problem, and 
        # we specify that our Individual is a string representation of 
        # the activation function tree. We chose 
        # a string representation and not the primitive tree itself because 
        # the primitive tree contains tensorflow operations, which can't be pickled
        # and thus can't be easily paralellized. The string representation can 
        # be paralellized, and we can easily obtain the primitive tree from it 
        # using DEAP's PrimitveTree.from_string function. 
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", str, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()

        # The "individual" function simply calls generate_tree_string_representation
        # to get a string representation of a random tree, and then places it 
        # in a "creator.Individual" container that has a fitness attribute.
        self.toolbox.register("individual", tools.initIterate, creator.Individual,
                              self.generate_tree_string_representation)

        # We define variational operators on the PrimitiveTree representations,
        # and we define static limits on those variational operators. The actual
        # operations used by the evolutionary algorithm will operate on the 
        # string representations. They will convert those string representations 
        # to PrimitiveTree objects, apply the operators on them, and return strings.

        self.toolbox.register("primitive_tree_crossover", cxOnePoint)
        self.toolbox.register("primitive_tree_mutation", mutShrink)

        # We decorate the variation operators with a limit on the maximum depth.

        self.toolbox.decorate("primitive_tree_crossover", staticLimit(key=operator.attrgetter("height"), max_value=5))
        self.toolbox.decorate("primitive_tree_mutation", staticLimit(key=operator.attrgetter("height"), max_value=5))

        # We register the evaluation function, the selection method,
        # and variational operators on the strings. 
        self.toolbox.register("evaluate", self.get_ebm_fitness)
        self.toolbox.register("select", tools.selRoulette)
        self.toolbox.register("mate", self.crossover_strings)
        self.toolbox.register("mutate", self.mutate_string)

        # We collect statistics on the fitness values.
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", lambda x: np.around(np.mean(x), 2))
        self.stats.register("std", lambda x: np.around(np.std(x), 2))
        self.stats.register("min", lambda x: np.around(np.min(x), 2))
        self.stats.register("max", lambda x: np.around(np.max(x), 2))

        # We also add a Hall of fame object, and we keep the 10 best individuals in it.

        self.hof = tools.HallOfFame(10)

        # Set DEAP to evolve functions in parallel
        self.pool = ProcessPool(nodes=n_parallel_nodes)
        self.toolbox.register("map", self.pool.map)

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

    def generate_tree_string_representation(self):
        act_tree = PrimitiveTree(genGrow(self.pset, min_=self.min_depth, max_=self.max_depth))
        str_representation = str(act_tree)
        return str_representation

    def crossover_strings(self, parent_1, parent_2):
        # parent_1 and parent_2 are string representations, so 
        # we convert them to primitive trees and apply
        # a crossover operation on them.

        act1 = PrimitiveTree.from_string(parent_1, pset=self.pset)
        act2 = PrimitiveTree.from_string(parent_2, pset=self.pset)
        child_1, child_2 = self.toolbox.primitive_tree_crossover(act1, act2)

        # We return the string representations of the children

        string_child_1 = creator.Individual(child_1)
        string_child_2 = creator.Individual(child_2)
        return string_child_1, string_child_2

    def mutate_string(self, parent):

        act = PrimitiveTree.from_string(parent, self.pset)
        child = self.toolbox.primitive_tree_mutation(act)[0]
        string_child = creator.Individual(child)
        # DEAP expects the mutation to return a tuple 
        # of one tree
        return (string_child,)

    def get_ebm_fitness(self, individual):
        #sleep(2)
        # The individual is a string representation. 
        act = PrimitiveTree.from_string(individual, pset=self.pset)
        # Choose some very bad value in case of an error, not infinity to prevent DEAP from encountering an error
        f = compile(act, self.pset)
        const = tf.constant([1.0, 2.0, 3.0])
        fitness = np.around(tf.reduce_sum(f(const)).numpy(), 2)
        if np.isnan(fitness) or np.isinf(fitness):
            fitness = -10 ** 3
        return (fitness,)

    def adjust_activation_tree(self, activation_tree_str):
        copied_str = deepcopy(activation_tree_str)
        copied_activation_tree = PrimitiveTree.from_string(copied_str, self.pset)
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
        # return the string representation of the new tree
        return creator.Individual(copied_activation_tree)

    def create_generation(self):
        generation = [self.toolbox.individual() for i in range(self.pop_size)]
        generation = [self.adjust_activation_tree(x) for x in generation]
        return generation

    def evolve_functions(self, num_generations):
        pop = self.create_generation()
        pop, log = eaSimple(pop, self.toolbox, ngen=num_generations, cxpb=0.8, mutpb=0.02, stats=self.stats,
                            halloffame=self.hof, verbose=True)
        return pop

    def get_hof_inds(self):
        return ['Function:{}\tFitness:{}'.format(str(mvp), mvp.fitness.values[0]) for mvp in self.hof]


if __name__ == '__main__':
    base_functions = [elu, gelu, linear, relu, selu, sigmoid, softplus, swish, tanh, atan, cos, erf, sin, sqrt]
    base_operations = [maximum, minimum, add, subtract]

    evo = EvolutionaryAlgorithm(base_functions, base_operations, min_depth=1, max_depth=5, pop_size=40,
                                n_parallel_nodes=1)
    final_pop = evo.evolve_functions(10)
