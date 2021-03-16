# import tensorflow as tf
from tensorflow.keras.activations import elu,exponential,gelu,linear,relu,selu,sigmoid,softmax,softplus,swish,tanh
from tensorflow.math import add,atan,cos,erf,maximum,minimum,sin,sqrt,subtract
from deap.gp import PrimitiveSet,PrimitiveTree,genGrow,genFull,compile,cxOnePoint,mutShrink,staticLimit
from deap.algorithms import eaSimple
from deap import creator,base,tools 
from copy import deepcopy
from collections import Counter
import numpy as np
# import tensorflow as tf 
import operator 
np.set_printoptions(precision=2)

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.python.platform import flags

from data import Cifar10
from models import DspritesNet, ResNet32, ResNet32Large, ResNet32Larger, ResNet32Wider, MnistNet, ResNet128
import os.path as osp
import os
from baselines.logger import TensorBoardOutputFormat
from utils import average_gradients, ReplayBuffer, optimistic_restore
from tqdm import tqdm
import random
from torch.utils.data import DataLoader
import time as time
from io import StringIO
from tensorflow.core.util import event_pb2
import torch
from custom_adam import AdamOptimizer
from scipy.misc import imsave
import matplotlib.pyplot as plt
from hmc import hmc
import math
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

import horovod.tensorflow as hvd
hvd.init()

from inception import get_inception_score

torch.manual_seed(hvd.rank())
np.random.seed(hvd.rank())
tf.set_random_seed(hvd.rank())

FLAGS = flags.FLAGS


# Dataset Options
flags.DEFINE_string('datasource', 'random',
    'initialization for chains, either random or default (decorruption)')
flags.DEFINE_string('dataset','mnist',
    'dsprites, cifar10, imagenet (32x32) or imagenetfull (128x128)')
flags.DEFINE_integer('batch_size', 256, 'Size of inputs')
flags.DEFINE_bool('single', False, 'whether to debug by training on a single image')
flags.DEFINE_integer('data_workers', 4,
    'Number of different data workers to load data in parallel')

# General Experiment Settings
flags.DEFINE_string('logdir', 'cachedir',
    'location where log of experiments will be stored')
flags.DEFINE_string('exp', 'default', 'name of experiments')
flags.DEFINE_integer('log_interval', 10, 'log outputs every so many batches')
flags.DEFINE_integer('save_interval', 1000,'save outputs every so many batches')
flags.DEFINE_integer('test_interval', 1000,'evaluate outputs every so many batches')
flags.DEFINE_integer('resume_iter', -1, 'iteration to resume training from')
flags.DEFINE_bool('train', True, 'whether to train or test')
flags.DEFINE_integer('epoch_num', 2, 'Number of Epochs to train on')
flags.DEFINE_float('lr', 3e-4, 'Learning for training')
flags.DEFINE_integer('num_gpus', 1, 'number of gpus to train on')

# EBM Specific Experiments Settings
flags.DEFINE_float('ml_coeff', 1.0, 'Maximum Likelihood Coefficients')
flags.DEFINE_float('l2_coeff', 1.0, 'L2 Penalty training')
flags.DEFINE_bool('cclass', False, 'Whether to conditional training in models')
flags.DEFINE_bool('model_cclass', False,'use unsupervised clustering to infer fake labels')
flags.DEFINE_integer('temperature', 1, 'Temperature for energy function')
flags.DEFINE_string('objective', 'cd', 'use either contrastive divergence objective(least stable),'
                    'logsumexp(more stable)'
                    'softplus(most stable)')
flags.DEFINE_bool('zero_kl', False, 'whether to zero out the kl loss')

# Setting for MCMC sampling
flags.DEFINE_float('proj_norm', 0.0, 'Maximum change of input images')
flags.DEFINE_string('proj_norm_type', 'li', 'Either li or l2 ball projection')
flags.DEFINE_integer('num_steps', 20, 'Steps of gradient descent for training')
flags.DEFINE_float('step_lr', 1.0, 'Size of steps for gradient descent')
flags.DEFINE_bool('replay_batch', False, 'Use MCMC chains initialized from a replay buffer.')
flags.DEFINE_bool('hmc', False, 'Whether to use HMC sampling to train models')
flags.DEFINE_float('noise_scale', 1.,'Relative amount of noise for MCMC')
flags.DEFINE_bool('pcd', False, 'whether to use pcd training instead')

# Architecture Settings
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets')
flags.DEFINE_bool('spec_norm', True, 'Whether to use spectral normalization on weights')
flags.DEFINE_bool('use_attention', False, 'Whether to use self attention in network')
flags.DEFINE_bool('large_model', False, 'whether to use a large model')
flags.DEFINE_bool('larger_model', False, 'Deeper ResNet32 Network')
flags.DEFINE_bool('wider_model', False, 'Wider ResNet32 Network')
flags.DEFINE_bool('resnet_model', False, 'Vanilla Renset Network')

# Dataset settings
flags.DEFINE_bool('mixup', False, 'whether to add mixup to training images')
flags.DEFINE_bool('augment', False, 'whether to augmentations to images')
flags.DEFINE_float('rescale', 1.0, 'Factor to rescale inputs from 0-1 box')

# Dsprites specific experiments
flags.DEFINE_bool('cond_shape', False, 'condition of shape type')
flags.DEFINE_bool('cond_size', False, 'condition of shape size')
flags.DEFINE_bool('cond_pos', False, 'condition of position loc')
flags.DEFINE_bool('cond_rot', False, 'condition of rot')

FLAGS.step_lr = FLAGS.step_lr * FLAGS.rescale

FLAGS.batch_size *= FLAGS.num_gpus

print("{} batch size".format(FLAGS.batch_size))


class EvolutionaryAlgorithm:
    
    def __init__(self,base_act_functions,base_operations,min_depth,max_depth,pop_size):
        
        # We first initialize the pset that contains our base 
        # building blocks.
        
        self.base_act_functions=base_act_functions
        self.base_operations=base_operations
        self.pset=self.initialize_pset(base_act_functions,base_operations)
        self.pop_size=pop_size 
        
        # We specify that we are dealing with a maximization problem, and 
        # that our individual is a PrimitiveTree. We add a reference 
        # to the pset in "Individual", since it is used by some DEAP 
        # GP operators when modifying an individual.
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual",PrimitiveTree,fitness=creator.FitnessMax,pset=self.pset)
        
        self.toolbox=base.Toolbox()
        
        # Our toolbox contains an "expr" function that calls genGrow with 
        # the pset as a parameter. genGrow also takes as parameter the 
        # minimum and maximum depth of the tree. We do not fix them
        # here, and we leave them as variables which can be changed when 
        # creating a population with the create_generation() function
        
        self.toolbox.register("expr",genGrow,pset=self.pset,min_=min_depth,max_=max_depth)
        
        #The "individual" function calls "expr()" on a "creator.Individual" object 
        # and returns it. In other words, it simply creates a new Primitive tree 
        # from our pset
        self.toolbox.register("individual",tools.initIterate,creator.Individual,self.toolbox.expr)
        
        # We register the variation operators and a simple EA algorithm.
        
        self.toolbox.register("evaluate",self.get_ebm_fitness, ebm_prob=EBMProbML)
        self.toolbox.register("select",tools.selRoulette)
        self.toolbox.register("mate",cxOnePoint)
        self.toolbox.register("mutate",mutShrink)
        
        # We decorate the mating and mutation operator with a limit on the maximum depth.
        
        self.toolbox.decorate("mate",staticLimit(key=operator.attrgetter("height"),max_value=5))
        self.toolbox.decorate("mutate",staticLimit(key=operator.attrgetter("height"),max_value=5))
        
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        self.mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        self.mstats.register("avg", np.mean)
        self.mstats.register("std", np.std)
        self.mstats.register("min", np.min)
        self.mstats.register("max", np.max)
        
        # We also add a Hall of fame object, and we keep the 10 best individuals in it.
        
        self.hof=tools.HallOfFame(10)
        print("Loading data...")
        path = "/home/abhi/Documents/courses/UofT/CSC2506/project/data/cifar10"
        train_dataset = Cifar10(train=True, augment=FLAGS.augment, rescale=FLAGS.rescale, path=path)
        indices = random.sample(range(1,50000), 5000);
        train_indices = indices[0:3500]; #training data
        valid_indices = indices[3500:]; # validation data
        
        train_data = torch.utils.data.Subset(train_dataset, train_indices)
        valid_data = torch.utils.data.Subset(train_dataset, valid_indices)
        print("Length of training data:%d"%len(train_data))
        print("Length of validation data:%d"%len(train_data))
        self.train_data = DataLoader(train_data, batch_size=FLAGS.batch_size, num_workers=FLAGS.data_workers, drop_last=True, shuffle=True)
        self.valid_data = DataLoader(valid_data, batch_size=FLAGS.batch_size, num_workers=FLAGS.data_workers, drop_last=True, shuffle=True)
        print("Done loading...")
        

    def initialize_pset(self,base_act_functions,base_operations):
        
        # Our desired functions are unary and we specificy this when creating the pset
        pset=PrimitiveSet("main",1)
        
        # Activation functions in a neural network have 
        # arity 1.
        for func in base_act_functions:
            pset.addPrimitive(func,1)
            
        # The operations that we are considering, such as 
        # maximum, add or subtract, have arity 2.
        for op in base_operations:
            pset.addPrimitive(op,2)
            
        pset.renameArguments(ARG0="x")
        
        return pset

    def adjust_activation_tree(self,activation_tree):
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
                    condition = copied_activation_tree[i].name == 'ARG0' and \
                                (copied_activation_tree[i + 1].name == 'ARG0' or two_literals)
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
        generation=[self.toolbox.individual() for i in range(self.pop_size)]
        generation=[self.adjust_activation_tree(x) for x in generation]
        return generation
    
    # This will be replaced with a proper EBM training function later on. Make sure to take care of nan case 
    def get_ebm_fitness(self,individual,ebm_prob):
        tf.keras.backend.clear_session()
        print("-------------------------------Activation function:%s-----------------------------------"%str(individual))
        # Choose some very bad value in case of an error, not infinity to prevent DEAP from encountering an error
        act_fun=compile(individual,self.pset)
        ebm_prob = EBMProbML(act_fun)
        # FLAGS.cclass = True
        train_inc_score = ebm_prob.train_unconditional(self.train_data)
        print("Training Inception score:.%3f"%train_inc_score)
        if train_inc_score > 0:
            test_inc_score = ebm_prob.test_unconditional(self.valid_data)
            print("Testing inception score:.%3f"%test_inc_score)
            log_file = open("ebm_log.txt", "a")
            line = "%s Inception score: Train:%.5f Test:%.5f,"%(str(individual), train_inc_score, test_inc_score)
            log_file.write(line + "\n")
            log_file.close()
            return (test_inc_score,)
        else:
            print("Training diverged!")
            return(-1.,)
    
    def evolve_functions(self,num_generations):
        pop=self.create_generation()
        pop,log=eaSimple(pop,self.toolbox,ngen=num_generations,cxpb=0.8,mutpb=0.02,stats=self.mstats,halloffame=self.hof,verbose=True)
        return pop
    
    def get_hof_inds(self):
        return ['Function:{}\tFitness:{}'.format(str(mvp),mvp.fitness.values[0]) for mvp in self.hof]

def compress_x_mod(x_mod):
    x_mod = (255 * np.clip(x_mod, 0, FLAGS.rescale) / FLAGS.rescale).astype(np.uint8)
    return x_mod


def decompress_x_mod(x_mod):
    x_mod = x_mod / 256 * FLAGS.rescale + \
        np.random.uniform(0, 1 / 256 * FLAGS.rescale, x_mod.shape)
    return x_mod


def make_image(tensor):
    """Convert an numpy representation image to Image protobuf"""
    from PIL import Image
    if len(tensor.shape) == 4:
        _, height, width, channel = tensor.shape
    elif len(tensor.shape) == 3:
        height, width, channel = tensor.shape
    elif len(tensor.shape) == 2:
        height, width = tensor.shape
        channel = 1
    tensor = tensor.astype(np.uint8)
    image = Image.fromarray(tensor)
    import io
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                            width=width,
                            colorspace=channel,
                            encoded_image_string=image_string)


def log_image(im, logger, tag, step=0):
    im = make_image(im)

    summary = [tf.Summary.Value(tag=tag, image=im)]
    summary = tf.Summary(value=summary)
    event = event_pb2.Event(summary=summary)
    event.step = step
    logger.writer.WriteEvent(event)
    logger.writer.Flush()


def rescale_im(image):
    image = np.clip(image, 0, FLAGS.rescale)
    if FLAGS.dataset == 'mnist' or FLAGS.dataset == 'dsprites':
        return (np.clip((FLAGS.rescale - image) * 256 / FLAGS.rescale, 0, 255)).astype(np.uint8)
    else:
        return (np.clip(image * 256 / FLAGS.rescale, 0, 255)).astype(np.uint8)


def train(target_vars, saver, sess, logger, dataloader, resume_iter, logdir):
    X = target_vars['X']
    Y = target_vars['Y']
    X_NOISE = target_vars['X_NOISE']
    train_op = target_vars['train_op']
    energy_pos = target_vars['energy_pos']
    energy_neg = target_vars['energy_neg']
    loss_energy = target_vars['loss_energy']
    loss_ml = target_vars['loss_ml']
    loss_total = target_vars['total_loss']
    gvs = target_vars['gvs']
    x_grad = target_vars['x_grad']
    x_grad_first = target_vars['x_grad_first']
    x_off = target_vars['x_off']
    temp = target_vars['temp']
    x_mod = target_vars['x_mod']
    LABEL = target_vars['LABEL']
    LABEL_POS = target_vars['LABEL_POS']
    weights = target_vars['weights']
    test_x_mod = target_vars['test_x_mod']
    eps = target_vars['eps_begin']
    label_ent = target_vars['label_ent']

    if FLAGS.use_attention:
        gamma = weights[0]['atten']['gamma']
    else:
        gamma = tf.zeros(1)

    val_output = [test_x_mod]

    gvs_dict = dict(gvs)

    log_output = [
        train_op,
        energy_pos,
        energy_neg,
        eps,
        loss_energy,
        loss_ml,
        loss_total,
        x_grad,
        x_off,
        x_mod,
        gamma,
        x_grad_first,
        label_ent,
        *gvs_dict.keys()]
    output = [train_op, x_mod]

    replay_buffer = ReplayBuffer(10000)
    itr = resume_iter
    x_mod = None
    gd_steps = 1

    dataloader_iterator = iter(dataloader)
    best_inception = 0.0

    for epoch in range(FLAGS.epoch_num):
        print("Training epoch:%d"%epoch)
        for data_corrupt, data, label in dataloader:
            data_corrupt = data_corrupt_init = data_corrupt.numpy()
            data_corrupt_init = data_corrupt.copy()

            data = data.numpy()
            label = label.numpy()

            label_init = label.copy()

            if FLAGS.mixup:
                idx = np.random.permutation(data.shape[0])
                lam = np.random.beta(1, 1, size=(data.shape[0], 1, 1, 1))
                data = data * lam + data[idx] * (1 - lam)

            if FLAGS.replay_batch and (x_mod is not None):
                replay_buffer.add(compress_x_mod(x_mod))

                if len(replay_buffer) > FLAGS.batch_size:
                    replay_batch = replay_buffer.sample(FLAGS.batch_size)
                    replay_batch = decompress_x_mod(replay_batch)
                    replay_mask = (
                        np.random.uniform(
                            0,
                            FLAGS.rescale,
                            FLAGS.batch_size) > 0.05)
                    data_corrupt[replay_mask] = replay_batch[replay_mask]

            if FLAGS.pcd:
                if x_mod is not None:
                    data_corrupt = x_mod

            feed_dict = {X_NOISE: data_corrupt, X: data, Y: label}

            if FLAGS.cclass:
                feed_dict[LABEL] = label
                feed_dict[LABEL_POS] = label_init

            if itr % FLAGS.log_interval == 0:
                _, e_pos, e_neg, eps, loss_e, loss_ml, loss_total, x_grad, x_off, x_mod, gamma, x_grad_first, label_ent, * \
                    grads = sess.run(log_output, feed_dict)

                kvs = {}
                kvs['e_pos'] = e_pos.mean()
                kvs['e_pos_std'] = e_pos.std()
                kvs['e_neg'] = e_neg.mean()
                kvs['e_diff'] = kvs['e_pos'] - kvs['e_neg']
                kvs['e_neg_std'] = e_neg.std()
                kvs['temp'] = temp
                kvs['loss_e'] = loss_e.mean()
                kvs['eps'] = eps.mean()
                kvs['label_ent'] = label_ent
                kvs['loss_ml'] = loss_ml.mean()
                kvs['loss_total'] = loss_total.mean()
                kvs['x_grad'] = np.abs(x_grad).mean()
                kvs['x_grad_first'] = np.abs(x_grad_first).mean()
                kvs['x_off'] = x_off.mean()
                kvs['iter'] = itr
                kvs['gamma'] = gamma

                for v, k in zip(grads, [v.name for v in gvs_dict.values()]):
                    kvs[k] = np.abs(v).max()

                string = "Obtained a total of "
                for key, value in kvs.items():
                    string += "{}: {}, ".format(key, value)
                    if math.isnan(value):
                        return -1.

                if hvd.rank() == 0:
                    print(string)
                    logger.writekvs(kvs)
            else:
                _, x_mod = sess.run(output, feed_dict)

            if itr % FLAGS.save_interval == 0 and hvd.rank() == 0:
                saver.save(
                    sess,
                    osp.join(
                        FLAGS.logdir,
                        FLAGS.exp,
                        'model_{}'.format(itr)))

            if itr % FLAGS.test_interval == 0 and hvd.rank() == 0 and FLAGS.dataset != '2d':
                try_im = x_mod
                orig_im = data_corrupt.squeeze()
                actual_im = rescale_im(data)

                orig_im = rescale_im(orig_im)
                try_im = rescale_im(try_im).squeeze()

                for i, (im, t_im, actual_im_i) in enumerate(
                        zip(orig_im[:20], try_im[:20], actual_im)):
                    shape = orig_im.shape[1:]
                    new_im = np.zeros((shape[0], shape[1] * 3, *shape[2:]))
                    size = shape[1]
                    new_im[:, :size] = im
                    new_im[:, size:2 * size] = t_im
                    new_im[:, 2 * size:] = actual_im_i

                    log_image(
                        new_im, logger, 'train_gen_{}'.format(itr), step=i)

                test_im = x_mod

                try:
                    data_corrupt, data, label = next(dataloader_iterator)
                except BaseException:
                    dataloader_iterator = iter(dataloader)
                    data_corrupt, data, label = next(dataloader_iterator)

                data_corrupt = data_corrupt.numpy()

                if FLAGS.replay_batch and (
                        x_mod is not None) and len(replay_buffer) > 0:
                    replay_batch = replay_buffer.sample(FLAGS.batch_size)
                    replay_batch = decompress_x_mod(replay_batch)
                    replay_mask = (
                        np.random.uniform(
                            0, 1, (FLAGS.batch_size)) > 0.05)
                    data_corrupt[replay_mask] = replay_batch[replay_mask]

                if FLAGS.dataset == 'cifar10' or FLAGS.dataset == 'imagenet' or FLAGS.dataset == 'imagenetfull':
                    n = 128

                    if FLAGS.dataset == "imagenetfull":
                        n = 32

                    if len(replay_buffer) > n:
                        data_corrupt = decompress_x_mod(replay_buffer.sample(n))
                    elif FLAGS.dataset == 'imagenetfull':
                        data_corrupt = np.random.uniform(
                            0, FLAGS.rescale, (n, 128, 128, 3))
                    else:
                        data_corrupt = np.random.uniform(
                            0, FLAGS.rescale, (n, 32, 32, 3))

                    if FLAGS.dataset == 'cifar10':
                        label = np.eye(10)[np.random.randint(0, 10, (n))]
                    else:
                        label = np.eye(1000)[
                            np.random.randint(
                                0, 1000, (n))]

                feed_dict[X_NOISE] = data_corrupt

                feed_dict[X] = data

                if FLAGS.cclass:
                    feed_dict[LABEL] = label

                test_x_mod = sess.run(val_output, feed_dict)

                try_im = test_x_mod
                orig_im = data_corrupt.squeeze()
                actual_im = rescale_im(data.numpy())

                orig_im = rescale_im(orig_im)
                try_im = rescale_im(try_im).squeeze()

                for i, (im, t_im, actual_im_i) in enumerate(
                        zip(orig_im[:20], try_im[:20], actual_im)):

                    shape = orig_im.shape[1:]
                    new_im = np.zeros((shape[0], shape[1] * 3, *shape[2:]))
                    size = shape[1]
                    new_im[:, :size] = im
                    new_im[:, size:2 * size] = t_im
                    new_im[:, 2 * size:] = actual_im_i
                    log_image(
                        new_im, logger, 'val_gen_{}'.format(itr), step=i)

                score, std = get_inception_score(list(try_im), splits=1)
                print(
                    "///Inception score of {} with std of {}".format(
                        score, std))
                kvs = {}
                kvs['inception_score'] = score
                kvs['inception_score_std'] = std
                logger.writekvs(kvs)

                if score > best_inception:
                    best_inception = score
                    saver.save(
                        sess,
                        osp.join(
                            FLAGS.logdir,
                            FLAGS.exp,
                            'model_best'))

            if itr > 60000 and FLAGS.dataset == "mnist":
                assert False
            itr += 1
            print("Training iteration:%d"%itr)

    saver.save(sess, osp.join(FLAGS.logdir, FLAGS.exp, 'model_{}'.format(itr)))
    return best_inception

cifar10_map = {0: 'airplane',
               1: 'automobile',
               2: 'bird',
               3: 'cat',
               4: 'deer',
               5: 'dog',
               6: 'frog',
               7: 'horse',
               8: 'ship',
               9: 'truck'}


def test(target_vars, saver, sess, logger, dataloader):
    X_NOISE = target_vars['X_NOISE']
    X = target_vars['X']
    Y = target_vars['Y']
    LABEL = target_vars['LABEL']
    energy_start = target_vars['energy_start']
    x_mod = target_vars['x_mod']
    x_mod = target_vars['test_x_mod']
    energy_neg = target_vars['energy_neg']

    np.random.seed(1)
    random.seed(1)

    output = [x_mod, energy_start, energy_neg]

    dataloader_iterator = iter(dataloader)
    data_corrupt, data, label = next(dataloader_iterator)
    data_corrupt, data, label = data_corrupt.numpy(), data.numpy(), label.numpy()

    orig_im = try_im = data_corrupt

    if FLAGS.cclass:
        try_im, energy_orig, energy = sess.run(
            output, {X_NOISE: orig_im, Y: label[0:1], LABEL: label})
    else:
        try_im, energy_orig, energy = sess.run(
            output, {X_NOISE: orig_im, Y: label[0:1]})

    orig_im = rescale_im(orig_im)
    try_im = rescale_im(try_im)
    actual_im = rescale_im(data)

    for i, (im, energy_i, t_im, energy, label_i, actual_im_i) in enumerate(
            zip(orig_im, energy_orig, try_im, energy, label, actual_im)):
        print("Generating new image:%d"%i)
        label_i = np.array(label_i)

        shape = orig_im.shape[1:]
        new_im = np.zeros((shape[0], shape[1] * 3, *shape[2:]))
        size = shape[1]
        new_im[:, :size] = im
        new_im[:, size:2 * size] = t_im

        if FLAGS.cclass:
            label_i = np.where(label_i == 1)[0][0]
            if FLAGS.dataset == 'cifar10':
                log_image(new_im, logger, '{}_{:.4f}_now_{:.4f}_{}'.format(
                    i, energy_i[0], energy[0], cifar10_map[label_i]), step=i)
            else:
                log_image(
                    new_im,
                    logger,
                    '{}_{:.4f}_now_{:.4f}_{}'.format(
                        i,
                        energy_i[0],
                        energy[0],
                        label_i),
                    step=i)
        else:
            log_image(
                new_im,
                logger,
                '{}_{:.4f}_now_{:.4f}'.format(
                    i,
                    energy_i[0],
                    energy[0]),
                step=i)

    test_ims = list(try_im)
    real_ims = list(actual_im)

    for i in tqdm(range(1500 // FLAGS.batch_size + 1)):
        print("Generating test and real images:%d"%i)
        try:
            data_corrupt, data, label = dataloader_iterator.next()
        except BaseException:
            dataloader_iterator = iter(dataloader)
            data_corrupt, data, label = dataloader_iterator.next()

        data_corrupt, data, label = data_corrupt.numpy(), data.numpy(), label.numpy()

        if FLAGS.cclass:
            try_im, energy_orig, energy = sess.run(
                output, {X_NOISE: data_corrupt, Y: label[0:1], LABEL: label})
        else:
            try_im, energy_orig, energy = sess.run(
                output, {X_NOISE: data_corrupt, Y: label[0:1]})

        try_im = rescale_im(try_im)
        real_im = rescale_im(data)

        test_ims.extend(list(try_im))
        real_ims.extend(list(real_im))

    score, std = get_inception_score(test_ims)
    print("!!!Inception score of {} with std of {}".format(score, std))
    return score


def setup(act_fun):
    channel_num = 3
    if FLAGS.resnet_model:
        print("------------------Using ResNet32 model------------")
        model = ResNet32(
            num_channels=channel_num,
            num_filters=128,
            act_fun=act_fun)
    elif FLAGS.large_model:
        print("------------------Using ResNet32Large model------------")
        model = ResNet32Large(
            num_channels=channel_num,
            num_filters=128,
            train=True,
            act_fun=act_fun)
    elif FLAGS.larger_model:
        print("------------------Using ResNet32Larger model------------")
        model = ResNet32Larger(
            num_channels=channel_num,
            num_filters=128,
            act_fun=act_fun)
    elif FLAGS.wider_model:
        print("------------------Using ResNet32Wider model------------")
        model = ResNet32Wider(
            num_channels=channel_num,
            num_filters=192,
            act_fun=act_fun)
    else:
        print("------------------Using MNIST model------------")
        model = MnistNet(
            num_channels=channel_num,
            num_filters=128,
            act_fun=act_fun)
    batch_size = FLAGS.batch_size
    weights = [model.construct_weights('context_0')]

    Y = tf.placeholder(shape=(None), dtype=tf.int32)
    LABEL = None
    X_NOISE = tf.placeholder(shape=(None, 32, 32, 3), dtype=tf.float32)
    X = tf.placeholder(shape=(None, 32, 32, 3), dtype=tf.float32)
    LABEL = tf.placeholder(shape=(None, 10), dtype=tf.float32)
    LABEL_POS = tf.placeholder(shape=(None, 10), dtype=tf.float32)
    # Varibles to run in training
    X_SPLIT = tf.split(X, FLAGS.num_gpus)
    X_NOISE_SPLIT = tf.split(X_NOISE, FLAGS.num_gpus)
    LABEL_SPLIT = tf.split(LABEL, FLAGS.num_gpus)
    LABEL_POS_SPLIT = tf.split(LABEL_POS, FLAGS.num_gpus)
    LABEL_SPLIT_INIT = list(LABEL_SPLIT)
    tower_grads = []
    tower_gen_grads = []
    x_mod_list = []

    optimizer = AdamOptimizer(FLAGS.lr, beta1=0.0, beta2=0.999)
    optimizer = hvd.DistributedOptimizer(optimizer)

    for j in range(FLAGS.num_gpus):
        if FLAGS.model_cclass:
            ind_batch_size = FLAGS.batch_size // FLAGS.num_gpus
            label_tensor = tf.Variable(
                tf.convert_to_tensor(
                    np.reshape(
                        np.tile(np.eye(10), (FLAGS.batch_size, 1, 1)),
                        (FLAGS.batch_size * 10, 10)),
                    dtype=tf.float32),
                trainable=False,
                dtype=tf.float32)
            x_split = tf.tile(
                tf.reshape(
                    X_SPLIT[j], (ind_batch_size, 1, 32, 32, 3)), (1, 10, 1, 1, 1))
            x_split = tf.reshape(x_split, (ind_batch_size * 10, 32, 32, 3))
            energy_pos = model.forward(
                x_split,
                weights[0],
                label=label_tensor,
                stop_at_grad=False)

            energy_pos_full = tf.reshape(energy_pos, (ind_batch_size, 10))
            energy_partition_est = tf.reduce_logsumexp(
                energy_pos_full, axis=1, keepdims=True)
            uniform = tf.random_uniform(tf.shape(energy_pos_full))
            label_tensor = tf.argmax(-energy_pos_full -
                                     tf.log(-tf.log(uniform)) - energy_partition_est, axis=1)
            label = tf.one_hot(label_tensor, 10, dtype=tf.float32)
            label = tf.Print(label, [label_tensor, energy_pos_full])
            LABEL_SPLIT[j] = label
            energy_pos = tf.concat(energy_pos, axis=0)
        else:
            energy_pos = [
                model.forward(
                    X_SPLIT[j],
                    weights[0],
                    label=LABEL_POS_SPLIT[j],
                    stop_at_grad=False)]
            energy_pos = tf.concat(energy_pos, axis=0)

        print("Building graph...")
        x_mod = x_orig = X_NOISE_SPLIT[j]

        x_grads = []

        energy_negs = []
        loss_energys = []

        energy_negs.extend([model.forward(tf.stop_gradient(
            x_mod), weights[0], label=LABEL_SPLIT[j], stop_at_grad=False, reuse=True)])
        eps_begin = tf.zeros(1)

        steps = tf.constant(0)
        c = lambda i, x: tf.less(i, FLAGS.num_steps)

        def langevin_step(counter, x_mod):
            x_mod = x_mod + tf.random_normal(tf.shape(x_mod),
                                             mean=0.0,
                                             stddev=0.005 * FLAGS.rescale * FLAGS.noise_scale)

            energy_noise = energy_start = tf.concat(
                [model.forward(
                        x_mod,
                        weights[0],
                        label=LABEL_SPLIT[j],
                        reuse=True,
                        stop_at_grad=False,
                        stop_batch=True)],
                axis=0)

            x_grad, label_grad = tf.gradients(
                FLAGS.temperature * energy_noise, [x_mod, LABEL_SPLIT[j]])
            energy_noise_old = energy_noise

            lr = FLAGS.step_lr

            if FLAGS.proj_norm != 0.0:
                if FLAGS.proj_norm_type == 'l2':
                    x_grad = tf.clip_by_norm(x_grad, FLAGS.proj_norm)
                elif FLAGS.proj_norm_type == 'li':
                    x_grad = tf.clip_by_value(
                        x_grad, -FLAGS.proj_norm, FLAGS.proj_norm)
                else:
                    print("Other types of projection are not supported!!!")
                    assert False

            # Clip gradient norm for now
            if FLAGS.hmc:
                # Step size should be tuned to get around 65% acceptance
                def energy(x):
                    return FLAGS.temperature * \
                        model.forward(x, weights[0], label=LABEL_SPLIT[j], reuse=True)

                x_last = hmc(x_mod, 15., 10, energy)
            else:
                x_last = x_mod - (lr) * x_grad

            x_mod = x_last
            x_mod = tf.clip_by_value(x_mod, 0, FLAGS.rescale)

            counter = counter + 1

            return counter, x_mod

        steps, x_mod = tf.while_loop(c, langevin_step, (steps, x_mod))

        energy_eval = model.forward(x_mod, weights[0], label=LABEL_SPLIT[j],
                                    stop_at_grad=False, reuse=True)
        x_grad = tf.gradients(FLAGS.temperature * energy_eval, [x_mod])[0]
        x_grads.append(x_grad)

        energy_negs.append(
            model.forward(
                tf.stop_gradient(x_mod),
                weights[0],
                label=LABEL_SPLIT[j],
                stop_at_grad=False,
                reuse=True))

        test_x_mod = x_mod

        temp = FLAGS.temperature

        energy_neg = energy_negs[-1]
        x_off = tf.reduce_mean(
            tf.abs(x_mod[:tf.shape(X_SPLIT[j])[0]] - X_SPLIT[j]))

        loss_energy = model.forward(
            x_mod,
            weights[0],
            reuse=True,
            label=LABEL,
            stop_grad=True)

        print("Finished processing loop construction ...")

        target_vars = {}

        if FLAGS.cclass or FLAGS.model_cclass:
            label_sum = tf.reduce_sum(LABEL_SPLIT[0], axis=0)
            label_prob = label_sum / tf.reduce_sum(label_sum)
            label_ent = -tf.reduce_sum(label_prob *
                                       tf.math.log(label_prob + 1e-7))
        else:
            label_ent = tf.zeros(1)

        target_vars['label_ent'] = label_ent

        if FLAGS.train:

            if FLAGS.objective == 'logsumexp':
                pos_term = temp * energy_pos
                energy_neg_reduced = (energy_neg - tf.reduce_min(energy_neg))
                coeff = tf.stop_gradient(tf.exp(-temp * energy_neg_reduced))
                norm_constant = tf.stop_gradient(tf.reduce_sum(coeff)) + 1e-4
                pos_loss = tf.reduce_mean(temp * energy_pos)
                neg_loss = coeff * (-1 * temp * energy_neg) / norm_constant
                loss_ml = FLAGS.ml_coeff * (pos_loss + tf.reduce_sum(neg_loss))
            elif FLAGS.objective == 'cd':
                pos_loss = tf.reduce_mean(temp * energy_pos)
                neg_loss = -tf.reduce_mean(temp * energy_neg)
                loss_ml = FLAGS.ml_coeff * (pos_loss + tf.reduce_sum(neg_loss))
            elif FLAGS.objective == 'softplus':
                loss_ml = FLAGS.ml_coeff * \
                    tf.nn.softplus(temp * (energy_pos - energy_neg))

            loss_total = tf.reduce_mean(loss_ml)

            if not FLAGS.zero_kl:
                loss_total = loss_total + tf.reduce_mean(loss_energy)

            loss_total = loss_total + \
                FLAGS.l2_coeff * (tf.reduce_mean(tf.square(energy_pos)) + tf.reduce_mean(tf.square((energy_neg))))

            print("Started gradient computation...")
            gvs = optimizer.compute_gradients(loss_total)
            gvs = [(k, v) for (k, v) in gvs if k is not None]

            print("Applying gradients...")

            tower_grads.append(gvs)

            print("Finished applying gradients.")

            target_vars['loss_ml'] = loss_ml
            target_vars['total_loss'] = loss_total
            target_vars['loss_energy'] = loss_energy
            target_vars['weights'] = weights
            target_vars['gvs'] = gvs

        target_vars['X'] = X
        target_vars['Y'] = Y
        target_vars['LABEL'] = LABEL
        target_vars['LABEL_POS'] = LABEL_POS
        target_vars['X_NOISE'] = X_NOISE
        target_vars['energy_pos'] = energy_pos
        target_vars['energy_start'] = energy_negs[0]

        if len(x_grads) >= 1:
            target_vars['x_grad'] = x_grads[-1]
            target_vars['x_grad_first'] = x_grads[0]
        else:
            target_vars['x_grad'] = tf.zeros(1)
            target_vars['x_grad_first'] = tf.zeros(1)

        target_vars['x_mod'] = x_mod
        target_vars['x_off'] = x_off
        target_vars['temp'] = temp
        target_vars['energy_neg'] = energy_neg
        target_vars['test_x_mod'] = test_x_mod
        target_vars['eps_begin'] = eps_begin

    if FLAGS.train:
        grads = average_gradients(tower_grads)
        train_op = optimizer.apply_gradients(grads)
        target_vars['train_op'] = train_op

    config = tf.ConfigProto()

    if hvd.size() > 1:
        config.gpu_options.visible_device_list = str(hvd.local_rank())

    sess = tf.Session(config=config)
    saver = loader = tf.train.Saver(max_to_keep=30, keep_checkpoint_every_n_hours=6)

    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("Model has a total of {} parameters".format(total_parameters))

    sess.run(tf.global_variables_initializer())

    resume_itr = 0

    if (FLAGS.resume_iter != -1 or not FLAGS.train) and hvd.rank() == 0:
        model_file = osp.join(logdir, 'model_{}'.format(FLAGS.resume_iter))
        resume_itr = FLAGS.resume_iter
        # saver.restore(sess, model_file)
        optimistic_restore(sess, model_file)

    sess.run(hvd.broadcast_global_variables(0))
    return target_vars, saver, sess, resume_itr


class EBMProbML:
    def __init__(self, act_fun=tf.nn.leaky_relu):
        print("Local rank: ", hvd.local_rank(), hvd.size())
        self.logdir = osp.join(FLAGS.logdir, FLAGS.exp)
        if hvd.rank() == 0:
            if not osp.exists(self.logdir):
                os.makedirs(self.logdir)
            self.logger = TensorBoardOutputFormat(self.logdir)
        else:
            self.logger = None
        self.act_fun = act_fun
        self.target_vars, self.saver, self.sess, self.resume_itr = setup(self.act_fun)
        
    def train_unconditional(self, data_loader):
        print("Training phase")
        inception_score = train(self.target_vars, 
                                self.saver,
                                self.sess,
                                self.logger,
                                data_loader,
                                self.resume_itr,
                                self.logdir)
        return inception_score

    def test_unconditional(self, data_loader):
        print("Testing phase")
        inception_score = test(self.target_vars,
                               self.saver,
                               self.sess,
                               self.logger,
                               data_loader)
        return inception_score

if __name__ == "__main__":
    
    # ebm_prob = EBMProbML(tf.nn.leaky_relu)
    # # ebm_prob = EBMProbML(custom_act)
    # # FLAGS.cclass = True
    # train_inc_score = ebm_prob.train_unconditional(data_loader)
    # print("Training inception score:%f"%train_inc_score)

    base_functions=[elu,gelu,linear,relu,selu,sigmoid,softplus,swish,tanh,atan,cos,erf,sin,sqrt]
    base_operations=[maximum,minimum,add,subtract]

    evo=EvolutionaryAlgorithm(base_functions,base_operations,min_depth=1,max_depth=3,pop_size=10)
    evo.evolve_functions(num_generations=5)

    # test_dataset = Cifar10(train=False, rescale=FLAGS.rescale, path=path)
    # test_dataset_1 = torch.utils.data.Subset(test_dataset, list(range(0, 500, 2)))
    # print("Length of test dataset:%d"%len(test_dataset_1))
    # data_loader_1 = DataLoader(test_dataset_1, batch_size=FLAGS.batch_size, num_workers=FLAGS.data_workers, drop_last=True, shuffle=True)   
    # print("Done loading...")
    # test_inc_score = ebm_prob.test_unconditional(data_loader_1)
    # print("Testing inception score:%f"%test_inc_score)
    # 