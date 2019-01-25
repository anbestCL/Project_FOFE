#!/usr/bin/python3

import random
import sys
import time
import copy
from wrapper import Wrapper
import argparse
import torch
import pickle


def extend_subrange(subrange, fullrange, best_config):
    is_extended = False
    if subrange == {}:
        # Initialize subrange to median values.
        for param, vals in fullrange.items():
            if type(vals) is set:
                subrange[param] = list(vals)
            else:
                median_index = (len(vals)-1)//2
                subrange[param] = [vals[median_index]]
                if len(vals) > 1:
                    subrange[param].append(vals[median_index + 1])
            is_extended = True
    else:
        # Increase subrange if best config is on the corners and can be extended.
        for param in fullrange.keys():
            if type(fullrange[param]) is set:
                continue
            best_setting = best_config[param]
            is_left_subrange = subrange[param][0] == best_setting
            is_right_subrange = subrange[param][-1] == best_setting
            is_left_fullrange = fullrange[param][0] == best_setting
            is_right_fullrange = fullrange[param][-1] == best_setting
            extend_index = fullrange[param].index(best_setting)
            if is_left_subrange and not is_left_fullrange:
                subrange[param].insert(0, fullrange[param][extend_index - 1])
                is_extended = True
            elif is_right_subrange and not is_right_fullrange:
                subrange[param].append(fullrange[param][extend_index + 1])
                is_extended = True
    return is_extended


def random_search(learner, params={}, seed=0, attempts_per_param=2):
    """
    Executes a random search over the parameters, given a learner (a wrapper over a learning algorithm), and a dictionary
    mapping parameter names to their ranges (lists for ordered ranges, sets for unordered value alternatives).
    The parameters for optimization are the maximal range, random sampling considers a smaller subrange of those.
    The subrange is extended if optimal configurations lie on the boundary of the subrange.

    The learner needs to implement the following method:

    epochs_to_model_costs = learner.learn(num_epochs=num_epochs, config=config, seed=seed)

    where num_epochs is the list of epochs/checkpoints to consider for optimization, and config is a dictionary with
    a chosen (sampled) value for each hyper-parameters (number of epochs is not one of them).
    The returned epochs_to_model_costs maps epoch numbers to tuples containing (model at epoch, validation loss, lest loss).

    :param learner: Wrapper for learning algorithm.
    :param params: Maximal range to optimize for.
    :param seed: random seed.
    :return:
    """
    print("full parameter range:")
    print(params)
    print("===")

    shuffle_seed = 0
    random.seed(shuffle_seed)
    params_subrange = {}

    best_cost = sys.float_info.max
    associated_test_cost = sys.float_info.max
    best_config = {}
    tried_configs = set()

    params_copy = params.copy()
    num_epochs = params_copy["num_epochs"]
    del params_copy["num_epochs"]

    # Two samples for each parameter to optimize (only those that have a choice)
    attempts_per_round = max(
        1, attempts_per_param * sum([1 for l in params_copy.values() if len(l) > 1]))

    while extend_subrange(params_subrange, params_copy, best_config):
        print("params_subrange:")
        print(params_subrange)
        print("===")

        for setting_nr in range(attempts_per_round):
            start = time.time()

            config = {}
            for param, settings in params_subrange.items():
                selection = random.choice(settings)
                config[param] = selection

            if frozenset(config.items()) not in tried_configs:
                print(" === Running config: ===")
                print(config)
                tried_configs.add(frozenset(config.items()))
                epochs_to_model_costs = learner.learn(
                    num_epochs=num_epochs, config_dict=config, seed=seed)
                shuffle_seed += 1
                random.seed(shuffle_seed)
                for num_epochs_selected, model_costs in epochs_to_model_costs.items():
                    model, _, cost_valid, cost_test, _, _, _ = model_costs
                    config["num_epochs"] = num_epochs_selected
                    print(config)
                    print("Cost (valid, test_info): %f, %s" %
                          (cost_valid, str(cost_test)))
                    if cost_valid < best_cost:
                        best_config = copy.deepcopy(config)
                        best_cost = cost_valid
                        best_model = model
                        associated_test_cost = cost_test
                time_elapsed = time.time() - start
                print("time (s):" + str(time_elapsed))
                print("Best config and cost so far:")
                print(best_config)
                print(best_cost)
                print(associated_test_cost)
                print("===")
            else:
                print(" === already tried: ===")
                print(config)
                print("===")
    print("Best config, dev cost, test cost:")
    print(best_config)
    print(best_cost)
    print(associated_test_cost)
    print("===")
    return best_model, best_config, associated_test_cost


parser = argparse.ArgumentParser(
    description='Training program of the BIOS Tagger.')
parser.add_argument('modelname', type=str,
                    help='type of model to be trained: FOFE encodings or Classic embeddings')
parser.add_argument('datafile', type=str,
                    help='file or folder containing the data')
parser.add_argument('--batch_size', type=int, default=8,
                    help='size of the data batches')
parser.add_argument('--num_epochs', nargs='+', type=int, default=[0, 1, 5, 10],
                    help='number of epochs used for training')

args = parser.parse_args()

learner = Wrapper(args.modelname, args.datafile, args.batch_size)
params = {'num_epochs': args.num_epochs, 'embedding_size': [50, 100, 200], 'hidden_size': [50, 100, 200], 'dropout': {
    0.3, 0.5, 0.7}, 'learn_rate': [0.001, 0.01, 0.1], 'reg_factor': [0.0, 0.001, 0.01]}

best_model, best_config, associated_test_cost = random_search(
    learner, params, seed=0, attempts_per_param=2)
torch.save(best_model, "hyper_best_model_classic.nnp")
with open("hyper_best_config_classic.txt", "wb"):
    pickle.dump(best_config)
    pickle.dump(associated_test_cost)
