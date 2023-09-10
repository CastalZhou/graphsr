import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score, roc_auc_score
from collections import defaultdict
from sklearn import manifold
import matplotlib.pyplot as plt

from model import Encoder, MeanAggregator, SupervisedGraphSage, MLP

from eval_policy import eval_policy
import sys
from env import Env
from arguments import get_args

import scipy.sparse as sp


def data_spilit(labels, num_cls):
    num_nodes = labels.shape[0]
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1500]
    val = rand_indices[1500:2000]
    train_set = list(rand_indices[2000:])
    # train = random.sample(train, 100)

    tr_ratio = []
    count_tr = np.zeros(num_cls)
    # count_tr_ratio = np.array([20, 6, 20, 6, 20, 20, 6])
    count_tr_ratio = np.array([20, 20, 20, 20, 6, 6, 6])
    for i in train_set:
        for j in range(num_cls):
            if labels[i] == j:
                count_tr[j] += 1
                break
        # if count_tr[labels[i]] <= 20:
        #     tr_balanced.append(i)
        # count_tr[labels[i]] += 1
        if count_tr[labels[i]] <= count_tr_ratio[labels[i]]:
            tr_ratio.append(i)
    train_set = tr_ratio

    test_balanced = []
    count_test = np.zeros(num_cls)
    for i in test:
        for j in range(num_cls):
            if labels[i] == j:
                count_test[j] += 1
                break
        if count_test[labels[i]] <= 100:
            test_balanced.append(i)
    test = test_balanced

    val_bal = []
    count_val = np.zeros(num_cls)
    for i in val:
        for j in range(num_cls):
            if labels[i] == j:
                count_val[j] += 1
                break
        if count_val[labels[i]] <= 30:
            val_bal.append(i)
    val = val_bal

    index = np.arange(0, num_nodes)
    unlable = np.setdiff1d(index, train_set)
    unlable = np.setdiff1d(unlable, val)
    unlable = np.setdiff1d(unlable, test)
    # train_x = train
    train_y = []
    for i in train_set:
        train_y.append(int(labels[i]))
    # print(train_y)
    val_y = []
    for i in val:
        val_y.append(int(labels[i]))
    test_y = []
    for i in test:
        test_y.append(int(labels[i]))

    return train_set, train_y, val, val_y, test, test_y, unlable



def load_cora():
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("cora/cora.content") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            feat_data[i, :] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open("cora/cora.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists


def run_cora():
    np.random.seed(1)
    random.seed(1)
    num_cls = 7
    feat_data, labels, adj_lists = load_cora()
    features = nn.Embedding(2708, 1433)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    # features.cuda()

    train_x, train_y, val_x, val_y, test_x, test_y, unlable = data_spilit(labels, num_cls)

    train_x = torch.LongTensor(train_x)
    train_y = torch.LongTensor(train_y)
    val_x = torch.LongTensor(val_x)
    val_y = torch.LongTensor(val_y)
    test_x = torch.LongTensor(test_x)
    test_y = torch.LongTensor(test_y)
    unlable = torch.LongTensor(unlable)

    return train_x, train_y, val_x, val_y, test_x, test_y, unlable, features, adj_lists, labels


def test_model(env, actor_model, test_x, test_y, features, adj_lists, labels):
    """
        Tests the model.

        Parameters:
            env - the environment to test the policy on
            actor_model - the actor model to load in

        Return:
            None
    """
    print(f"Testing {actor_model}", flush=True)

    # If the actor model is not specified, then exit
    if actor_model == '':
        print(f"Didn't specify model file. Exiting.", flush=True)
        sys.exit(0)

    # Extract out dimensions of observation and action spaces
    obs_dim = 384  # env.observation_space.shape[0]
    act_dim = 2  # env.action_space.shape[0]

    # Build our policy the same way we build our actor model in PPO
    policy = MLP(obs_dim, act_dim)

    # Load in the actor model saved by the PPO algorithm
    policy.load_state_dict(torch.load(actor_model))

    eval_policy(policy=policy, env=env, test_x=test_x, test_y=test_y, features=features, adj_lists=adj_lists, labels=labels)


def main(args):
    """
        The main function to run.

        Parameters:
            args - the arguments parsed from command line

        Return:
            None
    """

    train_x, train_y, val_x, val_y, test_x, test_y, unlable, features, adj_lists, labels = run_cora()

    env = Env(train_x, train_y, unlable, val_x, val_y, features, adj_lists, test_x, test_y)

    if args.mode == 'test':
        test_model(env=env, actor_model='ppo_actor.pth', test_x=test_x, test_y=test_y, features=features, adj_lists=adj_lists, labels=labels)


if __name__ == "__main__":
    args = get_args()  # Parse arguments from command line
    args.mode = "test"
    main(args)


