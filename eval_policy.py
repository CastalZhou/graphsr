import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from model import cls_model, cls_train
import matplotlib.pyplot as plt


def eval_policy(policy, env, test_x, test_y, features, adj_lists, labels):
    obs = env.reset()
    done = 0

    while not done:
        # Query deterministic action from policy and run it
        action = policy(obs).detach()
        obs, rew, done = env.step(action)

    train_x = env.train_node
    train_y = env.train_node_y

    count_sup = np.zeros(7)
    for i in range(98, len(train_x)):
        count_sup[train_y[i]] += 1

    classifer = cls_model(features, adj_lists, 1433, 128)
    final_train = cls_train(classifer, train_x, train_y)
    # final_train = env.cls_train(env.classifier, train_x, train_y)
    pred_test, _ = final_train.forward(test_x)
    print("Test F1:", f1_score(test_y, pred_test.data.numpy().argmax(axis=1), average="micro"))
    print("Test F1:", f1_score(test_y, pred_test.data.numpy().argmax(axis=1), average="macro"))
    one_hot = np.identity(7)[test_y]

    auc = roc_auc_score(one_hot, pred_test.data.numpy(), average='macro')
    print("Test auc:", auc)
