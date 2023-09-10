import numpy as np
import random
import torch
from sklearn import metrics
from collections import deque
from model import Encoder, MeanAggregator, SupervisedGraphSage
from collections import defaultdict
import torch.nn.functional as F
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Env:
    def __init__(self, node_label, node_label_y, node_unlabel, val_x, val_y, features, adj_lists, test_x, test_y):
        self.val_x = val_x
        self.val_y = val_y
        self.train_node_ori = node_label.clone().detach()
        self.train_node_y_ori = node_label_y.clone().detach()
        self.test_x = test_x
        self.test_y = test_y
        self.action_space = 2
        self.classifier = self.cls_model(features, adj_lists, 1433, 128)
        self.unlabel_set = node_unlabel
        self.candidate_node = []
        self.candidate_node_y = []
        self.pred_can = []
        self.emb_can = []
        self.pred_v = []
        self.past_performance = deque(maxlen=10)
        self.class_less = [4, 5, 6]
        self.count_tr_ratio = [20, 20, 20, 20, 6, 6, 6]
        self.count = 0
        self.done = 0
        self.emb_sum = torch.zeros((128, 7))
        self.emb_cen = torch.zeros((128, 7))
        self.mean = 0
        self.f1 = 0
        self.f1_max = 0

    def reset(self):
        self.count = 0
        self.done = 0
        self.train_node = self.train_node_ori
        self.train_node_y = self.train_node_y_ori
        self.supplement_emb = torch.zeros(128)
        # self.past_performance = torch.zeros(10)
        self.past_performance = deque(maxlen=10)
        # self.mean = 0

        pre_train = self.cls_train(50, self.classifier, self.train_node, self.train_node_y)

        pred_l, emb_l = pre_train.forward(self.train_node)
        self.emb_l = torch.sum(emb_l.detach(), 1)   # [128, 98] 按行求和 -> 128

        pred_u, emb_u = pre_train.forward(self.unlabel_set)
        self.candidate_node, self.candidate_node_y = self.SelectNode(emb_l, pred_u, emb_u)

        self.pred_can, self.emb_can = pre_train.forward(self.candidate_node)
        self.pred_v, _ = pre_train.forward(self.val_x)
        self.f1 = metrics.f1_score(self.val_y, self.pred_v.detach().numpy().argmax(axis=1), average="macro")
        self.f1_max = self.f1
        self.past_performance.append(self.f1)

        obs = torch.cat((self.emb_l, self.supplement_emb.detach()), -1)
        obs = torch.cat((obs, self.emb_can[:, self.count].detach()), -1)

        return obs

    def step(self, action_ori):
        action = torch.argmax(F.softmax(action_ori, dim=-1))

        current_node = torch.cat((self.train_node, torch.tensor([self.candidate_node[self.count]])), 0)
        current_y = torch.cat((self.train_node_y, torch.tensor([self.candidate_node_y[self.count]])), 0)

        # if (self.count+1)%10 ==
        retrain = self.cls_train(10, self.classifier, current_node, current_y)
        self.pred_v, _ = retrain.forward(self.val_x)
        f1_score = metrics.f1_score(self.val_y, self.pred_v.detach().numpy().argmax(axis=1), average="macro")
        self.mean = np.mean(list(self.past_performance))

        if action == 1:
            self.supplement_emb = self.supplement_emb + self.emb_can[:, self.count].detach()
            self.train_node = current_node
            self.train_node_y = current_y

            if f1_score - self.mean > 0:
                reward = 1
            elif f1_score - self.mean == 0:
                reward = 0
            else:
                reward = -1

            self.f1 = f1_score

        else:

            if f1_score - self.mean > 0:
                reward = -1
            elif f1_score - self.mean == 0:
                reward = 0
            else:
                reward = 1

        reward = torch.tensor(reward)

        self.count += 1
        self.past_performance.append(self.f1)

        if self.count == len(self.candidate_node):

            self.done = 1
            self.count = 0
            self.supplement_emb = torch.zeros(128)


        obs = torch.cat((self.emb_l, self.supplement_emb), -1)
        self.state = torch.cat((obs, self.emb_can[:, self.count].detach()), -1)

        return self.state, reward, self.done

    def SelectNode(self, emb_l, pred_u, emb_u):  # choose unlabel nodes based on distance
        # calculate centroids of clusters
        emb_sum = torch.zeros((128, 7))

        for i in range(len(self.train_node)):
            emb_sum[:, self.train_node_y[i]] = emb_sum[:, self.train_node_y[i]] + emb_l[:, i]
        emd_cen = torch.zeros((128, 7))
        for i in range(emb_sum.shape[1]):
            emd_cen[:, i] = emb_sum[:, i] / self.count_tr_ratio[i]

        dict_node = defaultdict(list)
        dict_unemb = defaultdict(list)

        pred_y = pred_u.data.numpy().argmax(axis=1)
        for i in range(len(self.unlabel_set)):
            dict_node[pred_y[i]].append(self.unlabel_set[i])
            dict_unemb[pred_y[i]].append(emb_u[:, i].detach().numpy())

        supplement = np.zeros(60, dtype=int)
        supplement_y = np.zeros(60, dtype=int)
        c = 0

        for i in self.class_less:
            cen = emd_cen[:, i].detach().numpy()  # shape=128
            node = np.array(dict_node.get(i))
            emb = np.array(dict_unemb.get(i))

            dis = []
            selnodes = node[0:20]

            for j in range(len(node)):
                distance = np.linalg.norm(cen - emb[j])
                if j < 20:
                    dis.append(distance)
                else:
                    dis_max = max(dis)
                    idx_max = dis.index(dis_max)
                    if distance < dis_max:
                        dis[idx_max] = distance
                        selnodes[idx_max] = node[j]

            dis_node = zip(dis, selnodes)
            dis_node_sort = sorted(dis_node, key=lambda x: x[0])
            dis_sort, selnodes_sort = [list(x) for x in zip(*dis_node_sort)]
            p = 0
            for x in range(len(selnodes)):
                supplement[p+c] = selnodes_sort[x]
                supplement_y[p+c] = i
                p += 3
            c += 1

        return supplement, supplement_y

    def cls_model(self, features, adj_lists, fea_size, hidden):
        agg1 = MeanAggregator(features, cuda=True)
        enc1 = Encoder(features, fea_size, hidden, adj_lists, agg1, gcn=True, cuda=False)
        agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False)
        enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, hidden, adj_lists, agg2,
                       base_model=enc1, gcn=True, cuda=False)
        enc1.num_samples = 5
        enc2.num_samples = 5
        graphsage = SupervisedGraphSage(7, enc2)

        return graphsage

    def cls_train(self, epoch, graphsage, train_x, train_y):
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.7)
        # train_y = labels[np.array(train)].squeeze()
        for batch in range(epoch):
            batch_nodes = train_x[:128]
            batch_y = train_y[:128]
            # random.shuffle(train)
            c = list(zip(train_x, train_y))
            random.shuffle(c)
            train_x, train_y = zip(*c)

            optimizer.zero_grad()
            loss = graphsage.loss(batch_nodes, batch_y)
            loss.backward()
            optimizer.step()

        return graphsage

