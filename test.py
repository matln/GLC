import os
import math
import torch
import logging
import numpy as np
from build_knn import GraphDataset
from scipy.sparse import csr_matrix
from models import HEAD_test, Model
from sklearn.metrics import normalized_mutual_info_score
from rich import print as pprint
from collections import Counter

import sys

sys.path.insert(0, "./")

from metrics import bcubed
from utils.rich_utils import track
from utils.utils import Timer
from utils.logging_utils import init_logger

logger = init_logger()

logger.setLevel(logging.ERROR)


xvector_path = "data/ffsvc2022_train_flt3s/feats.scp"
# xvector_path = "data/ffsvc2022_dev/feats.scp"
k1 = 40
k2 = 40

MODEL_ROOT = f"checkpoints/model_40_80"

knn_path1 = f"knns/ffsvc_k{k1}.npz"
knn_path2 = f"knns/ffsvc_k{k2}.npz"
# knn_path1 = f"knns/ffsvc_dev_k{k1}.npz"
# knn_path2 = f"knns/ffsvc_dev_k{k2}.npz"

torch.cuda.set_device(0)


def _find_parent(parent, u):
    idx = []
    # parent is a fixed point
    while u != parent[u]:
        idx.append(u)
        u = parent[u]
    for i in idx:
        parent[i] = u
    return u


def edge_to_connected_graph(edges, num):
    parent = list(range(num))
    for u, v in edges:
        p_u = _find_parent(parent, u)
        p_v = _find_parent(parent, v)
        parent[p_u] = p_v

    for i in range(num):
        parent[i] = _find_parent(parent, i)
    remap = {}
    uf = np.unique(np.array(parent))
    for i, f in enumerate(uf):
        remap[f] = i
    cluster_id = np.array([remap[f] for f in parent])
    return cluster_id


dataset1 = GraphDataset(
    xvector_path=xvector_path,
    k=k1,
    knn_path=knn_path1,
    force=False,
)
# knns1 = dataset1.knns
nbrs1 = dataset1.nbrs
# sims1 = dataset1.sims
utts = dataset1.utts
# features = torch.FloatTensor(dataset1.features)
nbr_features = dataset1.nbr_features
cos = dataset1.cos
jaccard = dataset1.jaccard
adj1 = dataset1.adj
gt_labels1 = dataset1.gt_labels

dataset2 = GraphDataset(
    xvector_path=xvector_path,
    k=k2,
    knn_path=knn_path2,
    force=False,
)
knns2 = dataset2.knns
density = dataset2.density

n = len(knns2)
nbrs = knns2[:, 0, :]
edges = []
score = []
inst_num = knns2.shape[0]
k_num = knns2.shape[2]
logger.info(f"inst_num: {inst_num}")


def main(epoch, lambda1, inst_num):
    model_path = f"{MODEL_ROOT}/Backbone_Epoch_{epoch}.pth"
    head_path = f"{MODEL_ROOT}/Head_Epoch_{epoch}.pth"
    logger.info(f"knn_path1: {knn_path1}, model_path: {model_path}, head_path: {head_path}")
    pprint(f"knn_path1: {knn_path1}, model_path: {model_path}, head_path: {head_path}")
    pprint(k1, k2)

    with Timer():
        # print(**cfg.model['kwargs'])
        model = Model(input_dim=512, out_dim=128, num_heads=1, dropout=0, leaky_alpha=0.1)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        HEAD_test1 = HEAD_test(nhid=512)
        HEAD_test1.load_state_dict(torch.load(head_path, map_location="cpu"))

        pair_a = []
        pair_b = []
        pair_a_new = []
        pair_b_new = []
        for i in range(inst_num):
            pair_a.extend([int(i)] * k_num)
            pair_b.extend([int(j) for j in nbrs[i]])
        for i in range(len(pair_a)):
            if pair_a[i] != pair_b[i]:
                pair_a_new.extend([pair_a[i]])
                pair_b_new.extend([pair_b[i]])
        pair_a = pair_a_new
        pair_b = pair_b_new

        pprint(len(pair_a))
        inst_num = len(pair_a)

        model.cuda()
        HEAD_test1.cuda()
        adj = adj1.cuda()
        labels = torch.from_numpy(gt_labels1).cuda()

        model.eval()
        HEAD_test1.eval()

        disconnect_num = 0
        positive_num = 0
        negative_num = 0
        TP_num = 0
        TN_num = 0
        edges = None
        for threshold1 in [lambda1]:
            with torch.no_grad():
                # Get the mean and var of the testset
                batch_num = 5
                batch_size = math.ceil(knns2.shape[0] / batch_num)
                for i in track(range(batch_num)):
                    # batch_nbrs = nbrs1[i * batch_size : (i + 1) * batch_size, :].astype(np.int64)
                    # batch_sims = sims1[i * batch_size : (i + 1) * batch_size, :]
                    # sorted_idx = torch.argsort(torch.from_numpy(batch_sims), descending=True, dim=1)
                    # sorted_batch_nbrs = torch.gather(torch.from_numpy(batch_nbrs), 1, sorted_idx)
                    # batch_feats = features[sorted_batch_nbrs]

                    batch_feats = nbr_features[i * batch_size : (i + 1) * batch_size]
                    batch_cos = cos[i * batch_size : (i + 1) * batch_size]
                    batch_jaccard = jaccard[i * batch_size : (i + 1) * batch_size]
                    out_feature = model(batch_feats.cuda(), batch_cos.cuda(), batch_jaccard.cuda())
                    if i == 0:
                        out_features = out_feature
                    else:
                        out_features = torch.cat((out_features, out_feature), dim=0)
                bn_weight = model.bn.weight
                bn_bias = model.bn.bias
                bn_mean = torch.mean(out_features, dim=0)
                bn_var = torch.var(out_features, dim=0, unbiased=False)
                out_features = (out_features - bn_mean) / torch.sqrt(bn_var + 1e-5) * bn_weight + bn_bias

                patch_num = 30
                patch_size = math.ceil(inst_num / patch_num)
                for i in track(range(patch_num)):
                    id1 = pair_a[i * patch_size : (i + 1) * patch_size]
                    id2 = pair_b[i * patch_size : (i + 1) * patch_size]
                    batch_nodes = list(Counter([*id1, *id2]).keys())
                    index_dict = {element: index for index, element in enumerate(batch_nodes)}
                    # 使用字典进行快速查找
                    rel_id1 = [index_dict[element] for element in id1]
                    rel_id2 = [index_dict[element] for element in id2]

                    output_feature = out_features[batch_nodes]

                    score_ = HEAD_test1(output_feature[rel_id1], output_feature[rel_id2])
                    score_ = np.array(score_)
                    idx = np.where(score_ > threshold1)[0].tolist()

                    patch_label = (labels[id1] == labels[id2]).long()
                    positive_idx = torch.nonzero(patch_label).squeeze(1).tolist()
                    negative_idx = torch.nonzero(patch_label - 1).squeeze(1).tolist()
                    TP_num += len(np.where(score_[positive_idx] > threshold1)[0].tolist())
                    TN_num += len(np.where(score_[negative_idx] < threshold1)[0].tolist())

                    # idx = positive_idx

                    negative_num += len(negative_idx)
                    positive_num += len(positive_idx)
                    disconnect_num += len(score_) - len(idx)

                    # score.extend(score_[idx].tolist())
                    id1 = np.array(id1)
                    id2 = np.array(id2)
                    id1 = np.array([id1[idx].tolist()])
                    id2 = np.array([id2[idx].tolist()])
                    if edges is None:
                        edges = np.concatenate([id1, id2], 0).transpose()
                    else:
                        if len(idx) != 0:
                            edges = np.concatenate([edges, np.concatenate([id1, id2], 0).transpose()], 0)

            # with Timer("edge_to_connected_graph"):
            labels1 = edge_to_connected_graph(edges.tolist(), n)

            nmi_1 = normalized_mutual_info_score(gt_labels1, labels1)
            b_pre, b_rec, b_fscore_1 = bcubed(gt_labels1, labels1)
            pprint(f"total clusters: {max(labels1)}, nmi_1: {nmi_1:.3f}, "
                  f"b_fscore: {b_fscore_1:.3f}, b_pre: {b_pre:.3f}, b_rec: {b_rec:.3f}")
            # pprint(f"model_path: {model_path}")

            labels2 = labels1

            # Filter
            clt2idx = {}
            for i, clt in enumerate(labels2):
                clt2idx.setdefault(clt, []).append(i)
            remain_idx = [v for k, v in clt2idx.items() if len(v) >= 10]
            remain_idx = [x for sub_lst in remain_idx for x in sub_lst]
            remain_labels2 = labels2[remain_idx]
            remain_gt_labels1 = gt_labels1[remain_idx]
            nmi_3 = normalized_mutual_info_score(remain_gt_labels1, remain_labels2)
            b_pre, b_rec, b_fscore_3 = bcubed(remain_gt_labels1, remain_labels2)
            pprint(f"total clusters: {len(list(set(remain_labels2)))}, nmi_3: {nmi_3:.3f}, "
                  f"b_fscore: {b_fscore_3:.3f}, b_pre: {b_pre:.3f}, b_rec: {b_rec:.3f}")
            counter_lbl = Counter(remain_labels2)
            sorted_counter = sorted(counter_lbl.items(), key=lambda x: x[1], reverse=True)
            print(f"remain: {len(remain_labels2)}, total: {len(labels2)}")
            print(dict(sorted_counter[:10]))

            cluster2lbl = {}
            lbl = 0
            with open("utt2cluster", "w") as fw:
                for idx in remain_idx:
                    utt = utts[idx]
                    _cluster = labels2[idx]
                    if _cluster not in cluster2lbl:
                        cluster2lbl[_cluster] = lbl
                        lbl += 1
                    fw.write(f"{utt} {cluster2lbl[_cluster]}\n")



if __name__ == "__main__":
    for lambda1 in [0.75]:
        main("1", lambda1, inst_num)
