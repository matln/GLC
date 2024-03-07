import os
import sys
import torch
import random
import logging
import numpy as np
import torch.optim as optim
from rich import print
from collections import Counter
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

sys.path.insert(0, "./")
from models import HEAD, Model
from build_knn import GraphDataset
import utils.utils as utils
from utils.rich_utils import track
from utils.logging_utils import init_logger

logger = init_logger()

tot_epochs = 1
seed = 1024
k1 = 40
k2 = 80
os.makedirs("checkpoints/", exist_ok=True)
MODEL_ROOT = f"checkpoints/model_{k1}_{k2}"

xvector_path = "data/voxceleb2_dev_sub/feats.scp"

os.makedirs("knns/", exist_ok=True)
knn_path1 = f"knns/vox_k{k1}.npz"
knn_path2 = f"knns/vox_k{k2}.npz"

torch.cuda.set_device(0)
utils.set_seed(seed, deterministic=True)


def main():
    # prepare dataset
    logger.info("Loading the training data...")
    dataset = GraphDataset(
        xvector_path=xvector_path,
        k=k1,
        knn_path=knn_path1,
        force=False,
    )
    # features = torch.FloatTensor(dataset.features)
    nbr_features = dataset.nbr_features
    jaccard = dataset.jaccard
    cos = dataset.cos
    labels = torch.LongTensor(dataset.gt_labels)
    logger.info("Have loaded the training data.")

    logger.info("Loading the larger knns ...")
    dataset2 = GraphDataset(
        xvector_path=xvector_path,
        k=k2,
        knn_path=knn_path2,
        force=False,
    )
    adj2 = dataset2.adj

    feature_dim = dataset.feature_dim
    assert feature_dim == 512

    model = Model(input_dim=512, out_dim=128, num_heads=1, dropout=0, leaky_alpha=0.5)
    HEAD1 = HEAD(nhid=512)

    optimizer = optim.SGD(
        [
            {"params": model.parameters(), "weight_decay": 1e-5},
            {"params": HEAD1.parameters(), "weight_decay": 1e-5},
        ],
        lr=0.01,
        momentum=0.9,
    )
    logger.info("the learning rate is 0.01")
    # lr_scheduler = optim.lr_scheduler.MultiStepLR(
    #     optimizer,
    #     milestones=[int(tot_epochs * 0.5), int(tot_epochs * 0.8), int(tot_epochs * 0.9)],
    #     gamma=0.1,
    # )

    model = model.cuda()
    HEAD1 = HEAD1.cuda()
    # 1. put selected feature and labels to cuda

    logger.info(f"the model save path is {MODEL_ROOT}")
    if not os.path.exists(MODEL_ROOT):
        os.makedirs(MODEL_ROOT)

    for epoch in range(tot_epochs):
        model.train()
        HEAD1.train()

        # 2. train the model
        train_id_inst = adj2._indices().size()[1]
        logger.info(f"train_id_inst: {train_id_inst}")

        # utils.set_seed(1024, deterministic=True)
        rad_id = torch.randperm(train_id_inst).tolist()
        patch_num = 500
        for i in track(range(patch_num)):
            optimizer.zero_grad()

            ids = rad_id[
                i * int(train_id_inst / patch_num) : (i + 1) * int(train_id_inst / patch_num)
            ]
            batch_row2 = adj2._indices()[0, ids].tolist()
            batch_col2 = adj2._indices()[1, ids].tolist()
            batch_label = (labels[batch_row2] == labels[batch_col2]).long()

            pos_idx = torch.nonzero(batch_label).squeeze(1).tolist()
            neg_idx = torch.nonzero(batch_label - 1).squeeze(1).tolist()
            pos_idx = random.sample(pos_idx, len(neg_idx))
            batch_idx = pos_idx + neg_idx
            batch_row2 = list(np.array(batch_row2)[batch_idx])
            batch_col2 = list(np.array(batch_col2)[batch_idx])
            batch_label = (labels[batch_row2] == labels[batch_col2]).long()


            batch_nodes = list(Counter([*batch_row2, *batch_col2]).keys())
            index_dict = {element: index for index, element in enumerate(batch_nodes)}
            rel_batch_row2 = [index_dict[element] for element in batch_row2]
            rel_batch_col2 = [index_dict[element] for element in batch_col2]


            # # Get batch-wise adj
            # # with utils.Timer():
            # #     batch_knns = knns[batch_nodes]
            # #     batch_nbrs = batch_knns[:, 0, :]
            # #     batch_sims = 1.0 - batch_knns[:, 1, :]
            # #     rel_batch_row, batch_col = np.where(batch_sims > 0)
            # #     batch_col = batch_nbrs[rel_batch_row, batch_col]
            # # rel_batch_edges = torch.from_numpy(np.vstack((rel_batch_row, batch_col)).astype(np.int64)).cuda()

            # # abs_batch_row = np.repeat(batch_nodes, k1)
            # # abs_batch_edges = torch.from_numpy(np.vstack((abs_batch_row, batch_col)).astype(np.int64))

            # # print(batch_nodes[0])
            # # print(col[np.where(row == batch_nodes[0])[0]])

            # batch_nbrs = nbrs[batch_nodes, :].astype(np.int64)
            # batch_sims = sims[batch_nodes, :]
            # sorted_idx = torch.argsort(torch.from_numpy(batch_sims), descending=True, dim=1)
            # # sorted_batch_nbrs = torch.from_numpy(batch_nbrs)[sorted_idx]
            # sorted_batch_nbrs = torch.gather(torch.from_numpy(batch_nbrs), 1, sorted_idx)
            # sorted_batch_sims = torch.gather(torch.from_numpy(batch_sims), 1, sorted_idx)

            # # with utils.Timer():
            # # batch_feats = features[sorted_batch_nbrs]

            # row_size, col_size = sorted_batch_nbrs.size()
            # batch_feats = features.index_select(0, sorted_batch_nbrs.reshape(-1))
            # batch_feats = batch_feats.reshape((row_size, col_size, features.size(1)))


            batch_feats = nbr_features.index_select(0, torch.LongTensor(batch_nodes))
            batch_cos = cos.index_select(0, torch.LongTensor(batch_nodes))
            batch_jaccard = jaccard.index_select(0, torch.LongTensor(batch_nodes))

            x = model(batch_feats.cuda(), batch_cos.cuda(), batch_jaccard.cuda())
            x = model.bn(x)
            loss, pos_acc, neg_acc = HEAD1(x, batch_label.cuda(), rel_batch_row2, rel_batch_col2)

            loss.backward()
            optimizer.step()

            print(
                "epoch: {}/{}, patch: {}/{}, loss: {:.4f}, pos_acc: {:.4f}, neg_acc: {:.4f}".format(
                    epoch + 1, tot_epochs, i, patch_num, loss, pos_acc, neg_acc
                )
            )
            if i % 100 == 0:
                torch.save(
                    model.state_dict(), os.path.join(MODEL_ROOT, f"Backbone_Epoch_{epoch + 1}.{i}.pth"),
                )
                torch.save(
                    HEAD1.state_dict(), os.path.join(MODEL_ROOT, f"Head_Epoch_{epoch + 1}.{i}.pth"),
                )

        # lr_scheduler.step(epoch)

        # 3. save model
        logger.info("save model in epoch:{} to {}".format(epoch, MODEL_ROOT))
        torch.save(
            model.state_dict(), os.path.join(MODEL_ROOT, "Backbone_Epoch_{}.pth".format(epoch + 1)),
        )
        torch.save(
            HEAD1.state_dict(), os.path.join(MODEL_ROOT, "Head_Epoch_{}.pth".format(epoch + 1)),
        )


if __name__ == "__main__":
    main()
