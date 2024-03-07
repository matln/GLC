import os
import kaldiio
import faiss
import time
import torch
import logging
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import scipy.sparse as sp
from scipy.sparse import csr_matrix

import sys
sys.path.insert(0, "./")

from utils.rich_utils import track
from utils.utils import Timer
from rich import print

# Logger
logger = logging.getLogger("fit_progressbar")
root_logger = logging.getLogger(__name__)

eps = 1e-2
th_sim = 0.


class GraphDataset(object):
    def __init__(self, xvector_path, k, knn_path, scp=True, force=False, use_sim = True, weighted=True):
        # 1. Load features and labels
        # root_logger.info("Load features and labels...")
        print("Load features and labels...")
        if scp:
            features = []
            spk2label = {}
            gt_labels = []
            utts = []
            num = 0
            # for idx, (utt, feat) in enumerate(track(kaldiio.load_scp_sequential(xvector_path))):
            for idx, (utt, feat) in enumerate(kaldiio.load_scp_sequential(xvector_path)):
                spk = utt.split("-")[0]
                if spk not in spk2label:
                    spk2label[spk] = num
                    num += 1
                gt_labels.append(spk2label[spk])
                utts.append(utt)
                features.append(feat)
            _features = np.array(features)
            self.features = _features
            features = _features / np.linalg.norm(_features, axis=1).reshape(-1, 1)
            # self.features = features
            self.feature_dim = self.features.shape[1]
            self.gt_labels = np.array(gt_labels)
            self.inst_num = len(features)
            self.utts = utts
        else:
            _features = np.load(xvector_path)
            self.features = _features
            features = _features / np.linalg.norm(_features, axis=1).reshape(-1, 1)
            self.feature_dim = self.features.shape[1]

        # 2. Build knns
        if not os.path.exists(knn_path) or force:
            # root_logger.info("Build knns...")
            print("Build knns...")
            features = features.astype("float32")
            size, dim = features.shape
            index = faiss.IndexFlatIP(dim)
            index.add(features)
            with Timer():
                sims, nbrs = index.search(features, k=k)

            if weighted:
                knns = [
                    (
                        np.array(nbr, dtype=np.int32),
                        1 - np.minimum(np.maximum(np.array(sim, dtype=np.float32), 0), 1),
                    )
                    for nbr, sim in zip(nbrs, sims)
                ]
            else:
                knns = [
                    (
                        np.array(nbr, dtype=np.int32),
                        np.ones_like(np.array(sim, dtype=np.float32)),
                    )
                    for nbr, sim in zip(nbrs, sims)
                ]
                use_sim = False
            # Save knns
            # root_logger.info("Save knns...")
            print("Save knns...")
            np.savez_compressed(knn_path, data=knns)
        else:
            # Read knns
            # root_logger.info("Read knns...")
            print("Read knns...")
            knns = np.load(knn_path, allow_pickle=True)['data']
        self.knns = np.array(knns)

        # 3. Convert knns to sparse matrix
        # root_logger.info("Convert knns to sparse matrix...")
        print("Convert knns to sparse matrix...")
        n = len(knns)
        if isinstance(knns, list):
            knns = np.array(knns)
        nbrs = knns[:, 0, :]
        self.nbrs = nbrs
        dists = knns[:, 1, :]
        assert -eps <= dists.min() <= dists.max() <= 1 + eps, "min: {}, max: {}".format(dists.min(), dists.max())
        if use_sim:
            sims = 1. - dists
        else:
            sims = dists
        row, col = np.where(sims >= th_sim)
        data = sims[row, col]
        # data = np.array([1.] * len(data))
        
        col = nbrs[row, col]  # convert to absolute column
        assert len(row) == len(col) == len(data)

        self.density = np.mean(sims, axis=1)

        adj = csr_matrix((data, (row, col)), shape=(n, n))

        nbrs = torch.from_numpy(nbrs.astype(np.int64))
        nbr_features = torch.FloatTensor(self.features).index_select(0, nbrs.reshape(-1))
        self.nbr_features = nbr_features.reshape((n, k, self.feature_dim))

        # root_logger.info("Compute Jaccard Similarity...")
        print("Compute Jaccard Similarity...")
        tmp_data = np.array([1.] * len(data))
        tmp_adj = csr_matrix((tmp_data, (row, col)), shape=(n, n))
        common_link = tmp_adj.dot(tmp_adj.T)
        link_num = np.array(tmp_adj.sum(axis=1))
        share_num = common_link[row.tolist(), col.tolist()].reshape(-1, 1)
        jaccard = share_num / (link_num[row.astype(np.int64)] + link_num[col.astype(np.int64)] - share_num + 1e-8)
        jaccard = np.squeeze(np.array(jaccard))
        self.jaccard = torch.FloatTensor(np.reshape(jaccard, (n, k)))
        self.cos = torch.FloatTensor(sims)

        indices, values, shape = self.normalize_sp(adj)
        self.adj = torch.sparse.FloatTensor(indices, values, shape)

    def normalize_sp(self, adj):
        # 6. Convert sparse matrix to indices values
        # root_logger.info("Convert sparse matrix to indices values...")
        print("Convert sparse matrix to indices values...")
        sparse_mx = adj.tocoo().astype(np.float32)
        # print(sparse_mx)
        adj_indices = np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
        adj_values = sparse_mx.data
        adj_shape = np.array(sparse_mx.shape)
        # print(adj_indices.shape)
        # print(adj_values.shape)

        indices = torch.from_numpy(adj_indices)
        values = torch.from_numpy(adj_values)
        shape = torch.Size(adj_shape)
        return indices, values, shape


if __name__ == "__main__":
    dataset = GraphDataset()
