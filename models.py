import torch
import random
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """
    """
    def __init__(self, att_head, in_dim, out_dim, dp, leaky_alpha=0.2):
        super(AttentionLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dp = dp

        self.beta = nn.Parameter(torch.tensor(1.0, requires_grad=True))

        self.att_head = att_head
        # self.W = nn.Parameter(torch.Tensor(self.att_head, self.in_dim, self.out_dim))
        self.W = nn.Parameter(torch.Tensor(self.att_head, self.in_dim + 40, self.out_dim))

        self.w_self = nn.Parameter(torch.Tensor(self.att_head, self.out_dim, 1))
        self.w_nbr = nn.Parameter(torch.Tensor(self.att_head, self.out_dim, 1))
        self.leaky_alpha = leaky_alpha
        self.init_att_param()

        # assert self.in_dim == self.out_dim * self.att_head

    def init_att_param(self):
        init.xavier_uniform_(self.W.data)
        init.xavier_uniform_(self.w_self.data)
        init.xavier_uniform_(self.w_nbr.data)

        init.constant_(self.beta.data, 0.1)

    def forward(self, feat_in, cos, jaccard):
        batch, K, in_dim = feat_in.size()
        assert in_dim == self.in_dim

        # # feat_in_: [64, 1, 80, 512]
        # # self.W: [4, 512, 128]
        # # h: [64, 4, 80, 128]
        # # feat_in_ = feat_in.unsqueeze(1)
        # h = torch.matmul(torch.cat((feat_in, cos), dim=-1).unsqueeze(1), self.W)
        # # h = torch.matmul(feat_in.unsqueeze(1), self.W)

        # # h_0: [64, 4, 1, 128]
        # # self.w_self: [4, 128, 1]
        # # attn_self: [64, 4, 1, 1]
        # h_0 = h[:, :, :1, :]
        # attn_self = torch.matmul(h_0, self.w_self)
        # # h: [64, 4, 80, 128]
        # # self.w_nbr: [4, 128, 1]
        # # attn_nbr: [64, 4, 80, 1]
        # attn_nbr = torch.matmul(h, self.w_nbr)
        # # attn_self.expand(-1, -1, -1, K): [64, 4, 1, 80]
        # attn = attn_self.expand(-1, -1, -1, K) + attn_nbr.permute(0, 1, 3, 2)
        # # attn: [64, 4, 1, 80]
        # attn = F.leaky_relu(attn, self.leaky_alpha, inplace=True)
        # attn = F.softmax(attn, dim=-1)

        attn = F.softmax(self.beta * jaccard, dim=-1).unsqueeze(1).unsqueeze(1)

        ## cosine
        # attn = cos.unsqueeze(1).unsqueeze(1)
        # attn = attn / (torch.sum(attn, dim=-1, keepdim=True) + 1e-5)


        # torch.matmul(attn, feat_in_):
        # attn: [64, 4, 1, 80]
        # feat_in_: [64, 1, 80, 512]
        # feat_out: [64, 4, 1, 512] -> [64, 1, 4, 512] -> [64, 1, 2048]
        feat_out = torch.matmul(attn, feat_in.unsqueeze(1))
        # feat_out = feat_in[:, 0, :].unsqueeze(1).unsqueeze(1)

        # attn: [64, 4, 1, 80]
        # h: [64, 4, 80, 128]
        # feat_out: [64, 4, 1, 128] -> [64, 1, 4, 128] -> [64, 1, 512]
        # feat_out = torch.matmul(attn, h)

        # feat_out: [64, 512]
        feat_out = feat_out.transpose(1, 2).contiguous().view(batch, 1, -1).squeeze()

        feat_out = F.dropout(feat_out, self.dp, training=self.training)
        # print(feat_out[:2, :5])

        return feat_out

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_dim) + ' -> ' + str(self.out_dim*self.att_head) + ')'


class Model(nn.Module):
    def __init__(self, input_dim=512, out_dim=128, num_heads=4, dropout=0, leaky_alpha=0.2):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.att_layer = AttentionLayer(self.num_heads, self.input_dim, self.out_dim, dropout, leaky_alpha)
        self.bn = nn.BatchNorm1d(input_dim, momentum=0.5)

    def forward(self, seq_h, cos, jaccard):
        batch, max_len, _ = seq_h.size()

        seq_h = self.att_layer(seq_h, cos, jaccard)

        self.bn.training = True
        # seq_h = self.bn(seq_h)

        return seq_h


def compute_acc(outs, labels):
    preds = torch.squeeze(torch.argmax(outs.detach(), dim=1))
    num_correct = (labels == preds).sum()
    if len(labels) == 0:
        return 0
    acc = num_correct.item() / len(labels)
    return acc


class HEAD(nn.Module):
    def __init__(self, nhid, dropout=0):
        super(HEAD, self).__init__()

        self.nhid = nhid
        self.classifier = nn.Sequential(
            nn.Linear(nhid * 2, nhid),
            nn.PReLU(nhid),
            nn.Linear(nhid, 2)
        )
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, feature, batch_label, row, col):
        assert len(feature.shape) == 2
        # print(feature[row][10:12, :5], feature[col][10:12, :5])
        pred = self.classifier(torch.cat((feature[row], feature[col]), 1))

        pos_idx = torch.nonzero(batch_label).squeeze(1).tolist()
        neg_idx = torch.nonzero(batch_label - 1).squeeze(1).tolist()
        pos_pred = pred[pos_idx]
        neg_pred = pred[neg_idx]
        pos_label = batch_label[pos_idx]
        neg_label = batch_label[neg_idx]
        print(len(pos_idx), len(neg_idx))

        # loss
        pos_loss = self.loss(pos_pred, pos_label)
        neg_loss = self.loss(neg_pred, neg_label)
        loss = 1 * pos_loss + 4 * neg_loss

        # Acc
        pos_acc = compute_acc(pos_pred, pos_label)
        neg_acc = compute_acc(neg_pred, neg_label)
        return loss, pos_acc, neg_acc


class HEAD_test(nn.Module):
    def __init__(self, nhid):
        super(HEAD_test, self).__init__()

        self.nhid = nhid
        self.classifier = nn.Sequential(
            nn.Linear(nhid * 2, nhid), nn.PReLU(nhid), nn.Linear(nhid, 2), nn.Softmax(dim=1)
        )

    def forward(self, feature1, feature2, no_list=False):
        if len(feature1.size()) == 1:
            pred = self.classifier(torch.cat((feature1, feature2)).unsqueeze(0))
            if pred[0][0] > pred[0][1]:
                is_same = False
            else:
                is_same = True
            return is_same
        else:
            pred = self.classifier(torch.cat((feature1, feature2), 1))
            # print(pred[0:10,:])
            if no_list:
                return pred[:, 1]
            score = list(pred[:, 1].cpu().detach().numpy())
            # is_same = (pred[:,0]<pred[:,1]).long()

        return score


if __name__ == "__main__":
    pass
