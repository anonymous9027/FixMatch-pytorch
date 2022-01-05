import torch.nn.functional as F
import torch
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
import torch.nn as nn


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    # print(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def cosine_dist(x, y):
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    dist = torch.mm(x, y.t())
    return dist


def loss_ort(b1, b2):
    # print(b1.shape)
    b1 = F.softmax(b1).t()
    b2 = F.softmax(b2).t()
    dist_mat = cosine_dist(b1, b2)
    t = np.array([i for i in range(dist_mat.shape[0])])
    onehot = torch.tensor(t, dtype=torch.long).cuda()
    # print(onehot)
    return (F.cross_entropy(dist_mat, onehot, reduction='mean')+F.cross_entropy(dist_mat.t(), onehot, reduction='mean'))*0.5

def loss_sel(b1,b2):
    b1=torch.pow(F.softmax(b1),2)
    b2=torch.pow(F.softmax(b2),2)
    n,m = b1.shape
    b1 = b1/ torch.sum(b1,dim=1,keepdim=True).expand(n,m)
    b2= b2 /torch.sum(b2,dim=1,keepdim=True).expand(n,m)
    #print(b1[0],torch.sum(b1[0]))
    loss = b1*torch.log(b2+1e-6)+b2*torch.log(b1+1e-6)

    return - loss.mean()



class Memory(nn.Module):
    """
    Build a MoCo memory with a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, dim=512, K=8192):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        """
        super().__init__()
        self.K = K

        self.margin = 0.25
        self.gamma = 32

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)

        self.register_buffer(
            "queue_label", torch.zeros((1, K), dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, targets):
        # gather keys/targets before updating queue
        keys = keys.detach()
        targets = targets.detach()
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_label[:, ptr:ptr + batch_size] = targets
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, emb_u_w, emb_u_s, emb_l_w, emb_l_s,index_u,index_l,sim):

        self._dequeue_and_enqueue(torch.cat((emb_u_w,emb_l_w)),torch.cat((index_u,index_l)))
        #print(sim.shape)
        la=torch.cat((index_u,index_l)).detach().cpu().numpy().tolist() 
        lb=self.queue_label.detach().cpu().numpy().tolist()
        ispos=[]
        for i in lb[0]:
            #print(len(la),i)
            ispos.append(sim[i,la])
        ispos=np.array(ispos,dtype='double').T
        #print(ispos.shape)

        is_pos = torch.tensor(ispos).cuda()
        l_con = self.loss_con(torch.cat((emb_u_s,emb_l_s)), self.queue.T.cuda(),is_pos)
        return l_con

    def loss_con(self,d1, d2, is_pos, T=1):
        dist_mat = cosine_dist(d1, d2)
        probs1 = F.softmax(dist_mat)  # exp(i)/ \sum exp
        loss1 = torch.log((is_pos.float()*probs1.float()+1e-6).sum(dim=1))
        loss1 = F.softplus(loss1).mean()
        '''probs2 = F.softmax(dist_mat).T  # exp(i)/ \sum exp
        loss2 = torch.log((is_pos*probs2).sum(dim=1))
        loss2 = F.softplus(loss2).mean()'''

        return loss1.sum()
