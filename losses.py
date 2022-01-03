import torch.nn.functional as F
import torch
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment


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
