import argparse
import logging
import math
import os
import random
import shutil
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# from dataset.cifar import DATASET_GETTERS
from dataset.labeledcifar import GetCifar
from utils import AverageMeter, accuracy
from losses import loss_ort

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100'],
                        help='dataset name')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='resnet18', type=str,
                        choices=['wideresnet', 'resnext',
                                 'resnet18', 'resnet50'],
                        help='dataset name')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--seenclass-number', type=int, default=10)
    parser.add_argument('--top-k', type=int, default=30)
    args = parser.parse_args()
    global best_acc

    def create_model(args):
        if args.arch == 'wideresnet':
            import models.wideresnet as models
            model = models.build_wideresnet(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0,
                                            num_classes=args.num_classes)
        elif args.arch == 'resnext':
            import models.resnext as models
            model = models.build_resnext(cardinality=args.model_cardinality,
                                         depth=args.model_depth,
                                         width=args.model_width,
                                         num_classes=args.num_classes)
        elif args.arch == 'resnet18':
            from models.resnet import resnet18
            model = resnet18(num_classes=args.num_classes)
        elif args.arch == 'resnet50':
            from models.resnet import resnet50
            model = resnet50(num_classes=args.num_classes)
        else:
            raise NotImplementedError
        logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters()) / 1e6))
        return model

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, ")

    logger.info(dict(args._get_kwargs()))
    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)

    if args.dataset == 'cifar10':
        args.num_classes = 10
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == 'cifar100':
        args.num_classes = 100
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 8
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if args.dataset.startswith('cifar'):
        labeled_dataset, unlabeled_dataset, test_dataset = GetCifar(
            args.dataset, args.seenclass_number, 0.5, 1)
    else:
        raise NotImplementedError
    # labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](args, './data')

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)
    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model = create_model(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    args.start_epoch = 0

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)
    res = []
    ind = []
    model.eval()
    for batch_idx, (inputs, targets, indexes) in enumerate(labeled_trainloader):
        embs, _ = model(inputs.cuda())
        for i in range(embs.shape[0]):
            res.append(embs[i].cpu().detach().numpy())
            ind.append(indexes[i].cpu().detach().numpy())
    for batch_idx, (inputs, targets, indexes) in enumerate(unlabeled_trainloader):
        embs, _ = model(inputs.cuda())
        for i in range(embs.shape[0]):
            res.append(embs[i].cpu().detach().numpy())
            ind.append(indexes[i].cpu().detach().numpy())

    '''for batch_idx, (inputs, targets, indexes) in enumerate(test_loader):
        embs, _ = model(inputs.cuda())
        for i in range(embs.shape[0]):
            res.append(embs[i].cpu().detach().numpy())
            ind.append(indexes[i].cpu().detach().numpy())'''
    embs = np.zeros((len(res), res[0].shape[0]))
    for i in range(len(res)):
        embs[ind[i]] = res[i]
    embs = torch.tensor(embs)
    from losses import cosine_dist
    dist = cosine_dist(embs, embs)
    _, topkindex = torch.topk(dist, k=args.top_k, dim=1)
    del dist
    print(topkindex.shape, topkindex[0:10])
    sim = np.zeros((len(res), len(res)))
    for i in range(len(res)):
        sim[i][topkindex[i].cpu().numpy().tolist()] = 1
    # print(sim[0:10],sim[0,13445])
    res = sim * sim.T
    #sav = {"embs": embs, "sim": res}
    import joblib
    joblib.dump(res, open(args.out+'/res.pkl', 'wb'))


if __name__ == '__main__':
    main()
