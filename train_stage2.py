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
from losses import loss_ort, loss_sel

logger = logging.getLogger(__name__)
best_acc = 0


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    # print("tmp",s, [-1, size] + s[1:],[-1] + s[1:])
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100'],
                        help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=4000,
                        help='number of labeled data')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='resnet18', type=str,
                        choices=['wideresnet', 'resnext',
                                 'resnet18', 'resnet50'],
                        help='dataset name')
    parser.add_argument('--total-steps', default=2 ** 20, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=1024, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--seenclass-number', type=int, default=5)
    parser.add_argument('--labeled-ratio', type=float, default=0.5)
    parser.add_argument('--embed_mat', type=str, default='result/res.pkl')
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
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}", )

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

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
            args.dataset, args.seenclass_number, args.labeled_ratio)
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
        batch_size=args.batch_size * args.mu,
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

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

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
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    if args.amp:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.opt_level)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)
    if args.eval_only:
        model.eval()
        lossesavg, seen_top1avg, novel_accavg, all_accavg = test(
            args, test_loader, model, 0,  args.seenclass_number)
        print(lossesavg, seen_top1avg, novel_accavg, all_accavg)
        return
    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size * args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    model.zero_grad()
    train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler)


def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler):
    if args.amp:
        from apex import amp
    global best_acc

    import joblib
    from losses import Memory
    sims = joblib.load(open(args.embed_mat, 'rb'))
    mem1= Memory()
    mem2= Memory()
    test_accs = []
    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    model.train()
    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_sup = AverageMeter()
        losses_ort = AverageMeter()
        losses_sel = AverageMeter()
        losses_con = AverageMeter()
        # mask_probs = AverageMeter()
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])
        for batch_idx in range(args.eval_step):
            try:
                (inputs_l_w, inputs_l_s), targets_x, index_l = labeled_iter.next()
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                (inputs_l_w, inputs_l_s), targets_x, index_l = labeled_iter.next()

            try:
                (inputs_u_w, inputs_u_s), _, index_u = unlabeled_iter.next()
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), _, index_u = unlabeled_iter.next()

            data_time.update(time.time() - end)
            # print(index_l)
            # inputs_x = inputs_l_s
            batch_size = inputs_l_w.shape[0]
            # print('shape1', torch.cat((inputs_l_w, inputs_u_w, inputs_u_s)).shape,batch_size, inputs_l_s.shape)

            inputs = interleave(
                torch.cat((inputs_l_w, inputs_l_s, inputs_u_w, inputs_u_s)), 2 * args.mu + 2).to(args.device)
            # print('shape2', inputs.shape )

            targets_x = targets_x.to(args.device)
            embeddings, logits = model(inputs)

            # print('shape3', logits.shape)
            logits = de_interleave(logits, 2 * args.mu + 2)
            embeddings = de_interleave(embeddings, 2 * args.mu + 2)
            # print('shape4', logits.shape)

            logits_l_w, logits_l_s = logits[:batch_size * 2].chunk(2)
            logits_u_w, logits_u_s = logits[batch_size * 2:].chunk(2)
            del logits
            
            embeddings_l_w, embeddings_l_s = embeddings[:batch_size * 2].chunk(
                2)
            embeddings_u_w, embeddings_u_s = embeddings[batch_size * 2:].chunk(
                2)
            del embeddings
            
            L_sup = (F.cross_entropy(logits_l_w, targets_x, reduction='mean') + F.cross_entropy(logits_l_s, targets_x,
                                                                                                reduction='mean')) * 0.5
            L_ort = loss_ort(torch.cat((logits_l_w, logits_u_w)),torch.cat((logits_l_s, logits_u_s)))

            L_sel = loss_sel(logits_u_s,logits_u_w)

            L_con = mem1(embeddings_u_w, embeddings_u_s, embeddings_l_w, embeddings_l_s,index_u,index_l,sims)+mem2(embeddings_u_s, embeddings_u_w, embeddings_l_s, embeddings_l_w,index_u,index_l,sims)
            #L_ort = torch.tensor(0).cuda()
            #       torch.cat((logits_l_s, logits_u_s)))
            '''pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()

            Lu = (F.cross_entropy(logits_u_s, targets_u,
                                  reduction='none') * mask).mean()

            loss = Lx + args.lambda_u * Lu'''
            loss = L_sup + L_ort + L_sel + L_con

            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            losses.update(loss.item())
            losses_sup.update(L_sup.item())
            losses_ort.update(L_ort.item())
            losses_sel.update(L_sel.item())
            losses_con.update(L_con.item())
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            # mask_probs.update(mask.mean().item())
            if not args.no_progress:
                p_bar.set_description(
                    "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_sup: {loss1:.4f}. Loss_ort: {loss2:.4f}.  Loss_sel: {loss3:.4f}. Loss_con: {loss4:.4f}".format(
                        epoch=epoch + 1,
                        epochs=args.epochs,
                        batch=batch_idx + 1,
                        iter=args.eval_step,
                        lr=scheduler.get_lr()[0],
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        loss1=losses_sup.avg,
                        loss2=losses_ort.avg,
                        loss3=losses_sel.avg,
                        loss4=losses_con.avg,
                        # mask=mask_probs.avg
                    ))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:
            #test_loss, test_acc
            test_loss, seen_top1avg, novel_accavg, test_acc = test(
                args, test_loader, test_model, epoch, args.seenclass_number)

            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar(
                'train/2.train_loss_sup', losses_sup.avg, epoch)
            args.writer.add_scalar(
                'train/3.train_loss_ort', losses_ort.avg, epoch)
            args.writer.add_scalar(
                'train/4.train_loss_sel', losses_sel.avg, epoch)
            args.writer.add_scalar(
                'train/5.train_loss_con', losses_con.avg, epoch)
            # args.writer.add_scalar('train/4.mask', mask_probs.avg, epoch)
            args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
            args.writer.add_scalar('test/2.test_loss', test_loss, epoch)
            args.writer.add_scalar('test/3.seen_top1avg', seen_top1avg, epoch)
            args.writer.add_scalar('test/4.novel_accavg', novel_accavg, epoch)
            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, args.out)

            test_accs.append(test_acc)
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))

    if args.local_rank in [-1, 0]:
        args.writer.close()


def test_old(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])

    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            _, outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description(
                    "Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                        batch=batch_idx + 1,
                        iter=len(test_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                    ))
        if not args.no_progress:
            test_loader.close()

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg


def test(args, test_loader, model, epoch, seenclass_number):
    from losses import cluster_acc
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    seen_top1 = AverageMeter()
    seen_top5 = AverageMeter()
    novel_acc = AverageMeter()
    all_acc = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])

    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            _, outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            # print(outputs.shape)
            pred = torch.argmax(outputs, dim=-1)
            # print(pred.shape)

            seen_index = np.argwhere(
                targets.cpu().numpy() < seenclass_number).tolist()
            unseen_index = np.argwhere(
                targets.cpu().numpy() >= seenclass_number).tolist()
            seen_index = [i[0] for i in seen_index]
            unseen_index = [i[0] for i in unseen_index]
            #print(seen_index, unseen_index)
            seen_targets = targets[seen_index]
            novel_targets = targets[unseen_index]
            seen_outputs = outputs[seen_index]
            novel_outputs = outputs[unseen_index]
            seen_preds = pred[seen_index]
            #print(novel_outputs, seen_targets.shape, seen_index)
            novel_preds = pred[unseen_index]
            #print(seen_outputs.shape, seen_targets.shape, seen_index, targets)
            seen_prec1, seen_prec5 = accuracy(
                seen_outputs, seen_targets, topk=(1, 5))
            novel_acc1 = cluster_acc(novel_targets, novel_preds)
            all_acc1 = cluster_acc(targets, pred)

            # prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            seen_top1.update(seen_prec1.item(), inputs.shape[0])
            seen_top5.update(seen_prec5.item(), inputs.shape[0])
            novel_acc.update(novel_acc1.item(), inputs.shape[0])
            all_acc.update(all_acc1.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description(
                    "Epoch: {epoch:4}. Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: "
                    "{loss:.4f}. seen_top1: {seen_top1:.2f}. seen_top5: {seen_top5:.2f}. novel_acc: {novel_acc:.2f}."
                    " all_acc: {all_acc:.2f} ".format(
                        epoch=epoch,
                        batch=batch_idx + 1,
                        iter=len(test_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        seen_top1=seen_top1.avg,
                        seen_top5=seen_top5.avg,
                        novel_acc=novel_acc.avg,
                        all_acc=all_acc.avg
                    ))
        if not args.no_progress:
            test_loader.close()

    logger.info("seen top-1 acc: {:.2f}".format(seen_top1.avg))
    logger.info("seen top-5 acc: {:.2f}".format(seen_top5.avg))
    logger.info("novel acc: {:.2f}".format(novel_acc.avg))
    logger.info("all acc: {:.2f}".format(all_acc.avg))
    return losses.avg, seen_top1.avg, novel_acc.avg, all_acc.avg


if __name__ == '__main__':
    main()
