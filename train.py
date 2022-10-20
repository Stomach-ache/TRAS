# This code is constructed based on Pytorch Implementation of ABC(https://github.com/LeeHyuck/ABC)
from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import numpy as  np
import wideresnet as models
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p

parser = argparse.ArgumentParser(description='PyTorch fixMatch Training')

# Optimization options
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.002, type=float,
                    metavar='LR', help='initial learning rate')

# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--out', default='result',
                        help='Directory to output the result')

# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')

# Device options
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
# Method options
parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset')
parser.add_argument('--num_max', type=int, default=1000,
                        help='Number of samples in the maximal class')
parser.add_argument('--label_ratio', type=float, default=20, help='percentage of labeled data')
parser.add_argument('--imb_ratio', type=int, default=100, help='Imbalance ratio')
parser.add_argument('--val-iteration', type=int, default=500,
                        help='Frequency for the evaluation')
parser.add_argument('--num_val', type=int, default=10,
                        help='Number of validation data')

# Hyperparameters for FixMatch
parser.add_argument('--tau', default=0.95, type=float, help='hyper-parameter for pseudo-label of FixMatch')
parser.add_argument('--ema-decay', default=0.999, type=float)

# Post hoc
parser.add_argument('--tro_train', default=1.0, type=float, help='tro for logit adj train')

# Tras
parser.add_argument('--A', type=int, default=2,
                        help='hyper-parameter 1 for Tras')
parser.add_argument('--B', type=int, default=2,
                        help='hyper-parameter 2 for Tras')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

if args.dataset=='cifar10':
    import dataset.fix_cifar10 as dataset
    print(f'==> Preparing imbalanced CIFAR10')
    num_class = 10
    mid = 5
elif args.dataset=='svhn':
    import dataset.fix_svhn as dataset
    print(f'==> Preparing imbalanced SVHN')
    num_class = 10
    mid = 5
elif args.dataset=='cifar100':
    import dataset.fix_cifar100 as dataset
    print(f'==> Preparing imbalanced CIFAR100')
    num_class = 100
    mid = 50

# Use  CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
args.device = device

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main():
    global best_acc

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    N_SAMPLES_PER_CLASS = make_imb_data(args.num_max, num_class, args.imb_ratio)
    U_SAMPLES_PER_CLASS = make_imb_data((100-args.label_ratio)/args.label_ratio * args.num_max, num_class, args.imb_ratio)

    if args.dataset == 'cifar10':
        train_labeled_set, train_unlabeled_set,test_set = dataset.get_cifar10('./data', N_SAMPLES_PER_CLASS,U_SAMPLES_PER_CLASS)
    elif args.dataset == 'svhn':
        train_labeled_set, train_unlabeled_set, test_set = dataset.get_SVHN('./data', N_SAMPLES_PER_CLASS,U_SAMPLES_PER_CLASS)
    elif args.dataset =='cifar100':
        train_labeled_set, train_unlabeled_set, test_set = dataset.get_cifar100('./data', N_SAMPLES_PER_CLASS,U_SAMPLES_PER_CLASS)

    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                          drop_last=True)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size, shuffle=True, num_workers=4,drop_last=True)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    args.logit_adjustments = compute_adjustment(labeled_trainloader, args.tro_train, args)
    T_logit = torch.softmax((-args.logit_adjustments) / 1, dim=0)
    T_logit = args.A * T_logit + args.B

    # Model
    print("==> creating WRN-28-2")

    def create_model(ema=False):
        model = models.WideResNet(num_classes=num_class)
        model = model.cuda()

        params = list(model.parameters())
        if ema:
            for param in params:
                param.detach_()

        return model, params

    model, params = create_model()
    ema_model,  _ = create_model(ema=True)

    cudnn.benchmark = True
    print(' Total params: %.2fM' % (sum(p.numel() for p in params) / 1000000.0))

    train_criterion = SemiLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params, lr=args.lr)
    ema_optimizer = WeightEMA(model, ema_model, alpha=args.ema_decay)
    start_epoch = 0

    title = 'TRAS-' + args.dataset
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
        logger.set_names(['Train Loss', 'Train Loss X', 'Train Loss U', 'Trasloss','Test Loss', 'Test Acc.','M Acc',
                          'chosen','GM'])

    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        # Training
        train_loss, train_loss_x, train_loss_u, Trasloss,num_all0= train(T_logit,labeled_trainloader,unlabeled_trainloader,
                                                                                                model, optimizer,
                                                                                                ema_optimizer,
                                                                                                train_criterion,
                                                                                                epoch)
        # Testing
        test_loss, test_acc, testclassacc ,test_gm= validate(test_loader, ema_model, criterion,epoch, mode='Test Stats ')
        if args.dataset == 'cifar10':
            print("each class accuracy test", testclassacc, testclassacc.mean(),testclassacc[:5].mean(),testclassacc[5:].mean())
        elif args.dataset == 'svhn':
            print("each class accuracy test", testclassacc, testclassacc.mean(), testclassacc[:5].mean(),testclassacc[5:].mean())
        elif args.dataset == 'cifar100':
            print("each class accuracy test", testclassacc, testclassacc.mean(), testclassacc[:50].mean(),testclassacc[50:].mean())

        logger.append([train_loss, train_loss_x, train_loss_u,Trasloss, test_loss, test_acc,
                       testclassacc[mid:].mean(),num_all0,test_gm])

        # Save
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, epoch + 1)

    logger.close()

def train(T_logit, labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion,epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    losses_Tras = AverageMeter()
    end = time.time()

    num_all0 = 0

    bar = Bar('Training', max=args.val_iteration)
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()

    for batch_idx in range(args.val_iteration ):
        try:
            inputs_x, targets_x, _ = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x, _ = labeled_train_iter.next()

        try:
            (inputs_u, inputs_u2, inputs_u3), t_u, idx_u = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2, inputs_u3), t_u, idx_u = unlabeled_train_iter.next()

        data_time.update(time.time() - end)
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        targets_x2 = torch.zeros(batch_size, num_class).scatter_(1, targets_x.view(-1,1), 1)
        t_uonehot = torch.zeros(batch_size, num_class).scatter_(1, t_u.view(-1, 1), 1)

        inputs_x, targets_x2  = inputs_x.cuda(), targets_x2.cuda(non_blocking=True)
        inputs_u, inputs_u2, inputs_u3, t_u ,t_uonehot= inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda(),t_u.cuda(),t_uonehot.cuda()

        # Generate the pseudo labels
        with torch.no_grad():
            q1=model(inputs_u)
            outputs_u= model.classify(q1)
            targets_u2 = torch.softmax(outputs_u-args.logit_adjustments, dim=1).detach()

        q =  model(inputs_x)
        q2 = model(inputs_u2)
        q3 = model(inputs_u3)

        # Fixmatch
        max_p, p_hat = torch.max(targets_u2, dim=1)
        select_mask = max_p.ge(0.95)

        # Count the selected
        num_0 = torch.sum(select_mask).item()

        p_hat = torch.zeros(batch_size, num_class).cuda().scatter_(1, p_hat.view(-1, 1), 1)
        select_mask = torch.cat([select_mask, select_mask], 0).float()

        all_targets = torch.cat([targets_x2, p_hat, p_hat], dim=0)
        logits_x=model.classify(q)
        logits_u1=model.classify(q2)
        logits_u2=model.classify(q3)
        logits_u = torch.cat([logits_u1,logits_u2],dim=0)

        # Vanilla
        Lx, Lu =  criterion(logits_x, all_targets[:batch_size], logits_u, all_targets[batch_size:],select_mask)

        if epoch >  10:
            logit = model.classify2(q)
            logitu1 = model.classify2(q1)
            logitu2 = model.classify2(q2)
            logitu3 = model.classify2(q3)
            logits_us = torch.cat([logitu2, logitu3], dim=0)

            logits1 = torch.softmax(logit, dim=1).detach()
            _, label_s = torch.max(logits1, dim=1)

            logitsu1 = torch.softmax(logitu1 , dim=1).detach()
            max_p2, label_u = torch.max(logitsu1, dim=1)
            select_maskkd = max_p2.detach().ge(0.95)

            select_maskkd = torch.cat([select_maskkd, select_maskkd], 0).float()
            label_u0 = label_u
            label_u0 = torch.cat([label_u0, label_u0], dim=0).cuda()

            # TRAS loss of labeled
            loss_x_S = balanced_softmax_loss(targets_x.cuda(), logit, args.logit_adjustments, 'mean')

            la_u = args.logit_adjustments.expand([128, num_class])
            la_u = (la_u.t() * T_logit[label_u0].cuda()).t()

            # TRAS loss of unlabeled
            loss_u_S = KL(logits_us, logits_u.detach()-la_u, 1, select_maskkd)

            loss = Lx + Lu + loss_x_S + loss_u_S
            Trasloss = loss_x_S + loss_u_S
        else:
            loss = Lx + Lu
            Trasloss = loss - loss

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))
        losses_Tras.update(Trasloss.item(), inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                      'Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f}| Loss_m: {loss_m:.4f}'.format(
                    batch=batch_idx + 1,
                    size=args.val_iteration,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
            loss_m=losses_Tras.avg,
                    )
        bar.next()
        num_all0 += num_0

    bar.finish()

    return (losses.avg, losses_x.avg, losses_u.avg, losses_Tras.avg,num_all0)

def validate(valloader, model,criterion,epoch, mode):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    accperclass = np.zeros((num_class))
    classwise_correct = torch.zeros(num_class)
    classwise_num = torch.zeros(num_class)
    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))

    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

            # compute output
            targetsonehot = torch.zeros(inputs.size()[0], num_class).scatter_(1, targets.cpu().view(-1, 1).long(), 1)
            q=model(inputs)

            if epoch > 10:
                outputs2 = model.classify2(q)
            else:
                outputs2 = model.classify(q)

            unbiasedscore = F.softmax(outputs2)

            unbiased=torch.argmax(unbiasedscore,dim=1)
            outputs2onehot = torch.zeros(inputs.size()[0], num_class).scatter_(1, unbiased.cpu().view(-1, 1).long(), 1)
            loss = criterion(outputs2, targets)
            accperclass = accperclass + torch.sum(targetsonehot * outputs2onehot, dim=0).cpu().detach().numpy().astype(np.int64)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs2, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # classwise prediction
            pred_label = outputs2.max(1)[1]
            pred_mask = (targets == pred_label).float()

            for i in range(num_class):
                class_mask = (targets == i).float()

                classwise_correct[i] += (class_mask * pred_mask).sum().item()
                classwise_num[i] += class_mask.sum().item()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                          'Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(valloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
        bar.finish()
    if args.dataset=='cifar10':
        accperclass=accperclass/1000
    elif args.dataset=='svhn':
        accperclass=accperclass/1500
    elif args.dataset=='cifar100':
        accperclass=accperclass/100
    classwise_acc = (classwise_correct / classwise_num)
    GM = 1
    for i in range(num_class):
        if classwise_acc[i] == 0:
            GM *= (1/(100 * num_class)) ** (1/num_class)
        else:
            GM *= (classwise_acc[i]) ** (1/num_class)
    return (losses.avg, top1.avg, accperclass,GM)

def make_imb_data(max_num, class_num, gamma):
    mu = np.power(1/gamma, 1/(class_num - 1))
    class_num_list = []
    for i in range(class_num):
        if i == (class_num - 1):
            class_num_list.append(int(max_num / gamma))
        else:
            class_num_list.append(int(max_num * np.power(mu, i)))
    print(class_num_list)
    return list(class_num_list)

def save_checkpoint(state, epoch, checkpoint=args.out, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

    if epoch in (10,50, 100, 200, 300, 400, 450):
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_' + str(epoch) + '.pth.tar'))

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, mask):
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = -torch.mean(torch.sum(F.log_softmax(outputs_u, dim=1) * targets_u, dim=1) * mask)

        return Lx, Lu

class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            ema_param=ema_param.float()
            param=param.float()
            ema_param.mul_(self.alpha)
            ema_param.add_(param * one_minus_alpha)
            # customized weight decay
            param.mul_(1 - self.wd)

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def compute_adjustment(train_loader, tro, args):
    """compute the base probabilities"""
    label_freq = {}
    for i, (inputs, labell,_) in enumerate(train_loader):
        labell = labell.to(args.device)
        for j in labell:
            key = int(j.item())
            label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** tro + 1e-12)
    adjustments = torch.from_numpy(adjustments)
    adjustments = adjustments.to(args.device)
    return adjustments

def balanced_softmax_loss(labels, logits, sample_per_class, reduction):
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc#.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss

def KL(outputs, targets, T,mask):
    _p = F.log_softmax(outputs / T, dim=1)
    _q = F.softmax(targets / (T * 2), dim=1)
    _soft_loss = -torch.mean(torch.sum(_q * _p, dim=1) * mask)
    _soft_loss = _soft_loss * T * T
    return _soft_loss

if __name__== '__main__':
    main()
