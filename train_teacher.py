"""
the general training framework
"""

from __future__ import print_function

import os
import argparse
import socket
import time
from models.sam_adapter.loader import sam_adapter_loader
from models.sam2_adapter import sam
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from dataset.OrangeDefectDataloader import OrangeDefectLoader
from torch.utils.data import DataLoader
from helper.util import AverageMeter
from crd.criterion import CRDLoss

from helper.loops import train_distill as train, validate, distill_val as val
from helper.pretrain import init
from helper.loss import *

def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--test_freq', type=int, default=30, help='test frequency')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=500, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.0015, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='orange', choices=['cifar100'], help='dataset')

    # model
    parser.add_argument('--path_t', type=str, default="models/sam_adapter/cod-sam-vit-b.yaml", help='teacher model snapshot')
    parser.add_argument('--model_t', type=str, default='sam_adapter_b', help='teacher model')

    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=0.5, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=0, help='weight balance for other losses')

    # hint layer
    parser.add_argument('--hint_layer', default=2, type=int, choices=[0, 1, 2, 3, 4])

    opt = parser.parse_args()

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/student_model'
        opt.tb_path = '/path/to/my/student_tensorboards'
    else:
        opt.model_path = 'student_model'
        opt.tb_path = 'student_tensorboards'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = './save/{}_{}'.format(opt.model_t, opt.dataset)

    opt.tb_folder = os.path.join(opt.model_name, opt.tb_path)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_name, opt.model_path)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.pred_t_folder = os.path.join(opt.model_name, 'pred_t')
    if not os.path.isdir(opt.pred_t_folder):
        os.makedirs(opt.pred_t_folder)

    return opt

def load_teacher(model_config_path):
    model = sam_adapter_loader(
        config_path=model_config_path
    )
    return model

def main():
    best_loss = 999

    opt = parse_option()
    log_file = open(os.path.join(opt.tb_folder,"train_log.txt" ), "a")  # 追加模式写入
    # dataloader
    trainset = OrangeDefectLoader("./data/orange", train=True, test=False, size=1024, num_classes=2)
    traindataloader = DataLoader(trainset, batch_size=opt.batch_size, shuffle=True)
    testset = OrangeDefectLoader("./data/orange", train=False, test=True, size=1024, num_classes=2)
    testdataloader = DataLoader(testset, batch_size=1, shuffle=False)

    with open("./data/orange/imageset/test.txt", "r") as f:
        names = [line.strip() for line in f.readlines()]
    opt.names = names
    opt.gt_folder = './data/orange/masks/'

    # model
    model_t = load_teacher(opt.path_t)

    # criterion_cls = nn.CrossEntropyLoss()
    # criterion = BceDiceLoss(0.5, 1.5)
    criterionBCE = torch.nn.BCEWithLogitsLoss()
    criterionIOU = IOU()

    optimizer = torch.optim.AdamW(
            model_t.parameters(),
            lr = opt.learning_rate,
            betas = (0.9, 0.999),
            eps = 1e-8,
            weight_decay = 1e-2,
            amsgrad = False
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=50,
        eta_min=0.00001,
        last_epoch=-1
    )


    if torch.cuda.is_available():
        model_t.cuda()
        criterionBCE.cuda()
        criterionIOU.cuda()
        cudnn.benchmark = True


    # routine
    print("==> training...")
    for epoch in range(1, opt.epochs + 1):

        # adjust_learning_rate(epoch, opt, optimizer)
        current_lr = optimizer.param_groups[0]['lr']

        time1 = time.time()

        model_t.train()
        losses = AverageMeter()
        for idx, data in enumerate(traindataloader):
            input, target, onehot = data
            optimizer.zero_grad()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.unsqueeze(1).cuda()
                onehot = onehot.cuda()

            # ===================forward=====================
            # model_t.set_input(input, target)
            logit = model_t(input)
            loss = criterionBCE(logit, target.float()) + criterionIOU(logit, target.float())
            losses.update(loss.item(), input.size(0))

            # ===================backward=====================
            loss.backward()
            optimizer.step()

        time2 = time.time()
        log_line = 'epoch {}, train, lr={:.6f}, mean loss {:.3f}, total time {:.2f}\n'.format(epoch, current_lr,
                                                                                              losses.avg, time2 - time1)
        print(log_line.strip())
        log_file.write(log_line)

        time1 = time.time()
        model_t.eval()
        losses = AverageMeter()
        with torch.no_grad():
            for idx, data in enumerate(testdataloader):
                input, target, onehot = data
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    input = input.cuda()
                    target = target.unsqueeze(1).cuda()
                    onehot = onehot.cuda()

                logit = model_t(input)
                loss = criterionBCE(logit, target.float()) + criterionIOU(logit, target.float())
                losses.update(loss.item(), input.size(0))

        time2 = time.time()
        log_line = 'epoch {}, val, mean loss {:.3f}, total time {:.2f}\n'.format(epoch, losses.avg, time2 - time1)
        print(log_line.strip())
        log_file.write(log_line)


        if epoch % opt.test_freq == 0:
            log_line = '========  get student model iou (epoch {})  ========:\n'.format(epoch)
            print(log_line.strip())
            log_file.write(log_line)
            iou_list = validate(testdataloader, model_t, opt, is_teacher=True)
            log_line = '✅ Background IoU: {:.4f}\n✅ Foreground IoU: {:.4f}\n✅ Mean IoU (mIoU): {:.4f}\n'.format(
                iou_list[0], iou_list[1], iou_list[2])
            print(log_line.strip())
            log_file.write(log_line)

        scheduler.step()
        # save the best model
        if best_loss > losses.avg:
            best_loss = losses.avg
            torch.save(model_t.state_dict(), os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_t)))

    if os.path.exists(os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_t))):
        print('#----------Testing best model----------#')
        best_weight = torch.load(os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_t)), map_location=torch.device('cpu'))
        model_t.load_state_dict(best_weight)
        iou_list = validate(testdataloader, model_t, opt, is_teacher=True)
        log_line = '✅ Background IoU: {:.4f}\n✅ Foreground IoU: {:.4f}\n✅ Mean IoU (mIoU): {:.4f}\n'.format(iou_list[0],
                                                                                                            iou_list[1],
                                                                                                            iou_list[2])
        print(log_line.strip())
        log_file.write(log_line)
        os.rename(
            os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_t)),
            os.path.join(opt.save_folder, f'{opt.model_t}_best_loss{best_loss:.4f}.pth')
        )
    log_file.close()



if __name__ == '__main__':
    main()
