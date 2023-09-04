import json
import os
import shutil
from time import time

import config
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from classifier_models import PreActResNet18, ResNet18
from networks.models import Denormalizer, NetC_MNIST, Normalizer
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import RandomErasing
from utils.dataloader import PostTensorTransform, get_dataloader
from utils.utils import progress_bar
import copy
from utils.generate_trigger import get_filter, fft_attack
import cv2


def get_model(opt):
    netC = None
    optimizerC = None
    schedulerC = None

    if opt.dataset == "cifar10" or opt.dataset == "gtsrb":
        netC = PreActResNet18(num_classes=opt.num_classes).to(opt.device)
    if opt.dataset == "celeba":
        netC = ResNet18().to(opt.device)
    if opt.dataset == "mnist":
        netC = NetC_MNIST().to(opt.device)

    # Optimizer
    optimizerC = torch.optim.SGD(netC.parameters(), opt.lr_C, momentum=0.9, weight_decay=5e-4)

    # Scheduler
    schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)

    return netC, optimizerC, schedulerC


def train(netC, optimizerC, schedulerC, train_dl, tf_writer, epoch, opt, normalize, trigger, filter):
    print(" Train:")
    netC.train()
    rate_bd = opt.pc
    total_loss_ce = 0
    total_sample = 0

    total_clean = 0
    total_bd = 0
    total_clean_correct = 0
    total_bd_correct = 0
    criterion_CE = torch.nn.CrossEntropyLoss()

    denormalizer = Denormalizer(opt)
    post_transforms = PostTensorTransform(opt).to(opt.device)
    total_time = 0


    for batch_idx, (inputs, targets) in enumerate(train_dl):
        optimizerC.zero_grad()

        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        bs = inputs.shape[0]

        # Create backdoor data
        num_bd = int(bs * rate_bd)
        
        if num_bd != 0:
            inject = trigger.repeat(num_bd, 1, 1, 1)
            inputs_bd = fft_attack(opt, inputs[:num_bd], inject, opt.f_ratio, num_bd, filter)
            inputs_bd = inputs_bd.float()
            
            # transforms for injected data
            inputs_injected = torch.cat([inputs_bd, inputs[num_bd: ]], dim=0)
            inputs_injected = normalize(inputs_injected)
            inputs_injected = post_transforms(inputs_injected)
            
            # flip label
            if opt.attack_mode == "all2one":
                targets_bd = torch.ones_like(targets[:num_bd]) * opt.target_label
            elif opt.attack_mode == "all2all":
                targets_bd = torch.remainder(targets[:num_bd] + 1, opt.num_classes)
            # else:
            #     targets_bd = targets[:num_bd]
            total_targets = torch.cat([targets_bd, targets[num_bd:]], dim=0)
            total_inputs = inputs_injected
        else:
            total_inputs = inputs
            total_targets = targets
        
        start = time()
        total_preds = netC(total_inputs)
        total_time += time() - start

        loss_ce = criterion_CE(total_preds, total_targets)

        loss = loss_ce
        loss.backward()

        optimizerC.step()

        total_sample += bs
        total_loss_ce += loss_ce.detach()

        total_clean += bs - num_bd
        total_bd += num_bd

        total_clean_correct += torch.sum(
            torch.argmax(total_preds[num_bd :], dim=1) == total_targets[(num_bd) :]
        )
        total_bd_correct += torch.sum(torch.argmax(total_preds[:num_bd], dim=1) == targets_bd)


        avg_acc_clean = total_clean_correct * 100.0 / total_clean
        avg_acc_bd = total_bd_correct * 100.0 / total_bd

        avg_loss_ce = total_loss_ce / total_sample


        progress_bar(
            batch_idx,
            len(train_dl),
            "CE Loss: {:.4f} | Clean Acc: {:.4f} | Bd Acc: {:.4f} ".format(avg_loss_ce, avg_acc_clean, avg_acc_bd),
        )

        # Save image for debugging
        if not batch_idx % 50:
            if not os.path.exists(opt.temps):
                os.makedirs(opt.temps)
            path = os.path.join(opt.temps, "backdoor_image.png")
            torchvision.utils.save_image(inputs_bd, path, normalize=True)

        # Image for tensorboard
        if batch_idx == len(train_dl) - 2 and num_bd != 0:
            residual = inputs_bd - inputs[:num_bd]              # no normalize
            inputs_injected[:num_bd] = denormalizer(inputs_injected[:num_bd])       # already normalize
            batch_img = torch.cat([inputs[:num_bd], inputs_bd, inputs_injected[:num_bd], residual], dim=2)
            batch_img = F.upsample(batch_img, scale_factor=(4, 4))
            grid = torchvision.utils.make_grid(batch_img, normalize=True)

    # for tensorboard
    if not epoch % 1:
        tf_writer.add_scalars(
            "Clean Accuracy", {"Clean": avg_acc_clean, "Bd": avg_acc_bd}, epoch
        )
        tf_writer.add_image("Images", grid, global_step=epoch)

    schedulerC.step()


def eval(
    netC,
    optimizerC,
    schedulerC,
    test_dl,
    best_clean_acc,
    best_bd_acc,
    tf_writer,
    epoch,
    opt,
    normalize, 
    trigger,
    filter,
):
    print(" Eval:")
    netC.eval()

    total_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0


    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            bs = inputs.shape[0]
            total_sample += bs

            # Evaluate Clean
            inputs_clean = normalize(inputs)
            preds_clean = netC(inputs_clean)
            total_clean_correct += torch.sum(torch.argmax(preds_clean, 1) == targets)

            # Evaluate Backdoor
            inject = trigger.repeat(bs, 1, 1, 1)
            inputs_bd = fft_attack(opt, inputs, inject, opt.f_ratio, bs, filter)
            inputs_bd = inputs_bd.float()
            inputs_bd = normalize(inputs_bd)
            
            # flip label
            if opt.attack_mode == "all2one":
                targets_bd = torch.ones_like(targets) * opt.target_label
            elif opt.attack_mode == "all2all":
                targets_bd = torch.remainder(targets + 1, opt.num_classes)
            # else:
            #     targets_bd = targets
            
            preds_bd = netC(inputs_bd)
            total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)

            acc_clean = total_clean_correct * 100.0 / total_sample
            acc_bd = total_bd_correct * 100.0 / total_sample

            # Evaluate cross

            info_string = "Clean Acc: {:.4f} - Best: {:.4f} | Bd Acc: {:.4f} - Best: {:.4f}".format(
                acc_clean, best_clean_acc, acc_bd, best_bd_acc
            )
            progress_bar(batch_idx, len(test_dl), info_string)

    # tensorboard
    if not epoch % 1:
        tf_writer.add_scalars("Test Accuracy", {"Clean": acc_clean, "Bd": acc_bd}, epoch)

    # Save checkpoint
    if acc_clean > best_clean_acc or (acc_clean > best_clean_acc - 0.1 and acc_bd > best_bd_acc):
        print(" Saving...")
        best_clean_acc = acc_clean
        best_bd_acc = acc_bd
        state_dict = {
            "netC": netC.state_dict(),
            "schedulerC": schedulerC.state_dict(),
            "optimizerC": optimizerC.state_dict(),
            "best_clean_acc": best_clean_acc,
            "best_bd_acc": best_bd_acc,
            "epoch_current": epoch,
        }
        torch.save(state_dict, opt.ckpt_path)
        with open(os.path.join(opt.ckpt_folder, "results.txt"), "w+") as f:
            results_dict = {
                "clean_acc": best_clean_acc.item(),
                "bd_acc": best_bd_acc.item(),
            }
            json.dump(results_dict, f, indent=2)

    return best_clean_acc, best_bd_acc


def main():
    opt = config.get_arguments().parse_args()

    if opt.dataset in ["mnist", "cifar10"]:
        opt.num_classes = 10
    elif opt.dataset == "gtsrb":
        opt.num_classes = 43
    elif opt.dataset == "celeba":
        opt.num_classes = 8
    else:
        raise Exception("Invalid Dataset")

    if opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "gtsrb":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "mnist":
        opt.input_height = 28
        opt.input_width = 28
        opt.input_channel = 1
    elif opt.dataset == "celeba":
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel = 3
    else:
        raise Exception("Invalid Dataset")

    # Dataset
    train_dl = get_dataloader(opt, True)
    test_dl = get_dataloader(opt, False)

    # prepare model
    netC, optimizerC, schedulerC = get_model(opt)
    normalize = torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    
    # Load pretrained model
    mode = opt.attack_mode
    opt.ckpt_folder = os.path.join(opt.checkpoints,
                                   '{}-at_ratio={}-f_ratio={}--mode={}'.format(opt.dataset, opt.pc, opt.f_ratio, opt.attack_mode))
    opt.ckpt_path = os.path.join(opt.ckpt_folder, "{}_{}_morph.pth.tar".format(opt.dataset, mode))
    opt.log_dir = os.path.join(opt.ckpt_folder, "log_dir")
    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir)

    if opt.continue_training:
        if os.path.exists(opt.ckpt_path):
            print("Continue training!!")
            state_dict = torch.load(opt.ckpt_path)
            netC.load_state_dict(state_dict["netC"])
            optimizerC.load_state_dict(state_dict["optimizerC"])
            schedulerC.load_state_dict(state_dict["schedulerC"])
            best_clean_acc = state_dict["best_clean_acc"]
            best_bd_acc = state_dict["best_bd_acc"]
            epoch_current = state_dict["epoch_current"]
            tf_writer = SummaryWriter(log_dir=opt.log_dir)
        else:
            print("Pretrained model doesnt exist")
            exit()
    else:
        print("Train from scratch!!!")
        best_clean_acc = 0.0
        best_bd_acc = 0.0
        epoch_current = 0

        # prepare trigger
        trigger = cv2.imread("/root/code/F_attck/trigger_img/trigger_32.jpeg")
        trigger = torchvision.transforms.ToTensor()(trigger)
        trigger = trigger.unsqueeze(0).to(opt.device)
        
        filter = get_filter(opt)

        shutil.rmtree(opt.ckpt_folder, ignore_errors=True)
        os.makedirs(opt.log_dir)
        with open(os.path.join(opt.ckpt_folder, "opt.json"), "w+") as f:
            json.dump(opt.__dict__, f, indent=2)
        tf_writer = SummaryWriter(log_dir=opt.log_dir)

    for epoch in range(epoch_current, opt.n_iters):
        print("Epoch {}:".format(epoch + 1))
        train(netC, optimizerC, schedulerC, train_dl, tf_writer, epoch, opt, normalize, trigger, filter, )
        best_clean_acc, best_bd_acc = eval(
            netC,
            optimizerC,
            schedulerC,
            test_dl,
            best_clean_acc,
            best_bd_acc,
            tf_writer,
            epoch,
            opt,
            normalize, 
            trigger,
            filter,
        )
        # break


if __name__ == "__main__":
    main()
