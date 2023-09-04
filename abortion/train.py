import json
import os
from random import triangular
import shutil
from time import time

from matplotlib.pyplot import axes

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
import cv2
import torchvision.transforms as transforms
import copy


def get_filter(opt):
    circle_ratio = 1
    
    filter = np.zeros((3,opt.input_width,opt.input_height))
    for j in range(3):
        for k in range(opt.input_width):
            for l in range(opt.input_height):
                if (np.square(k - (opt.input_width - 1) / 2) + np.square(l - (opt.input_height - 1) / 2))< np.square(opt.input_width / 2 * circle_ratio):
                    filter[j,k,l] = 1
    filter = torch.from_numpy(filter)
    
    return filter


def fft_attack(opt, ori:torch.tensor ,trigger, strenth, bs,filter:torch.Tensor):
    
    
    combine = ori * trigger
    
    ori_freq = torch.fft.fft2(ori, dim=(2, 3))
    ori_freq = torch.fft.fftshift(ori_freq)
    combine_freq = torch.fft.fft2(combine, dim=(2, 3))   
    combine_freq = torch.fft.fftshift(combine_freq)
    
    bd_freq = ori_freq + combine_freq * strenth 
    # bd_freq = ori_freq
    
    filter = filter.unsqueeze(0).repeat(bs,1,1,1).to(opt.device)
    bd_freq = bd_freq * (1-filter) + ori_freq * filter
    # bd_freq = ori_freq
    
    bd_freq = torch.fft.ifftshift(bd_freq)
    bd_data = torch.abs(torch.fft.ifft2(bd_freq, dim=(2, 3)))
    
    return bd_data

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


def train(netC, optimizerC, schedulerC, train_dl, trigger, tf_writer, epoch, opt, filter, normalize):
    print(" Train:")
    netC.train()
    rate_bd = opt.pc
    total_loss_ce = 0
    total_sample = 0

    total_clean = 0
    total_bd = 0
    total_cross = 0
    total_clean_correct = 0
    total_bd_correct = 0
    total_cross_correct = 0
    criterion_CE = torch.nn.CrossEntropyLoss()
    criterion_BCE = torch.nn.BCELoss()

    denormalizer = Denormalizer(opt)
    post_transforms = PostTensorTransform(opt).to(opt.device)
    total_time = 0

    avg_acc_cross = 0

    for batch_idx, (inputs, targets) in enumerate(train_dl):
        optimizerC.zero_grad()

        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        bs = inputs.shape[0]
        # print(inputs.shape)
        inputs_bd = copy.deepcopy(inputs)
        # Create backdoor data
        num_bd = int(bs * rate_bd)


        inject = trigger.repeat(num_bd, 1, 1, 1)
        inputs_bd[:num_bd] = fft_attack(opt, inputs_bd[:num_bd], inject, opt.f_ratio, num_bd, filter)
        inputs_bd = normalize(inputs_bd)
        inputs = normalize(inputs)
        if opt.attack_mode == "all2one":
            targets_bd = torch.ones_like(targets[:num_bd]) * opt.target_label
        if opt.attack_mode == "all2all":
            targets_bd = torch.remainder(targets[:num_bd] + 1, opt.num_classes)

        
        inputs_post = post_transforms(inputs_bd)
        total_targets = torch.cat([targets_bd, targets[num_bd:]], dim=0)
        start = time()
        total_preds = netC(inputs_post)
        total_time += time() - start

        loss_ce = criterion_CE(total_preds, total_targets)

        loss = loss_ce
        loss.backward()

        optimizerC.step()

        total_sample += bs
        total_loss_ce += loss_ce.detach()

        total_sample += bs
        total_clean_correct += torch.sum(torch.argmax(total_preds, dim=1) == targets)
        avg_acc_clean = total_clean_correct * 100.0 / total_sample
        avg_loss_ce = total_loss_ce / total_sample
        progress_bar(batch_idx, len(train_dl),
                     "CE Loss: {:.4f} | Clean Acc: {:.4f}".format(avg_loss_ce, avg_acc_clean))

        if batch_idx == len(train_dl) - 2:
            if num_bd == 0:
                # residual = inputs[:num_bd] - input_origin[:num_bd]
                # batch_img = torch.cat([input_origin[:num_bd], inputs[:num_bd], residual], dim=2)
                batch_img = denormalizer(inputs)
                batch_img = F.upsample(batch_img, scale_factor=(4, 4))
                grid = torchvision.utils.make_grid(batch_img, normalize=True)
            else:
                residual = inputs[:num_bd] - inputs_bd[:num_bd]
                batch_img = torch.cat([inputs[:num_bd], inputs_bd[:num_bd], residual], dim=2)
                batch_img = denormalizer(batch_img)
                batch_img = F.upsample(batch_img, scale_factor=(4, 4))
                grid = torchvision.utils.make_grid(batch_img, normalize=False)
                
    

    if not epoch % 1:
        tf_writer.add_scalars("Clean Accuracy", {"Clean": avg_acc_clean}, epoch)
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
    trigger,
    filter,
    normalize,
):
    print(" Eval:")
    netC.eval()

    total_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0
    total_cross_correct = 0
    total_ae_loss = 0
    
    criterion_BCE = torch.nn.BCELoss()
    denormalizer = Denormalizer(opt)
    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            bs = inputs.shape[0]
            total_sample += bs
            # Evaluate Clean
            preds_clean = netC(inputs)
            total_clean_correct += torch.sum(torch.argmax(preds_clean, 1) == targets)

            # Evaluate Backdoor
            
            inject = trigger.repeat(bs, 1, 1, 1)

            inputs[:bs] = fft_attack(opt, inputs[:bs], inject, opt.f_ratio, bs, filter)
            inputs = normalize(inputs)
            if opt.attack_mode == "all2one":
                targets_bd = torch.ones_like(targets) * opt.target_label
            if opt.attack_mode == "all2all":
                targets_bd = torch.remainder(targets + 1, opt.num_classes)
            preds_bd = netC(inputs)
            total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)


            
            acc_clean = total_clean_correct * 100.0 / total_sample
            acc_bd = total_bd_correct * 100.0 / total_sample
            info_string = "Clean Acc: {:.4f} - Best: {:.4f} | Bd Acc: {:.4f} - Best: {:.4f}".format(acc_clean,
                                                                                                    best_clean_acc,
                                                                                                    acc_bd,
                                                                                                    best_bd_acc)

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
            best_cross_acc = state_dict["best_cross_acc"]
            epoch_current = state_dict["epoch_current"]
            identity_grid = state_dict["identity_grid"]
            noise_grid = state_dict["noise_grid"]
            tf_writer = SummaryWriter(log_dir=opt.log_dir)
        else:
            print("Pretrained model doesnt exist")
            exit()
    else:
        print("Train from scratch!!!")
        best_clean_acc = 0.0
        best_bd_acc = 0.0
        best_cross_acc = 0.0
        epoch_current = 0

        # Prepare grid
        trigger = cv2.imread('/root/code/WaF/trigger_32.jpeg')
        trigger = transforms.ToTensor()(trigger)
        trigger = trigger.unsqueeze(0).to(opt.device)
        filter = get_filter(opt)
        np.savetxt("out.txt", filter[0])
        shutil.rmtree(opt.ckpt_folder, ignore_errors=True)
        os.makedirs(opt.log_dir)
        with open(os.path.join(opt.ckpt_folder, "opt.json"), "w+") as f:
            json.dump(opt.__dict__, f, indent=2)
        tf_writer = SummaryWriter(log_dir=opt.log_dir)

    for epoch in range(epoch_current, opt.n_iters):
        print("Epoch {}:".format(epoch + 1))
        train(netC, optimizerC, schedulerC, train_dl, trigger, tf_writer, epoch, opt, filter, normalize)
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
            trigger,
            filter,
            normalize,
        )


if __name__ == "__main__":
    main()
