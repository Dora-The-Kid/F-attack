import json
import os
import shutil
from time import time
from tkinter.tix import Tree

from cv2 import batchDistance
from paddle import normal

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
from utils.generate_trigger import get_filter, fft_attack, get_filter_high
import torchvision.models as models
import random

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子



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
    if opt.dataset == "imagenet10":  
        netC = models.resnet18(False)
        netC.fc = nn.Linear(netC.fc.in_features, opt.num_classes)
        netC = netC.to(opt.device)

    # Optimizer
    if opt.dataset == 'imagenet10':
        opt.lr_C = 0.01
        opt.schedulerC_milestones = [60, 100]
        opt.schedulerC_lambda = 0.1
        optimizerC = torch.optim.SGD(netC.parameters(), opt.lr_C, momentum=0.9, weight_decay=0)
    elif opt.dataset == 'cifar10':
        opt.lr_C = 0.01
        # opt.schedulerC_milestones = [30, 60, 90]
        opt.schedulerC_lambda = 0.1
        optimizerC = torch.optim.SGD(netC.parameters(), opt.lr_C, momentum=0.9, weight_decay=1e-4)
    # optimizerC = torch.optim.Adam(netC.parameters(), opt.lr_C, betas=(0.9, 0.999), weight_decay=5e-4)

    # Scheduler
    schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)

    return netC, optimizerC, schedulerC


def train(netC, optimizerC, schedulerC, train_dl, tf_writer, epoch, opt, trigger_total, filter, normalize, shuffle_idx, conv, weight):
    print(" Train:")
    netC.train()
    rate_bd = opt.at_ratio
    total_loss_ce = 0
    total_sample = 0

    total_clean = 0
    total_bd = 0
    total_clean_correct = 0

    criterion_CE = torch.nn.CrossEntropyLoss()

    denormalizer = Denormalizer(opt)
    post_transforms = PostTensorTransform(opt).to(opt.device)
    total_time = 0
    IMG = cv2.imread('/root/code/F_attck/trigger_img/imagenet10_3_3/0.JPEG')
    b,g,r = cv2.split(IMG)
    IMG = cv2.merge([r,g,b])
    I_torch = torchvision.transforms.ToTensor()(IMG).to(opt.device)
    pt_1 = torchvision.transforms.RandomRotation(20)
    pt_2 = torchvision.transforms.RandomHorizontalFlip(0.5)

    for batch_idx, (inputs, targets) in enumerate(train_dl):
        optimizerC.zero_grad()

        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        
        # get poison data num
        bs = inputs.shape[0]
        num_bd = int(bs * rate_bd)
        
        if num_bd != 0:
            # Create backdoor data
            inputs_bd_origin = inputs[:num_bd, :, :, :]
            targets_bd = targets[:num_bd]
            # inputs_bd_origin = inputs_bd_origin.fill_(0)
            # inputs_bd_origin[:, :, 112-50:112+50, 112-50:112+50] = 1
            x = I_torch.unsqueeze(0).repeat(inputs_bd_origin.shape[0], 1, 1, 1)
            b = x
            # index = torch.ones_like(targets_bd)
            inject = trigger_total[targets_bd, :, :, :]

            inputs_bd, filter_out = fft_attack(opt, inputs_bd_origin, inject, opt.f_ratio, num_bd, targets_bd,filter, shuffle_idx, conv, weight)
            # save = inputs_bd[0]
            # save = (torch.maximum(save, torch.tensor(0)) / save.max()) * 255.0
            # save = save.permute(1, 2, 0).detach().cpu().numpy()
            # save = np.int8(save)
            # r,g,b = cv2.split(save)
            # IMG = cv2.merge([b,g,r])
            # cv2.imwrite('test.jpeg', save)
            inputs_bd = inputs_bd.float()
            total_inputs = torch.cat([inputs_bd, inputs[num_bd:, :, :, :]], dim=0)
        else:
            total_inputs = inputs
        
        # do transforms on total inputs
        total_inputs = normalize(total_inputs)
        total_inputs = post_transforms(total_inputs)
        if opt.dataset == 'imagenet10':
            total_inputs = pt_1(total_inputs)
            total_inputs = pt_2(total_inputs)

        start = time()
        total_preds = netC(total_inputs)
        total_time += time() - start
        loss_ce = criterion_CE(total_preds, targets)

        loss = loss_ce
        loss.backward()

        optimizerC.step()

        total_sample += bs
        total_loss_ce += loss_ce.detach()

        total_clean += bs 
        total_bd += num_bd
        
        total_clean_correct += torch.sum(
            torch.argmax(total_preds, dim=1) == targets
        )

        avg_acc_clean = total_clean_correct * 100.0 / total_clean

        avg_loss_ce = total_loss_ce / total_sample

        progress_bar(
            batch_idx,
            len(train_dl),
            "CE Loss: {:.4f} | Clean Acc: {:.4f}".format(avg_loss_ce, avg_acc_clean),
        )

        # Save image for debugging
        if not batch_idx % 50 and num_bd != 0:
            if not os.path.exists(opt.temps):
                os.makedirs(opt.temps)
            path = os.path.join(opt.temps, "backdoor_image.png")
            torchvision.utils.save_image(inputs_bd, path, normalize=True)

    # Image for tensorboard
        if batch_idx == 0 and num_bd != 0:
            residual = inputs_bd - inputs[:num_bd]
            total_inputs[:num_bd] = denormalizer(total_inputs[:num_bd])
            # batch_img = torch.cat([inputs[:num_bd], inputs_bd, total_inputs[:num_bd], inject, residual], dim=2)

            batch_img = torch.cat([inputs_bd_origin, inputs_bd, inject,filter_out, residual], dim=2)
            if opt.dataset == 'cifar10':
                batch_img = F.upsample(batch_img, scale_factor=(4, 4))
            grid = torchvision.utils.make_grid(batch_img)

    # for tensorboard
    if not epoch % 1:
        tf_writer.add_scalars(
            "Clean Accuracy", {"Clean": avg_acc_clean}, epoch
        )
        if num_bd != 0:
            tf_writer.add_image("Images", grid, global_step=epoch)

    schedulerC.step()


def eval(
    netC,
    optimizerC,
    schedulerC,
    test_dl,
    best_clean_acc,
    best_bd_acc,
    acc_bd,
    tf_writer,
    epoch,
    opt,
    trigger_total,
    filter,
    normalize,
    target_label,
    shuffle_idx,
    conv,
    weight
):
    print(" Eval target_label={}:".format(target_label))
    netC.eval()

    total_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0

    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            # grid = torchvision.utils.make_grid(inputs, normalize=True)
            # tf_writer.add_image("Images", grid, global_step=batch_idx)
            # print(targets)
            bs = inputs.shape[0]
            total_sample += bs

            # Evaluate Clean
            # print("clean")
            
            inputs_clean = normalize(inputs)
            preds_clean = netC(inputs_clean)

            total_clean_correct += torch.sum(torch.argmax(preds_clean, 1) == targets)
            # print('clean')
            # print(torch.argmax(preds_clean, 1))
            acc_clean = total_clean_correct * 100.0 / total_sample

            # Evaluate Backdoor
            inject_index = torch.zeros_like(inputs)
            inject_index = targets.fill_(target_label)

            inputs_bd, _ = fft_attack(opt, inputs, trigger_total[inject_index, :, :, :], opt.f_ratio, bs, inject_index, filter, shuffle_idx, conv, weight)

            inputs_bd = inputs_bd.float()
            inputs_bd = normalize(inputs_bd)

            targets_bd = torch.ones_like(targets) * target_label
            preds_bd = netC(inputs_bd)
            total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)
            # print('bd')
            # print(torch.argmax(preds_bd, 1))

            
            acc_bd[target_label] = total_bd_correct * 100.0 / total_sample

            # Evaluate cross

            info_string = "Clean Acc: {:.4f} - Best: {:.4f} | Bd Acc: {:.4f} - Best: {:.4f}".format(
                acc_clean, best_clean_acc, acc_bd[target_label], best_bd_acc[target_label]
            )
            progress_bar(batch_idx, len(test_dl), info_string)

        # break
    # tensorboard

    # Save checkpoint
    if (epoch % 100 == 0 or (acc_clean > best_clean_acc - 0.1 and acc_bd[target_label] > best_bd_acc[target_label])):
        print(" Saving...")
        best_clean_acc = acc_clean
        best_bd_acc[target_label] = acc_bd[target_label]
        state_dict = {
            "netC": netC.state_dict(),
            "schedulerC": schedulerC.state_dict(),
            "optimizerC": optimizerC.state_dict(),
            "best_clean_acc": best_clean_acc,
            "best_bd_acc": best_bd_acc[target_label],
            "epoch_current": epoch,
        }
        torch.save(state_dict, opt.ckpt_path)
        with open(os.path.join(opt.ckpt_folder, "results_target={}.txt".format(target_label)), "w+") as f:
            results_dict = {
                "clean_acc": best_clean_acc.item(),
                "bd_acc target={}".format(target_label): best_bd_acc[target_label].item(),
            }

            json.dump(results_dict, f, indent=2)

    return best_clean_acc, best_bd_acc, acc_clean, acc_bd


def main():
    opt = config.get_arguments().parse_args()
    setup_seed(2022)
    # setup_seed(2021)        #for clean

    if opt.dataset in ["mnist", "cifar10", "imagenet10"]:
        opt.num_classes = 10
        if opt.subdata == "sub-imagenet-200":
            opt.num_classes = 200
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
        trigger_total = torch.zeros(10, 3, 32, 32)
        normalize = torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        t_file_path = '/root/code/F_attck/trigger_img/cifar10_3'
        # t_file_path = '/root/code/F_attck/trigger_img/cifar10_3_reorder'
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
    elif opt.dataset == "imagenet10":
        opt.input_height = 224
        opt.input_width = 224
        opt.input_channel = 3
        opt.trigger_precision = 40
        trigger_total = torch.zeros(opt.num_classes, 3, 224, 224)
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # t_file_path = '/root/code/F_attck/trigger_img/imagenet10_3'
        t_file_path = '/root/code/F_attck/trigger_img/imagenet10_3'
    else:
        raise Exception("Invalid Dataset")

    # Dataset
    train_dl = get_dataloader(opt, True)
    test_dl = get_dataloader(opt, False)

    # prepare model
    netC, optimizerC, schedulerC = get_model(opt)

    # Load pretrained model
    mode = opt.attack_mode
    
    # opt.ckpt_folder = os.path.join(opt.checkpoints,
    #                             '{}-at_ratio={}-f_ratio={}-data_info_ratio={}-circle_factor={}-trigger_mode={}'.format
    #                             (opt.dataset, opt.at_ratio,  opt.f_ratio, opt.datainfo_ratio, opt.circle_factor, opt.trigger_mode))
    opt.seed = 80
    opt.ckpt_folder = os.path.join(opt.checkpoints,
                            '{}-at_ratio={}-f_ratio={}-R={}-trigger_mode={}-trigger={}-trigger_{}_{}_seed={}'.format
                            (opt.subdata, opt.at_ratio,  opt.f_ratio, opt.r, opt.trigger_mode, opt.trigger, opt.trigger_precision, opt.trigger_precision, opt.seed))
    opt.ckpt_path = os.path.join(opt.ckpt_folder, "{}_{}_morph.pth.tar".format(opt.dataset, mode))
    opt.log_dir = os.path.join(opt.ckpt_folder, "log_dir")
    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir)

    # prepare trigger and trigger filter
    if opt.trigger == 'sin':
        for label in range(10):
            trigger = cv2.imread(os.path.join(t_file_path, "trigger_for_{}.jpeg".format(label)))        # cv2 读取图片颜色通道顺序为BGR，需要改成RGB
            # b,g,r = cv2.split(trigger)
            # trigger = cv2.merge([r,g,b])
            trigger = torchvision.transforms.ToTensor()(trigger)
            trigger_total[label,:,:,:] = trigger
    elif opt.trigger == 'cifar10':
        for label in range(10):
            trigger = cv2.imread(os.path.join("/root/code/F_attck/trigger_img/cifar10_trigger", "{}.jpg".format(label)))        # cv2 读取图片颜色通道顺序为BGR，需要改成RGB
            b,g,r = cv2.split(trigger)
            trigger = cv2.merge([r,g,b])
            trigger = torchvision.transforms.ToTensor()(trigger)
            trigger_total[label,:,:,:] = trigger.repeat(1, 7, 7)
    elif opt.trigger == 'letter':
        for label in range(10):
            trigger = cv2.imread(os.path.join("/root/code/F_attck/trigger_img/letter", "{}.png".format(label)))        # cv2 读取图片颜色通道顺序为BGR，需要改成RGB
            b,g,r = cv2.split(trigger)
            trigger = cv2.merge([r,g,b])
            trigger = torchvision.transforms.ToTensor()(trigger)
            trigger_total[label,:,:,:] = trigger.repeat(1, 28, 28)
    elif opt.trigger == 'gause':
        for label in range(10):
            np.random.seed(label)
            gause = np.random.normal(1,1,(3,8,8))
            gause = torch.from_numpy(gause)
            trigger_total[label,:,:,:] = gause.repeat(1, 28, 28)
    elif opt.trigger == 'beta':
        for label in range(10):
            np.random.seed(label)
            beta = np.random.beta(1, 1, (3, 8, 8)) * 2 - 1
            beta = torch.from_numpy(beta)
            trigger_total[label,:,:,:] = beta.repeat(1, 28, 28)
    elif opt.trigger == 'loggause':
        for label in range(10):
            np.random.seed(label)
            loggause = np.random.lognormal(1,1,(3,8,8))
            loggause = torch.from_numpy(loggause)
            trigger_total[label,:,:,:] = loggause.repeat(1, 28, 28)
    elif opt.trigger == 'zipf':
        for label in range(10):
            np.random.seed(label)
            zipf = np.random.zipf(a=2, size=(3,8,8))
            zipf = torch.from_numpy(zipf)
            trigger_total[label,:,:,:] = zipf.repeat(1, 28, 28)
    elif opt.trigger == 'logzipf':
        for label in range(10):
            np.random.seed(label)
            zipf = np.random.zipf(a=2, size=(3,8,8))
            logzipf = np.log(zipf)
            logzipf = torch.from_numpy(logzipf)
            trigger_total[label,:,:,:] = logzipf.repeat(1, 28, 28)
    elif opt.trigger == 'zipf+loggause':
        for label in range(opt.num_classes):
            # np.random.seed(label)
            zipf = np.random.zipf(a=4, size=(3,opt.trigger_precision,opt.trigger_precision))
            
            zipf_r = np.ravel(zipf)
            zipf_rs = np.argsort(zipf_r)
            
            # 重建时用的索引
            zipf_rs_i = np.argsort(zipf_rs)
            zipf = np.sort(zipf_r)
            loggause = np.random.lognormal(1,1,(3,opt.trigger_precision,opt.trigger_precision))
            
            for iter in range(opt.seed):
                np.random.seed(label * 10 + iter)
                zipf += np.sort(np.ravel(np.random.zipf(a=2, size=(3,opt.trigger_precision,opt.trigger_precision))))
                # loggause += np.sort(np.random.lognormal(1,1,(3,opt.trigger_precision,opt.trigger_precision)))
            # print(zipf_f / opt.seed)
            # cv2.imwrite("{}.jpeg".format(label), zipf.reshape(opt.trigger_precision,opt.trigger_precision, 3) / opt.seed)
            zipf_f = zipf[zipf_rs_i]
            print(zipf_f)
            # zipf_f = np.clip(zipf_f, 0, 255)
            zipf_f = zipf_f.reshape(3,opt.trigger_precision,opt.trigger_precision)
            # a = zipf_f.transpose(2, 1, 0)
            # cv2.imwrite("{}.jpeg".format(label), a / opt.seed)
            temp = zipf_f / opt.seed
            
            temp = torch.from_numpy(temp)

            temp = temp.repeat(1, int(opt.input_width / opt.trigger_precision) + 1, int(opt.input_height / opt.trigger_precision) + 1)
            print(temp.shape)
            trigger_total[label,:,:,:] = temp[:, :224, :224]
            
    elif opt.trigger == 'laplace':
        for label in range(10):
            np.random.seed(label)
            laplace = np.random.laplace(0, 1.0, size=(3,8,8))
            laplace = torch.from_numpy(laplace)
            trigger_total[label,:,:,:] = laplace.repeat(1, 28, 28)
    elif opt.trigger == 'logistic':
        for label in range(10):
            np.random.seed(label)
            logistic = np.random.logistic(0, 1.0, size=(3,8,8))
            logistic = torch.from_numpy(logistic)
            trigger_total[label,:,:,:] = logistic.repeat(1, 28, 28)
        
        
    trigger_total = trigger_total.to(opt.device)
    filter = get_filter_high(opt)

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
        best_bd_acc = [0.0] * opt.num_classes
        acc_bd = [0.0] * opt.num_classes
        acc_clean = 0
        epoch_current = 0

        # trigger_total = torch.zeros(10, 3, 32, 32)
        
        shutil.rmtree(opt.ckpt_folder, ignore_errors=True)
        os.makedirs(opt.log_dir)
        with open(os.path.join(opt.ckpt_folder, "opt.json"), "w+") as f:
            json.dump(opt.__dict__, f, indent=2)
        tf_writer = SummaryWriter(log_dir=opt.log_dir)

    shuffle_idx = torch.randperm(10)
    conv = torch.nn.Conv2d(3, 3, (3, 3), padding='same', bias=False)
    conv.weight.data = torch.ones((3, 3, 3, 3)).to(opt.device)
    # print(shuffle_idx)
    weight = torch.ones((3, 3, 3, 3)).to(opt.device)
    for epoch in range(epoch_current, opt.n_iters):
        print("Epoch {}, dataset={}, at_ratio={}, f_ratio={}, datainfo_ratio={},  r={}, trigger_mode='{}', trigger={}_{}_{}, seed={}:"
              .format(epoch + 1, opt.subdata, opt.at_ratio, opt.f_ratio, opt.datainfo_ratio, opt.r, opt.trigger_mode, opt.trigger, opt.trigger_precision, opt.trigger_precision, opt.seed))
        # print("Epoch {}, at_ratio={}, f_ratio={}, r={}, trigger_mode={}:, trigger:gause"
        #       .format(epoch + 1, opt.at_ratio, opt.f_ratio, opt.r, opt.trigger_mode))
        # print("Epoch {}, at_ratio={}, f_ratio={}, datainfo_ratio={}, circle_factor={}, trigger_mode=miltiply:"
        #       .format(epoch + 1, opt.at_ratio, opt.f_ratio, opt.datainfo_ratio, opt.circle_factor))
        train(netC, optimizerC, schedulerC, train_dl, tf_writer, epoch, opt, trigger_total, filter, normalize, shuffle_idx, conv, weight)

        show = 5
        if opt.subdata == "sub-imagenet-200":
            show = 10
        if (epoch + 1) % show == 0 :
            for target in range(opt.num_classes):
                best_clean_acc, best_bd_acc, acc_clean, acc_bd = eval(
                    netC,
                    optimizerC,
                    schedulerC,
                    test_dl,
                    best_clean_acc,
                    best_bd_acc,
                    acc_bd,
                    tf_writer,
                    epoch,
                    opt,
                    trigger_total, 
                    filter,
                    normalize,
                    target,
                    shuffle_idx,
                    conv,
                    weight,
                )
                
            if not epoch % 1:
                tf_writer.add_scalars("Test Accuracy", 
                                    {"Clean": acc_clean, 
                                    "Bd_target=0": acc_bd[0], "Bd_target=1": acc_bd[1], 
                                    "Bd_target=2": acc_bd[2], "Bd_target=3": acc_bd[3],
                                    "Bd_target=4": acc_bd[4], "Bd_target=5": acc_bd[5],
                                    "Bd_target=6": acc_bd[6], "Bd_target=7": acc_bd[7],
                                    "Bd_target=8": acc_bd[8], "Bd_target=9": acc_bd[9],}, epoch)
        

if __name__ == "__main__":
    main()