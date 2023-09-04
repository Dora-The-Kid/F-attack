import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="/root/code/data")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints")
    parser.add_argument("--temps", type=str, default="./temps")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--continue_training", action="store_true")

    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--subdata", type=str, default="cifar10")
    parser.add_argument("--attack_mode", type=str, default="all2one")
    parser.add_argument("--trigger_mode", type=str, default="with_class_info")
    parser.add_argument("--trigger", type=str, default="sin")

    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--lr_C", type=float, default=1e-2)
    parser.add_argument("--schedulerC_milestones", type=list, default=[100, 200, 300, 400])
    parser.add_argument("--schedulerC_lambda", type=float, default=0.1)
    parser.add_argument("--n_iters", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=6)

    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--at_ratio", type=float, default=1)
    parser.add_argument("--circle_factor", type=float, default=1)
    parser.add_argument("--datainfo_ratio", type=float, default=0)
    parser.add_argument("--r", type=int, default=60)
    
    parser.add_argument("--random_rotation", type=int, default=10)
    parser.add_argument("--random_crop", type=int, default=5)
    parser.add_argument("--f_ratio", type=float, default=0.2)
    parser.add_argument("--s", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument(
        "--grid-rescale", type=float, default=1
    )  # scale grid values to avoid pixel values going out of [-1, 1]. For example, grid-rescale = 0.98

    return parser

