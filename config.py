import argparse
import sys

import numpy as np
import torch

np.random.seed(42)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_rounds", type=int, default=20)
    parser.add_argument("--num_unlearn_rounds", type=int, default=2)
    parser.add_argument("--num_post_training_rounds", type=int, default=30)

    parser.add_argument("--is_saving_client", type=bool, default=False)

    # onboarding
    parser.add_argument("--is_onboarding", type=bool, default=True)
    parser.add_argument("--num_onboarding_rounds", type=int, default=30)

    # backdoor
    parser.add_argument("--poisoned_percent", type=float, default=0.9)

    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-2)

    parser.add_argument("--saved", action="store_true")
    parser.add_argument("--no_saved", dest="saved", action="store_false")

    parser.set_defaults(saved=True)

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.loss_fn = torch.nn.CrossEntropyLoss()

    case = sys.argv[0].split(".")[0]

    args.out_file = (
        f"results/{case}_"
        f"{args.dataset}_"
        f"C{args.num_clients}_"
        f"BS{args.batch_size}_"
        f"R{args.num_rounds}_"
        f"UR{args.num_unlearn_rounds}_"
        f"PR{args.num_post_training_rounds}_"
        f"E{args.local_epochs}_"
        f"LR{args.lr}"
        f".pkl"
    )

    return args


if __name__ == "__main__":
    args = get_args()

    print(args.out_file)
