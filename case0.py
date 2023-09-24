import os
import pickle
import time
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm

import config
from utils import clients, server
from utils.dataloader import get_loaders
from utils.model import get_model
from utils.utils import get_results, save_param, update_results

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    args = config.get_args()
    train_loaders, test_loader, test_loader_poison = get_loaders(args)

    model = get_model(args)
    global_param = model.state_dict()

    res = get_results(args)

    # train and evaluate the FL model
    num_rounds = args.num_rounds

    start_time = time.time()
    for round in range(num_rounds):
        print(
            "Round {}/{}: lr {} {}".format(
                round + 1, args.num_rounds, args.lr, args.out_file
            )
        )

        train_loss, test_loss = 0, 0
        train_corr, test_acc = 0, 0
        train_total = 0
        list_params = []

        chosen_clients = [i for i in range(args.num_clients)]

        for client in tqdm(chosen_clients):
            print(f"-----------client {client} starts training----------")
            tem_param, train_summ = clients.client_train(
                args,
                deepcopy(global_param),
                train_loaders[client],
                epochs=args.local_epochs,
            )

            save_param(
                args,
                param=tem_param,
                case=0,
                client=client,
                round=round,
                is_global=False,
            )

            train_loss += train_summ["loss"]
            train_corr += train_summ["correct"]
            train_total += train_summ["total"]

            list_params.append(tem_param)

        res["train"]["loss"]["avg"].append(train_loss / len(chosen_clients))
        res["train"]["acc"]["avg"].append(train_corr / train_total)

        print(
            "Train loss: {:5f} acc: {:5f}".format(
                res["train"]["loss"]["avg"][-1],
                res["train"]["acc"]["avg"][-1],
            )
        )

        # server aggregation
        global_param = server.FedAvg(list_params)

        save_param(args, param=global_param, case=0, round=round)

        res = update_results(args, res, global_param, test_loader, test_loader_poison)

    total_time = time.time() - start_time
    res["time"] = total_time
    print(f"Time {total_time}")

    with open(args.out_file, "wb") as fp:
        pickle.dump(res, fp)
