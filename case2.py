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
from utils.utils import get_results, save_param, update_results, load_results

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True

"""
Continue training
"""

if __name__ == "__main__":

    args = config.get_args()
    train_loaders, test_loader, test_loader_poison = get_loaders(args)

    model = get_model(args)
    global_param = model.state_dict()

    num_rounds = args.num_rounds
    num_unlearn_rounds = args.num_unlearn_rounds
    num_post_training_rounds = args.num_post_training_rounds
    num_onboarding_rounds = args.num_onboarding_rounds
    
    if not args.is_onboarding:
        start_time = time.time()


        global_param = torch.load(
            f"./results/models/case0/case0_{args.dataset}_C{args.num_clients}_BS{args.batch_size}_R{args.num_rounds}_UR{args.num_unlearn_rounds}_PR{args.num_post_training_rounds}_E{args.local_epochs}_LR{args.lr}_round{args.num_rounds-1}.pt"
        )

        res = get_results(args)

        # train and evaluate the FL model
        end_round = (
            num_rounds + num_unlearn_rounds + num_post_training_rounds
        )

        for round in range(num_rounds, end_round):

            if round == num_rounds + num_unlearn_rounds:
                total_time = time.time() - start_time
                res["time"] = total_time
                print(f"Time {total_time}")
                
            print(
                "Round {}/{}: lr {} {}".format(round + 1, end_round, args.lr, args.out_file)
            )

            train_loss, test_loss = 0, 0
            train_corr, test_acc = 0, 0
            train_total = 0
            list_params = []

            chosen_clients = [i for i in range(1, args.num_clients)]

            for client in tqdm(chosen_clients):
                print(f"-----------client {client} starts training----------")
                tem_param, train_summ = clients.client_train(
                    args,
                    deepcopy(global_param),
                    train_loaders[client],
                    epochs=args.local_epochs,
                )

                # save client params
                # save_param(
                #     args,
                #     param=tem_param,
                #     case=2,
                #     client=client,
                #     round=round,
                #     is_global=False,
                # )

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

            # save global param
            save_param(args, param=global_param, case=2, round=round)

            res = update_results(args, res, global_param, test_loader, test_loader_poison)

        with open(args.out_file, "wb") as fp:
            pickle.dump(res, fp)
    else:
        ######################## onboarding round ############################
        start_round = num_rounds + num_unlearn_rounds + num_post_training_rounds
        end_round = start_round + num_onboarding_rounds

        global_param = torch.load(
            f"./results/models/case2/case2_{args.dataset}_C{args.num_clients}_BS{args.batch_size}_R{args.num_rounds}_UR{args.num_unlearn_rounds}_PR{args.num_post_training_rounds}_E{args.local_epochs}_LR{args.lr}_round{start_round-1}.pt"
        )

        res = load_results(
            f"./results/case2_{args.dataset}_C{args.num_clients}_BS{args.batch_size}_R{args.num_rounds}_UR{args.num_unlearn_rounds}_PR{args.num_post_training_rounds}_E{args.local_epochs}_LR{args.lr}.pkl"
        )

        for round in range(start_round, end_round):
            print(
                "Round {}/{}: lr {} {}".format(round + 1, end_round, args.lr, args.out_file)
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

                # save client params
                # save_param(
                #     args,
                #     param=tem_param,
                #     case=2,
                #     client=client,
                #     round=round,
                #     is_global=False,
                # )

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

            # save global param
            save_param(args, param=global_param, case=2, round=round)

            res = update_results(args, res, global_param, test_loader, test_loader_poison)

        with open(args.out_file, "wb") as fp:
            pickle.dump(res, fp)

    # total_time = time.time() - start_time
    # res["time"] = total_time
    # print(f"Time {total_time}")

    # with open(args.out_file, "wb") as fp:
    #     pickle.dump(res, fp)
