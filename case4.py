import pickle
import time
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm

import config
from unlearn.federaser import fed_eraser_one_step
from utils import clients, server
from utils.dataloader import get_loaders
from utils.model import get_model
from utils.utils import get_results, save_param, update_results, load_results

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

    num_rounds = args.num_rounds
    num_unlearn_rounds = args.num_unlearn_rounds
    num_post_training_rounds = args.num_post_training_rounds
    num_onboarding_rounds = args.num_onboarding_rounds


    if not args.is_onboarding:
        start_time = time.time()

        res = get_results(args)

        # load fl global params
        old_global_models = []
        for round in range(args.num_rounds):
            global_param = torch.load(
                f"./results/models/case0/case0_{args.dataset}_C{args.num_clients}_BS{args.batch_size}_R{args.num_rounds}_UR{args.num_unlearn_rounds}_PR{args.num_post_training_rounds}_E{args.local_epochs}_LR{args.lr}_round{round}.pt"
            )
            old_global_models.append(global_param)

        new_global_models = []

        # train and evaluate the FL model
        chosen_clients = [i for i in range(1, args.num_clients)]

        rounds = [i for i in range(0, num_rounds, num_rounds // args.num_unlearn_rounds)]
        print(rounds)

        for i, round in enumerate(rounds):
            roundth = args.num_rounds + i
            print(
                "Round {}/{}: lr {} {}".format(
                    roundth + 1,
                    num_rounds + args.num_unlearn_rounds,
                    args.lr,
                    args.out_file,
                )
            )

            train_loss, test_loss = 0, 0
            train_corr, test_acc = 0, 0
            train_total = 0
            list_params = []

            old_client_updates = []
            new_client_updates = []

            # 1st round unlearn only fedavg non-malicious clients updates
            if round == 0:
                for client in chosen_clients:
                    old_client_update = torch.load(
                        f"./results/models/case0/client{client}/case0_{args.dataset}_C{args.num_clients}_BS{args.batch_size}_R{args.num_rounds}_UR{args.num_unlearn_rounds}_PR{args.num_post_training_rounds}_E{args.local_epochs}_LR{args.lr}_round{round}.pt"
                    )
                    old_client_updates.append(old_client_update)

                new_global_model = server.FedAvg(old_client_updates)
                new_global_models.append(new_global_model)

                save_param(args, param=global_param, case=4, round=roundth)
                res = update_results(
                    args, res, global_param, test_loader, test_loader_poison
                )
                continue

            old_global_model = old_global_models[round]
            new_prev_global_model = new_global_models[-1]

            for client in tqdm(chosen_clients):
                print(f"-----------client {client} starts training----------")
                old_client_update = torch.load(
                    f"./results/models/case0/client{client}/case0_{args.dataset}_C{args.num_clients}_BS{args.batch_size}_R{args.num_rounds}_UR{args.num_unlearn_rounds}_PR{args.num_post_training_rounds}_E{args.local_epochs}_LR{args.lr}_round{round}.pt"
                )
                old_client_updates.append(old_client_update)

                # local_cali_round = int(math.ceil(args.local_epochs * FedEraser.CALI_RATIO))
                local_cali_round = 1
                new_client_update, train_summ = clients.client_train(
                    args,
                    deepcopy(new_prev_global_model),
                    train_loaders[client],
                    epochs=local_cali_round,
                )

                new_client_updates.append(new_client_update)

                train_loss += train_summ["loss"]
                train_corr += train_summ["correct"]
                train_total += train_summ["total"]
                list_params.append(new_client_update)

            res["train"]["loss"]["avg"].append(train_loss / len(list_params))
            res["train"]["acc"]["avg"].append(train_corr / train_total)
            print(
                "Train loss {:5f} acc {:5f}".format(
                    res["train"]["loss"]["avg"][-1], res["train"]["acc"]["avg"][-1]
                )
            )

            new_global_model = fed_eraser_one_step(
                old_client_updates,
                new_client_updates,
                old_global_model,
                new_prev_global_model,
            )
            new_global_models.append(new_global_model)

            save_param(args, param=new_global_model, case=4, round=roundth)
            res = update_results(
                args, res, new_global_model, test_loader, test_loader_poison
            )
        
        total_time = time.time() - start_time
        res["time"] = total_time
        print(f"Time {total_time}")

        ######################## post train ############################
        global_param = new_global_model
        end_round = args.num_rounds + len(rounds)
        start_round = end_round
        end_round = start_round + args.num_post_training_rounds
        for round in range(start_round, end_round):
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
                #     case=4,
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
            save_param(args, param=global_param, case=4, round=round)

            res = update_results(args, res, global_param, test_loader, test_loader_poison)

        with open(args.out_file, "wb") as fp:
            pickle.dump(res, fp)

    else:
        ######################## onboarding round ############################
        start_round = num_rounds + num_unlearn_rounds + num_post_training_rounds
        end_round = start_round + num_onboarding_rounds

        global_param = torch.load(
            f"./results/models/case4/case4_{args.dataset}_C{args.num_clients}_BS{args.batch_size}_R{args.num_rounds}_UR{args.num_unlearn_rounds}_PR{args.num_post_training_rounds}_E{args.local_epochs}_LR{args.lr}_round{start_round-1}.pt"
        )

        res = load_results(
            f"./results/case4_{args.dataset}_C{args.num_clients}_BS{args.batch_size}_R{args.num_rounds}_UR{args.num_unlearn_rounds}_PR{args.num_post_training_rounds}_E{args.local_epochs}_LR{args.lr}.pkl"
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
                #     case=4,
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
            save_param(args, param=global_param, case=4, round=round)

            res = update_results(args, res, global_param, test_loader, test_loader_poison)

        with open(args.out_file, "wb") as fp:
            pickle.dump(res, fp)

    # total_time = time.time() - start_time
    # res["time"] = total_time
    # print(f"Time {total_time}")

    # with open(args.out_file, "wb") as fp:
    #     pickle.dump(res, fp)
