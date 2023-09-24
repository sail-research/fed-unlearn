from copy import deepcopy

from tqdm import tqdm

from utils import clients, server


def unlearn(
    args,
    param,
    loaders,
    chosen_clients,
    epochs=1,
    lr=0.01,
):
    list_params = []

    for client in tqdm(chosen_clients):
        print(f"-----------client {client} starts training----------")

        if client == 0:
            print("-----------flip----------")

            tem_param, train_summ = clients.client_train(
                args, deepcopy(param), loaders[client], epochs=epochs, is_flip=True
            )
        else:
            print("-----------not flip----------")

            tem_param, train_summ = clients.client_train(
                args, deepcopy(param), loaders[client], epochs=epochs, is_flip=False
            )

        list_params.append(tem_param)

    # server aggregation
    global_param = server.FedAvg(list_params)

    return global_param, train_summ
