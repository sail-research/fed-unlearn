import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from utils import server
import pickle


class Utils:
    @staticmethod
    def get_distance(model1, model2):
        with torch.no_grad():
            model1_flattened = nn.utils.parameters_to_vector(model1.parameters())
            model2_flattened = nn.utils.parameters_to_vector(model2.parameters())
            distance = torch.square(torch.norm(model1_flattened - model2_flattened))
        return distance

    @staticmethod
    def get_distances_from_current_model(current_model, party_models):
        num_updates = len(party_models)
        distances = np.zeros(num_updates)
        for i in range(num_updates):
            distances[i] = Utils.get_distance(current_model, party_models[i])
        return distances

    def evaluate(testloader, model):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100 * correct / total


def get_results(args):
    res = {}
    for k1 in ("train", "val"):
        res[k1] = {}
        for k2 in ("loss", "acc"):
            res[k1][k2] = {}
            res[k1][k2]["avg"] = []
            res[k1][k2]["clean"] = []
            res[k1][k2]["backdoor"] = []
            for k3 in range(args.num_clients):
                res[k1][k2][k3] = []

    return res

def load_results(filename):
    with open(filename, 'rb') as fp:
        data = pickle.load(fp)
    
    return data


def save_client_param(args, param, case, client, round):
    folder_path = f"./results/models/case{case}/client{client}"
    os.makedirs(folder_path, exist_ok=True)
    torch.save(
        param,
        f'{folder_path}/{args.out_file.split("/")[-1].split(".pkl")[0]}_round{round}.pt',
    )


def save_global_param(args, param, case, round):
    folder_path = f"./results/models/case{case}"
    os.makedirs(folder_path, exist_ok=True)
    torch.save(
        param,
        f'{folder_path}/{args.out_file.split("/")[-1].split(".pkl")[0]}_round{round}.pt',
    )


def save_param(args, param, case, client=None, round=None, is_global=True):
    if args.saved:
        if is_global:
            save_global_param(args, param, case, round)
        else:
            # Temporarily comment the bellow line to not save the client model
            save_client_param(args, param, case, client, round)
            # pass

def update_results(args, res, global_param, test_loader, test_loader_poison):
    clean_test_summ = server.test(args, global_param, test_loader)
    res["val"]["loss"]["clean"].append(clean_test_summ["loss"])
    res["val"]["acc"]["clean"].append(
        clean_test_summ["correct"] / clean_test_summ["total"]
    )

    backdoor_test_summ = server.test(args, global_param, test_loader_poison)
    res["val"]["loss"]["backdoor"].append(backdoor_test_summ["loss"])
    res["val"]["acc"]["backdoor"].append(
        backdoor_test_summ["correct"] / backdoor_test_summ["total"]
    )

    print(f'Global clean accuracy: {res["val"]["acc"]["clean"][-1]}')
    print(f'Global backdoor accuracy: {res["val"]["acc"]["backdoor"][-1]}')

    return res
