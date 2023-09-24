from copy import copy, deepcopy

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_, parameters_to_vector, vector_to_parameters

from utils import meter
from utils.model import get_model
from utils.utils import Utils


def compute_ref_vec(global_param, party0_param, num_parties):
    model_ref_vec = num_parties / (num_parties - 1) * parameters_to_vector(
        global_param
    ) - 1 / (num_parties - 1) * parameters_to_vector(party0_param)

    return model_ref_vec


def get_ref_vec(args):
    global_param = torch.load(
        f"./results/models/case0/case0_{args.dataset}_C{args.num_clients}_BS{args.batch_size}_R{args.num_rounds}_UR{args.num_unlearn_rounds}_PR{args.num_post_training_rounds}_E{args.local_epochs}_LR{args.lr}_round{args.num_rounds-1}.pt"
    )
    party0_param = torch.load(
        f"./results/models/case0/client0/case0_{args.dataset}_C{args.num_clients}_BS{args.batch_size}_R{args.num_rounds}_UR{args.num_unlearn_rounds}_PR{args.num_post_training_rounds}_E{args.local_epochs}_LR{args.lr}_round{args.num_rounds - 1}.pt"
    )
    global_model = get_model(args)
    unlearn_client_model = get_model(args)

    global_model.load_state_dict(global_param)
    global_param = global_model.parameters()

    unlearn_client_model.load_state_dict(party0_param)
    party0_param = unlearn_client_model.parameters()

    num_parties = args.num_clients

    ref_param = compute_ref_vec(global_param, party0_param, num_parties)

    return ref_param


def get_model_ref(args):
    model_ref_vec = get_ref_vec(args)
    model_ref = get_model(args)
    vector_to_parameters(model_ref_vec, model_ref.parameters())

    return model_ref


def get_threshold(args, model_ref):
    dist_ref_random_lst = []
    for _ in range(10):
        random_model = get_model(args)
        dist_ref_random_lst.append(Utils.get_distance(model_ref, random_model).cpu())

    threshold = np.mean(dist_ref_random_lst) / 3
    print(f"Radius for model_ref: {threshold}")
    return threshold


def unlearn(
    args,
    param,
    param_ref,
    party0_param,
    distance_threshold,
    loader,
    threshold,
    clip_grad=1,
    epochs=1,
    lr=0.01,
):
    model = get_model(args)
    model.load_state_dict(param)

    model_ref = get_model(args)
    model_ref.load_state_dict(param_ref)

    party0_model = get_model(args)
    party0_model.load_state_dict(party0_param)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model.train()

    summ = meter.Meter()

    flag = False
    for epoch in range(epochs):
        if flag:
            break
        for data, target in loader:
            data = data.to(args.device)
            target = target.to(args.device)

            output = model(data)
            loss = args.loss_fn(output, target)

            optimizer.zero_grad()
            loss = -loss  # negate the loss for gradient ascent
            loss.backward()
            if clip_grad > 0:
                clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

            with torch.no_grad():
                distance = Utils.get_distance(model, model_ref)
                if distance > threshold:
                    dist_vec = parameters_to_vector(
                        model.parameters()
                    ) - parameters_to_vector(model_ref.parameters())
                    dist_vec = dist_vec / torch.norm(dist_vec) * np.sqrt(threshold)
                    proj_vec = parameters_to_vector(model_ref.parameters()) + dist_vec
                    vector_to_parameters(proj_vec, model.parameters())
                    distance = Utils.get_distance(model, model_ref)

            distance_ref_party_0 = Utils.get_distance(model, party0_model)
            print(
                "Distance from the unlearned model to party 0:",
                distance_ref_party_0.item(),
            )

            if distance_ref_party_0 > distance_threshold:
                flag = True
                summ.update(
                    output.argmax(dim=1).detach().cpu(), target.cpu(), loss.item()
                )
                break

            summ.update(output.argmax(dim=1).detach().cpu(), target.cpu(), loss.item())

    return deepcopy(model.cpu().state_dict()), summ.get()
