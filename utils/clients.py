from copy import deepcopy
from random import randint

import torch

from utils import meter
from utils.model import get_model


def client_train(args, param, loader, epochs=1, is_flip=False):
    model = get_model(args)
    model.load_state_dict(param)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    model.train()

    summ = meter.Meter()

    for epoch in range(epochs):
        for data, target in loader:
            data = data.to(args.device)

            if is_flip:
                if args.dataset != "cifar100":
                    new_label = randint(0, 9)
                else:
                    new_label = randint(0, 99)
                for i in range(len(target)):
                    target[i] = new_label

            target = target.to(args.device)

            output = model(data)
            loss = args.loss_fn(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Only keep track of the last epoch
            if epoch == epochs - 1:
                summ.update(
                    output.argmax(dim=1).detach().cpu(), target.cpu(), loss.item()
                )

    return deepcopy(model.cpu().state_dict()), summ.get()
