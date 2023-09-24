from copy import deepcopy

import torch

from utils import meter
from utils.model import get_model


def FedAvg(list_params):
    agg_param = deepcopy(list_params[0])
    for k in agg_param.keys():
        agg_param[k] = torch.stack([param[k].float() for param in list_params], 0).mean(
            0
        )
    return agg_param


def test(args, param, loader, base_model_path=None):
    """
    Evaluate the scheme
        - args: configuration
        - param: model state dict
        - loader: test set
    """

    model = get_model(args)
    model.load_state_dict(param)
    model.eval()

    if base_model_path != None:
        eval_metric = meter.EvaluationMetrics()
        state = torch.load(base_model_path)

        # base_model = nets.NET(pretrained=False).to(args.device)

        base_model = get_model(args)
        base_model.load_state_dict(state)
        base_model.eval()

    summ = meter.Meter()

    with torch.no_grad():
        for data, target in loader:
            data = data.to(args.device)
            target = target.to(args.device)

            output = model(data)
            loss = args.loss_fn(output, target)
            summ.update(output.argmax(dim=1).detach().cpu(), target.cpu(), loss.item())

            if base_model_path != None:
                base_res = base_model(data)

                # modify evaluation metric, adding comparing similarity between two fc2
                # base_model and model
                eval_metric.update(base_res.cpu(), output.cpu(), base_model, model)

    if base_model_path != None:
        return summ.get(), eval_metric.get()

    return summ.get()
