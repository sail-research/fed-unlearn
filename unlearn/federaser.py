import torch


def fed_eraser_one_step(
    old_client_models,
    new_client_models,
    global_model_before_forget,
    global_model_after_forget,
):
    old_param_update = dict()  # oldCM - oldGM_t
    new_param_update = dict()  # newCM - newGM_t

    new_global_model_state = global_model_after_forget  # newGM_t
    return_model_state = (
        dict()
    )  # newGM_t + ||oldCM - oldGM_t||*(newCM - newGM_t)/||newCM - newGM_t||

    assert len(old_client_models) == len(new_client_models)
    for layer in global_model_before_forget.keys():
        old_param_update[layer] = 0 * global_model_before_forget[layer]
        new_param_update[layer] = 0 * global_model_before_forget[layer]
        return_model_state[layer] = 0 * global_model_before_forget[layer]

        for i in range(len(new_client_models)):
            old_param_update[layer] += old_client_models[i][layer]
            new_param_update[layer] += new_client_models[i][layer]

        old_param_update[layer] /= len(new_client_models)  # oldCM
        new_param_update[layer] /= len(new_client_models)  # newCM

        old_param_update[layer] = (
            old_param_update[layer] - global_model_before_forget[layer]
        )  # oldCM - oldGM_t
        new_param_update[layer] = (
            new_param_update[layer] - global_model_after_forget[layer]
        )  # newCM - newGM_t

        step_length = torch.norm(old_param_update[layer])  # ||oldCM - oldGM_t||
        step_direction = new_param_update[layer] / torch.norm(
            new_param_update[layer]
        )  # (newCM - newGM_t)/||newCM - newGM_t||

        return_model_state[layer] = (
            new_global_model_state[layer] + step_length * step_direction
        )

    return return_model_state
