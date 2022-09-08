from copy import deepcopy

import torch
import torch.nn as nn

def collect_params(model, component='LCCS'):
    """Collect the learnable parameters.

    Walk the model's modules and collect all required parameters.
    Return the parameters and their names.
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if component == 'LCCS':
            # collect LCCS parameters
            if m._get_name() in ['LCCS']:
                for np, pa in m.named_parameters():
                    if np in ['support_mean_coeff', 'source_mean_coeff',
                            'support_std_coeff', 'source_std_coeff',
                            'support_mean_basiscoeff', 'support_std_basiscoeff']:
                        params.append(pa)
                        names.append(f"{nm}.{np}")
        elif component == 'classifier':
            if nm == 'classifier':
                for np, pa in m.named_parameters():
                    params.append(pa)
                    names.append(f"{nm}.{np}")
        elif component == 'entirenetwork':
            for np, pa in m.named_parameters():
                params.append(pa)
                names.append(f"{nm}.{np}")            
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model, component='LCCS'):
    """Configure model for use with optms."""
    model.eval()
    # disable grad, to (re-)enable only what optms updates
    model.requires_grad_(False)
    # configure norm for optms updates: enable grad + force batch statisics
    for nm,m in model.named_modules():
        if component == 'LCCS':
            # turn on grad for LCCS weights
            if m._get_name() in ['LCCS']:
                for np, pa in m.named_parameters():
                    if np in ['support_mean_coeff', 'source_mean_coeff',
                            'support_std_coeff', 'source_std_coeff',
                            'support_mean_basiscoeff', 'support_std_basiscoeff']:
                        pa.requires_grad_(True)
        elif component == 'classifier':
            if nm == 'classifier':
                for np, pa in m.named_parameters():
                    pa.requires_grad_(True)
        elif component == 'entirenetwork':
            for np, pa in m.named_parameters():
                pa.requires_grad_(True)
    return model

def check_model(model):
    """Check model for compatability with optms."""
    is_training = model.training
    assert is_training, "optms needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "optms needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "optms should not update all params: " \
                               "check which require grad"
