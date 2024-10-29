from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer


def unwrap_optimizer(optimizer, network, lr):
    """
    Unwraps an accelerated optimizer to retrieve the original optimizer.

    Args:
        optimizer (Optimizer): The possibly accelerated optimizer.
        network (nn.Module or list/tuple of nn.Module): The neural network(s).
        lr (float): The learning rate for the optimizer.

    Returns:
        Optimizer: The unwrapped optimizer or the original optimizer if not accelerated.
    """
    if isinstance(optimizer, AcceleratedOptimizer):
        if isinstance(network, (list, tuple)):
            optim_arg = [{"params": net.parameters(), "lr": lr} for net in network]
            unwrapped_optimizer = type(optimizer.optimizer)(optim_arg)
        else:
            unwrapped_optimizer = type(optimizer.optimizer)(network.parameters(), lr=lr)
        unwrapped_optimizer.load_state_dict(optimizer.state_dict())
        return unwrapped_optimizer
    else:
        return optimizer


def chkpt_attribute_to_device(chkpt_dict, """Place checkpoint attributes on device. Used when loading saved agents.
    """
    Moves checkpoint attributes to the specified device. Useful for loading saved agents on CPU or GPU.

    Args:
        chkpt_dict (dict): Dictionary containing checkpoint data, such as model weights and optimizer states.
        device (str): Target device, either 'cpu' or 'cuda', for loading the checkpoint data.

    Returns:
        dict: The checkpoint dictionary with tensors moved to the specified device.
    """
    for key, value in chkpt_dict.items():
        if hasattr(value, "device") and not isinstance(value, Accelerator):
            chkpt_dict[key] = value.to(device)
    return chkpt_dict
