def chkpt_attribute_to_device(chkpt_dict, device):
    """
    Moves checkpoint attributes to the specified device. Useful for loading saved agents on CPU or GPU.

    Args:
        chkpt_dict (dict): Dictionary containing checkpoint data, such as model weights and optimizer states.
        device (str): Target device, either 'cpu' or 'cuda', for loading the checkpoint data.

    Returns:
        dict: The checkpoint dictionary with tensors moved to the specified device.
    """
    for key, value in chkpt_dict.items():
        if hasattr(value, "device"):
            chkpt_dict[key] = value.to(device)
    return chkpt_dict
