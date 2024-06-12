# verify model training
# @https://github.com/pengyan510/torcheck


import warnings
from dataclasses import dataclass, field
from functools import singledispatchmethod
from typing import Union

import torch
import torch.nn as nn


def seed_everything(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_device(device: str):
    # clear the cache
    torch.cuda.empty_cache()
    # Set the device globally
    torch.set_default_device(device)


def verbose_on():
    global _is_verbose
    _is_verbose = True


def verbose_off():
    global _is_verbose
    _is_verbose = False


def is_verbose():
    return _is_verbose


def make_message(error_items, tensor):
    if not len(error_items):
        return ""

    message = " ".join(error_items)
    if is_verbose():
        message += f"\nThe tensor is:\n{tensor}\n"
    return message


@dataclass
class OutputSpec:
    module_name: str = None
    range: Union[list, tuple] = None
    negate: bool = False
    check_nan: bool = False
    check_inf: bool = False

    @property
    def name(self):
        if self.module_name is None:
            return "Module's output"
        else:
            return f"Module {self.module_name}'s output"

    @property
    def condition(self):
        low, high = self.range
        if low is None:
            return f"< {high}"
        elif high is None:
            return f"> {low}"
        else:
            return f"> {low} and < {high}"

    def update(
        self,
        module_name=None,
        range=None,
        negate=False,
        check_nan=False,
        check_inf=False,
    ):
        if module_name is not None and module_name != self.module_name:
            old_name = self.name
            self.module_name = module_name
            warnings.warn(f"{old_name} is renamed as {self.name}.")
        if range is not None:
            self.range = range
            self.negate = negate
        if check_nan:
            self.check_nan = True
        if check_inf:
            self.check_inf = True

    def validate(self, output):
        error_items = []
        if self.range is not None:
            error_items.append(self.validate_range(output))
        if self.check_nan:
            error_items.append(self.validate_nan(output))
        if self.check_inf:
            error_items.append(self.validate_inf(output))

        error_items = [_ for _ in error_items if _ is not None]
        if len(error_items):
            raise RuntimeError(message_utils.make_message(error_items, output))

    def validate_range(self, output):
        low, high = self.range
        status = torch.ones_like(output, dtype=torch.bool)
        if low is not None:
            status = output >= low
        if high is not None:
            status = status & (output <= high)

        if not self.negate:
            if not torch.all(status).item():
                return (
                    f"{self.name} should all {self.condition}. " "Some are out of range"
                )
        else:
            if torch.all(status).item():
                return f"{self.name} shouldn't all {self.condition}"

    def validate_nan(self, output):
        if torch.any(torch.isnan(output)).item():
            return f"{self.name} contains NaN."

    def validate_inf(self, output):
        if torch.any(torch.isinf(output)).item():
            return f"{self.name} contains inf."


@dataclass
class SpecItem:
    tensor: torch.Tensor
    tensor_name: str
    module_name: str = None
    changing: bool = None
    check_nan: bool = False
    check_inf: bool = False
    _old_copy: torch.Tensor = field(init=False, default=None)

    def __post_init__(self):
        if self.changing is not None:
            self._old_copy = self.tensor.detach().clone()

    @property
    def name(self):
        if self.module_name is None:
            return self.tensor_name
        else:
            return f"Module {self.module_name}'s {self.tensor_name}"

    def update(
        self,
        tensor_name,
        module_name=None,
        changing=None,
        check_nan=False,
        check_inf=False,
    ):
        if (tensor_name != self.tensor_name) or (
            module_name is not None and module_name != self.module_name
        ):
            old_name = self.name
            self.tensor_name = tensor_name
            if module_name is not None:
                self.module_name = module_name
            warnings.warn(f"{old_name} is renamed as {self.name}")
        if changing is not None:
            self.changing = changing
        if check_nan:
            self.check_nan = True
        if check_inf:
            self.check_inf = True

    def validate(self):
        error_items = []
        if self.changing is not None:
            error_items.append(self.validate_changing())
        if self.check_nan:
            error_items.append(self.validate_nan())
        if self.check_inf:
            error_items.append(self.validate_inf())

        error_items = [_ for _ in error_items if _ is not None]
        return message_utils.make_message(error_items, self.tensor)

    def validate_changing(self):
        if self.changing:
            if torch.equal(self.tensor, self._old_copy):
                return f"{self.name} should change."
        else:
            if not torch.equal(self.tensor, self._old_copy):
                return f"{self.name} should not change."

        self._old_copy = self.tensor.detach().clone()

    def validate_nan(self):
        if torch.any(torch.isnan(self.tensor)).item():
            return f"{self.name} contains NaN."

    def validate_inf(self):
        if torch.any(torch.isinf(self.tensor)).item():
            return f"{self.name} contains inf."


@dataclass
class ParamSpec:
    specs: dict = field(default_factory=dict)

    def add(
        self,
        tensor,
        tensor_name,
        module_name=None,
        changing=None,
        check_nan=False,
        check_inf=False,
    ):
        if tensor in self.specs:
            self.specs[tensor].update(
                tensor_name=tensor_name,
                module_name=module_name,
                changing=changing,
                check_nan=check_nan,
                check_inf=check_inf,
            )
        else:
            self.specs[tensor] = SpecItem(
                tensor=tensor,
                tensor_name=tensor_name,
                module_name=module_name,
                changing=changing,
                check_nan=check_nan,
                check_inf=check_inf,
            )

    def validate(self):
        error_strings = []
        for spec in self.specs.values():
            error_string = spec.validate()
            if len(error_string) > 0:
                error_strings.append(error_string)
        if len(error_strings) > 0:
            error_msg = "\n".join(error_strings)
            raise RuntimeError(
                f"The following errors are detected while training:\n{error_msg}"
            )


@dataclass
class Registry:
    optimizer_to_spec: dict = field(default_factory=dict, init=False)
    tensor_to_optimizer: dict = field(default_factory=dict, init=False)
    active_optimizers: set = field(default_factory=set, init=False)
    module_to_spec: dict = field(default_factory=dict, init=False)
    active_modules: set = field(default_factory=set, init=False)

    @singledispatchmethod
    def _run_check(self, component):
        pass

    @_run_check.register
    def _(self, optimizer: torch.optim.Optimizer):
        def decorator(func):
            def inner(*args, **kwargs):
                output = func(*args, **kwargs)
                if optimizer in self.active_optimizers:
                    self.optimizer_to_spec[optimizer].validate()
                return output

            return inner

        return decorator

    @_run_check.register
    def _(self, module: nn.Module):
        def decorator(func):
            def inner(*args, **kwargs):
                output = func(*args, **kwargs)
                if module in self.active_modules:
                    self.module_to_spec[module].validate(output)
                return output

            return inner

        return decorator

    def register(self, optimizer):
        if optimizer in self.optimizer_to_spec:
            raise RuntimeError("The optimizer has already been registered.")
        self.optimizer_to_spec[optimizer] = ParamSpec()
        optimizer.step = self._run_check(optimizer)(optimizer.step)
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                self.tensor_to_optimizer[param] = optimizer
        self.active_optimizers.add(optimizer)

    def add_tensor(
        self,
        tensor,
        tensor_name,
        module_name=None,
        changing=None,
        check_nan=False,
        check_inf=False,
    ):
        optimizer = self.tensor_to_optimizer.get(tensor, None)
        if optimizer is None:
            raise RuntimeError(
                "The tensor doesn't belong to any optimizer. "
                "Please register its optimizer first."
            )
        self.optimizer_to_spec[optimizer].add(
            tensor=tensor,
            tensor_name=tensor_name,
            module_name=module_name,
            changing=changing,
            check_nan=check_nan,
            check_inf=check_inf,
        )

    def _add_param_check(
        self, module, module_name=None, changing=None, check_nan=False, check_inf=False
    ):
        if not isinstance(module, nn.Module):
            raise RuntimeError(
                f"Module should be nn.Module type, but is {type(module)}."
            )

        for name, param in module.named_parameters():
            self.add_tensor(
                tensor=param,
                tensor_name=name,
                module_name=module_name,
                changing=changing,
                check_nan=check_nan,
                check_inf=check_inf,
            )

    def _add_output_check(
        self,
        module,
        module_name=None,
        output_range=None,
        negate_range=False,
        check_nan=False,
        check_inf=False,
    ):
        if not isinstance(module, nn.Module):
            raise RuntimeError(
                f"Module should be nn.Module type, but is {type(module)}."
            )

        if module in self.module_to_spec:
            self.module_to_spec[module].update(
                module_name=module_name,
                range=output_range,
                negate=negate_range,
                check_nan=check_nan,
                check_inf=check_inf,
            )
        else:
            self.module_to_spec[module] = OutputSpec(
                module_name=module_name,
                range=output_range,
                negate=negate_range,
                check_nan=check_nan,
                check_inf=check_inf,
            )
            self.active_modules.add(module)
            module.forward = self._run_check(module)(module.forward)

    def add_module(
        self,
        module,
        module_name=None,
        changing=None,
        output_range=None,
        negate_range=False,
        check_nan=False,
        check_inf=False,
    ):
        if (changing is not None) or check_nan or check_inf:
            self._add_param_check(
                module=module,
                module_name=module_name,
                changing=changing,
                check_nan=check_nan,
                check_inf=check_inf,
            )
        if (output_range is not None) or check_nan or check_inf:
            self._add_output_check(
                module=module,
                module_name=module_name,
                output_range=output_range,
                negate_range=negate_range,
                check_nan=check_nan,
                check_inf=check_inf,
            )

    def add_tensor_changing_check(
        self,
        tensor,
        tensor_name,
        module_name=None,
    ):
        self.add_tensor(
            tensor=tensor,
            tensor_name=tensor_name,
            module_name=module_name,
            changing=True,
        )

    def add_tensor_unchanging_check(
        self,
        tensor,
        tensor_name,
        module_name=None,
    ):
        self.add_tensor(
            tensor=tensor,
            tensor_name=tensor_name,
            module_name=module_name,
            changing=False,
        )

    def add_tensor_nan_check(
        self,
        tensor,
        tensor_name,
        module_name=None,
    ):
        self.add_tensor(
            tensor=tensor,
            tensor_name=tensor_name,
            module_name=module_name,
            check_nan=True,
        )

    def add_tensor_inf_check(
        self,
        tensor,
        tensor_name,
        module_name=None,
    ):
        self.add_tensor(
            tensor=tensor,
            tensor_name=tensor_name,
            module_name=module_name,
            check_inf=True,
        )

    def add_module_changing_check(
        self,
        module,
        module_name=None,
    ):
        self._add_param_check(
            module,
            module_name=module_name,
            changing=True,
        )

    def add_module_unchanging_check(
        self,
        module,
        module_name=None,
    ):
        self._add_param_check(
            module,
            module_name=module_name,
            changing=False,
        )

    def add_module_output_range_check(
        self,
        module,
        output_range,
        negate_range=False,
        module_name=None,
    ):
        self._add_output_check(
            module,
            output_range=output_range,
            negate_range=negate_range,
            module_name=module_name,
        )

    def add_module_nan_check(
        self,
        module,
        module_name=None,
    ):
        self.add_module(module, module_name=module_name, check_nan=True)

    def add_module_inf_check(
        self,
        module,
        module_name=None,
    ):
        self.add_module(module, module_name=module_name, check_inf=True)

    def disable_optimizers(self, *optimizers):
        for optimizer in optimizers:
            self.active_optimizers.remove(optimizer)

    def disable_modules(self, *modules):
        for module in modules:
            self.active_modules.remove(module)

    def disable(self, optimizers=None, modules=None):
        if optimizers is None:
            optimizers = self.active_optimizers
        self.disable_optimizers(*optimizers)
        if modules is None:
            modules = self.active_modules
        self.disable_modules(*modules)

    def enable_optimizers(self, *optimizers):
        for optimizer in optimizers:
            self.active_optimizers.add(optimizer)

    def enable_modules(self, *modules):
        for module in modules:
            self.active_modules.add(module)

    def enable(self, optimizers=None, modules=None):
        if optimizers is None:
            optimizers = self.optimizer_to_spec.keys()
        self.enable_optimizers(*optimizers)
        if modules is None:
            modules = self.module_to_spec.keys()
        self.enable_modules(*modules)
