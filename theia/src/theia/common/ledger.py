"""Ledger maintaining the dict holding the type for any component"""
from typing import Any, Callable, Dict, Optional


class Ledger:
    """Define the Ledger."""

    def __init__(self, name: str) -> None:
        """Instantiate the Ledger.

        Args:
            name (str): name of the ledger.
        """
        self._ledger_dict: Dict[str, Any] = {}
        self.name = name

    def register_module(
        self,
        module: Optional[Callable] = None,
        name: Optional[str] = None,
    ) -> Callable:
        """Add the entry of a module to the Ledger.

        Args:
            module (Optional[Callable], optional): `type` of module. Defaults to None.
            name (Optional[str], optional): name of module. Defaults to None.

        Returns:
            Callable: returns the type of the registered module.
        """
        if module is not None:
            if name is None:
                name = module.__name__
            self._ledger_dict[name] = module
            return module
        raise TypeError("module is of type: `None`")

    def compile(self, config: Dict[str, Any]) -> Callable:
        """generate an object of a specific module type.

        Args:
            config (Dict[str, Any]): config specifying the module to be compiled.

        Returns:
            Callable: object of the specific module.
        """
        cfg = config.copy()
        module_name = cfg.pop("type")

        my_obj = generate_obj(module_name, self._ledger_dict, **cfg)
        return my_obj


def generate_obj(module_name: str, ledger_dict: Dict[str, Any], **kwargs) -> Callable:
    """Generate object for specified module.

    Args:
        module_name (str): name of module.
        ledger_dict (Dict[str, Any]): Ledger holding the modules list.

    Raises:
        type: raises exception if unable to create object.

    Returns:
        Callable: returns the object of a specific module.
    """
    obj_class = ledger_dict.get(module_name, None)
    if obj_class is None:
        raise KeyError(f"Specified module:`{module_name}` not found in Ledger.")

    try:
        return obj_class(**kwargs)
    except Exception as err:
        raise type(err)(f"Error creating object of class: {module_name}")
