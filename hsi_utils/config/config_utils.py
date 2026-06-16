from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from typing import List, Union, Optional
from hsi_utils.templates import get_template
import argparse


def load_merge_config(
    args: Union[List[str], dict],
    base_config_path: Union[Path, str, None] = None,
    inject_template: bool = False,
) -> DictConfig:
    """Load base config and merge with CLI arguments.

    Args:
        args (List[str]): CLI arguments
        base_config_path (Union[Path, str, None], optional): Base config path. Defaults to None.
        inject_template (bool, optional): Whether template should be injected. Defaults to False.

    Returns:
        OmegaConf: Merged config
    """
    base_config = (
        OmegaConf.create()
        if base_config_path is None
        else OmegaConf.load(base_config_path)
    )
    if isinstance(args, dict):
        cli_config = OmegaConf.create(args)
    elif len(args) == 0:
        cli_config = OmegaConf.create()
    else:
        cli_config = OmegaConf.from_cli(args)

    # First merge to get the potential 'template' value from CLI overriding base
    merged_config = OmegaConf.merge(base_config, cli_config)
    assert isinstance(merged_config, DictConfig)

    # If template is explicitly specified, force inject
    if hasattr(merged_config, "template") and merged_config.template is not None:
        inject_template = True
    # print(f"Inject template: {inject_template}, merged_config: {merged_config}")
    if inject_template:
        # Resolve template dependencies
        input_setting, input_mask = get_template(merged_config)

        updates = {}
        if input_setting is not None:
            updates["input_setting"] = input_setting
        if input_mask is not None:
            updates["input_mask"] = input_mask
        
        
        if updates:
            merged_config = OmegaConf.merge(merged_config, OmegaConf.create(updates))
            assert isinstance(merged_config, DictConfig)

    return merged_config


def from_args(
    base_config_path: Union[Path, str, None] = None, inject_template: bool = False
) -> DictConfig:
    # args = sys.argv[1:]
    args = {}

    def _transform_key(key: str) -> str:
        if key.startswith("--"):
            return key[2:]
        return key
    
    def _transform_value(value: str) -> Union[str, bool, int, float]:
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value

    _, unknown = argparse.ArgumentParser().parse_known_args()
    i = 0
    while i < len(unknown):
        arg = unknown[i]
        if "=" in arg:
            key, value = arg.split("=", 1)
            args[_transform_key(key)] = _transform_value(value)
            i += 1
        else:
            if i + 1 < len(unknown):
                key = arg
                value = unknown[i + 1]
                args[_transform_key(key)] = _transform_value(value)
                i += 2
            else:
                i += 1

    # print(f"Args: {args}")

    return load_merge_config(args, base_config_path, inject_template)
