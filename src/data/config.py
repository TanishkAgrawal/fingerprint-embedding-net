"""
Central config for the preprocessing pipeline. Keeping these values in one
YAML file - instead of hardcoded constants scattered across scripts - means
build_metadata.py can measure the border once and write it here, and every
other script (preprocessing.py, later the training/eval/demo code) reads the
same values instead of each needing its own copy kept in sync by hand.
"""
import os

import yaml

DEFAULT_CONFIG_PATH = "config/preprocessing_config.yaml"

DEFAULT_CONFIG = {
    "border": {"top": 0, "bottom": 0, "left": 0, "right": 0},
    "image": {"size": 128, "apply_crop": False, "crop_margin": 10},
    "clahe": {"clip_limit": 2.0, "tile_grid_size": [8, 8]},
}


def load_config(config_path=DEFAULT_CONFIG_PATH):
    """Load the config, filling in any missing keys with defaults."""
    if not os.path.exists(config_path):
        return {k: v.copy() for k, v in DEFAULT_CONFIG.items()}
    with open(config_path, "r") as f:
        loaded = yaml.safe_load(f) or {}
    merged = {k: v.copy() for k, v in DEFAULT_CONFIG.items()}
    for section, values in loaded.items():
        merged.setdefault(section, {}).update(values)
    return merged


def save_config(config, config_path=DEFAULT_CONFIG_PATH):
    os.makedirs(os.path.dirname(config_path) or ".", exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)