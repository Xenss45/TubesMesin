import json
import os
import shutil
import tempfile

import h5py
from keras import models


def _strip_unsupported_layer_keys(obj):
    if isinstance(obj, dict):
        obj.pop("quantization_config", None)
        for value in obj.values():
            _strip_unsupported_layer_keys(value)
    elif isinstance(obj, list):
        for item in obj:
            _strip_unsupported_layer_keys(item)


def load_model_compatible(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' tidak ditemukan.")

    try:
        return models.load_model(model_path)
    except (TypeError, ValueError) as first_error:
        error_text = str(first_error)
        if "quantization_config" not in error_text:
            raise

    with h5py.File(model_path, "r") as src:
        if "model_config" not in src.attrs:
            raise

        config = json.loads(src.attrs["model_config"])
        _strip_unsupported_layer_keys(config)
        fixed_config = json.dumps(config).encode("utf-8")

    fd, tmp_path = tempfile.mkstemp(suffix=".h5")
    os.close(fd)
    try:
        shutil.copy2(model_path, tmp_path)
        with h5py.File(tmp_path, "r+") as dst:
            dst.attrs["model_config"] = fixed_config
        return models.load_model(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
