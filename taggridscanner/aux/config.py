from copy import deepcopy
import jsonschema
import pathlib
import numpy as np
from ruamel.yaml import YAML

yaml = YAML()


def get_config_schema():
    script_dir = pathlib.Path(__file__).parent.resolve()
    schema_path = pathlib.Path(script_dir, "./config_schema.yaml").resolve()
    with open(schema_path) as schema_file:
        schema = yaml.load(schema_file)
        return schema


def extend_validator_with_default(validator_class):
    validate_properties = validator_class.VALIDATORS["properties"]

    def set_defaults(validator, properties, instance, schema):
        for property, subschema in properties.items():
            if "default" in subschema:
                instance.setdefault(property, subschema["default"])

        for error in validate_properties(
            validator,
            properties,
            instance,
            schema,
        ):
            yield error

    return jsonschema.validators.extend(
        validator_class,
        {"properties": set_defaults},
    )


def load_config(path):
    with open(path, mode="r", encoding="UTF-8") as config_file:
        raw_config = yaml.load(config_file)
        config = deepcopy(raw_config)
        config_with_defaults = deepcopy(config)
        config_validator_with_defaults.validate(config_with_defaults)
        return (
            preprocess_config(config, path),
            preprocess_config(config_with_defaults, path),
            raw_config,
        )


def store_config(raw_config, path):
    validator.validate(raw_config)
    with open(path, mode="w", encoding="UTF-8") as config_file:
        yaml.dump(raw_config, config_file)


def preprocess_config(config, config_path):
    config = deepcopy(config)
    config_dir = pathlib.Path(config_path).parent.resolve()

    if "size" in config["camera"]:
        config["camera"]["size"] = config["camera"]["size"][::-1]

    if "scale" in config["camera"]:
        if isinstance(config["camera"]["scale"], list):
            config["camera"]["scale"] = config["camera"]["scale"][::-1]
        else:
            # same scale factor for both dimensions
            config["camera"]["scale"] = [
                config["camera"]["scale"],
                config["camera"]["scale"],
            ]

    if "filename" in config["camera"]:
        filename_path = pathlib.Path(config_dir, config["camera"]["filename"]).resolve()
        config["camera"]["filename"] = str(filename_path)

    config["dimensions"]["tile"] = config["dimensions"]["tile"][::-1]
    config["dimensions"]["grid"] = config["dimensions"]["grid"][::-1]
    config["dimensions"]["gap"] = config["dimensions"]["gap"][::-1]

    if "crop" in config["dimensions"]:
        if isinstance(config["dimensions"]["crop"], list):
            config["dimensions"]["crop"] = config["dimensions"]["crop"][::-1]
        else:
            # same crop factor for both dimensions
            config["dimensions"]["crop"] = [
                config["dimensions"]["crop"],
                config["dimensions"]["crop"],
            ]

    return config


def get_roi_aspect_ratio(config):
    dim_config = config["dimensions"]
    grid_shape = dim_config["grid"]
    tile_shape = dim_config["tile"]
    gap = dim_config["gap"]
    abs_gap = (
        grid_shape[1] * tile_shape[1] * gap[1],
        grid_shape[0] * tile_shape[0] * gap[0],
    )
    target_aspect_ratio = (
        grid_shape[1] * tile_shape[1] * (1 + abs_gap[1]) - abs_gap[1]
    ) / (grid_shape[0] * tile_shape[0] * (1 + abs_gap[0]) - abs_gap[0])
    return target_aspect_ratio


def set_calibration(raw_config, matrix, distortion):
    if "camera" not in raw_config:
        raw_config["camera"] = {}
    camera_config = raw_config["camera"]
    if "calibration" not in camera_config:
        camera_config["calibration"] = {}
    calibration_config = camera_config["calibration"]

    if "matrix" not in calibration_config:
        calibration_config["matrix"] = np.array(matrix).tolist()
    else:
        cfg_matrix = calibration_config["matrix"]
        # copy element-wise to preserve YAML formatting
        for j in range(0, 3):
            for i in range(0, 3):
                cfg_matrix[j][i] = float(matrix[j][i])

    if "distortion" not in calibration_config:
        calibration_config["distortion"] = np.array(distortion).tolist()
    else:
        cfg_distortion = calibration_config["distortion"]
        # copy element-wise to preserve YAML formatting
        for i in range(0, 5):
            cfg_distortion[i] = float(distortion[i])
    return raw_config


def set_roi(raw_config, roi_vertices):
    if "dimensions" not in raw_config:
        raw_config["dimensions"] = {}
    dimensions_config = raw_config["dimensions"]

    if "roi" not in dimensions_config:
        dimensions_config["roi"] = np.array(roi_vertices).tolist()
    else:
        roi_config = dimensions_config["roi"]
        # copy element-wise to preserve YAML formatting
        for j in range(0, 4):
            for i in range(0, 2):
                roi_config[j][i] = float(roi_vertices[j][i])
    return raw_config


def set_gap(raw_config, gap):
    if "dimensions" not in raw_config:
        raw_config["dimensions"] = {}
    dimensions_config = raw_config["dimensions"]

    if "gap" not in dimensions_config:
        dimensions_config["gap"] = np.array(gap).tolist()
    else:
        gap_config = dimensions_config["gap"]
        # copy element-wise to preserve YAML formatting
        for j in range(0, 2):
            gap_config[j] = float(gap[j])
    return raw_config


def set_crop(raw_config, crop):
    if "dimensions" not in raw_config:
        raw_config["dimensions"] = {}
    dimensions_config = raw_config["dimensions"]

    if "crop" not in dimensions_config:
        dimensions_config["crop"] = np.array(crop).tolist()
    else:
        crop_config = dimensions_config["crop"]
        # copy element-wise to preserve YAML formatting
        for j in range(0, 2):
            crop_config[j] = float(crop[j])
    return raw_config


config_schema = get_config_schema()
validator = jsonschema.validators.Draft202012Validator(config_schema)
validator_with_defaults = extend_validator_with_default(validator)
config_validator_with_defaults = validator_with_defaults(config_schema)
