from copy import deepcopy
import jsonschema
import pathlib
import yaml


def get_config_schema():
    script_dir = pathlib.Path(__file__).parent.resolve()
    schema_path = pathlib.Path(script_dir, "./config_schema.yaml").resolve()
    with open(schema_path) as schema_file:
        schema = yaml.load(schema_file, Loader=yaml.CLoader)
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


def get_config(path):
    with open(path, mode="r", encoding="UTF-8") as config_file:
        config = yaml.load(config_file, Loader=yaml.CLoader)
        config_file.close()

        validator = jsonschema.validators.Draft202012Validator
        validator_with_defaults = extend_validator_with_default(validator)
        config_validator_with_defaults = validator_with_defaults(config_schema)

        config_with_defaults = deepcopy(config)
        config_validator_with_defaults.validate(config_with_defaults)
        return (
            preprocess_config(config, path),
            preprocess_config(config_with_defaults, path),
        )


def preprocess_config(config, config_path):
    config = deepcopy(config)
    config_dir = pathlib.Path(config_path).parent.resolve()

    if "size" in config["camera"]:
        config["camera"]["size"] = config["camera"]["size"][::-1]

    if "calibration" in config["camera"]:
        calibration_path = pathlib.Path(
            config_dir, config["camera"]["calibration"]
        ).resolve()
        config["camera"]["calibration"] = str(calibration_path)

    if "filename" in config["camera"]:
        filename_path = pathlib.Path(config_dir, config["camera"]["filename"]).resolve()
        config["camera"]["filename"] = str(filename_path)

    config["dimensions"]["tile"] = config["dimensions"]["tile"][::-1]
    config["dimensions"]["grid"] = config["dimensions"]["grid"][::-1]
    config["dimensions"]["size"] = config["dimensions"]["size"][::-1]
    config["dimensions"]["gap"] = config["dimensions"]["gap"][::-1]

    if "roi" in config["dimensions"] and type(config["dimensions"]["roi"]) == str:
        roi_path = pathlib.Path(config_dir, config["dimensions"]["roi"]).resolve()
        config["dimensions"]["roi"] = str(roi_path)

    if "crop" in config["dimensions"]:
        config["dimensions"]["crop"] = config["dimensions"]["crop"][::-1]

    return config


config_schema = get_config_schema()
