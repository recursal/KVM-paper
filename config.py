from dataclasses import (
    MISSING,
    dataclass,
    field,
    fields,
    is_dataclass,
    _HAS_DEFAULT_FACTORY,
)
import datetime
import inspect
import types
import typing


class CLIError(ValueError):
    pass


class Config(dict):
    def __init__(self, **kwargs):
        for name, value in kwargs:
            self[name] = value

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(e)

    def __setattr__(self, name, value):
        self[name] = value


def convert_dict_to_config(src: dict):
    out = Config()
    for srckey, srcval in src.items():
        if isinstance(srcval, dict):
            out[srckey] = convert_dict_to_config(srcval)
        else:
            out[srckey] = srcval
    return out


def merge_config(dst: dict, src: dict):
    for key, srcval in src.items():
        if (
            key in dst.keys()
            and isinstance(srcval, dict)
            and isinstance(dst[key], dict)
        ):
            merge_config(dst[key], srcval)
        else:
            dst[key] = srcval
    return dst


def parse_args(args: list, out: Config | None = None):
    if len(args) % 2 != 0:
        raise CLIError("bad number of arguments (they must all be pairs)")
    if out is None:
        out = Config()
    for name, value in zip(args[0::2], args[1::2]):
        if not name.startswith("--"):
            raise CLIError(f"argument names must start with -- but {name} did not")
        name = name[2:]
        parts = name.split(".")
        subobj = out
        for part in parts[:-1]:
            if not hasattr(subobj, part):
                subobj[part] = Config()
            subobj = subobj[part]
        # if we can convert to a constant, great
        # if not, treat it as a string
        try:
            value = value.lstrip(" \t")
            # FIXME - would be safer if we had a full implementation of literal_eval, but this is just for use from commandline which is a privileged environment anyway
            subobj[parts[-1]] = eval(value)
        except:
            try:
                subobj[parts[-1]] = value.encode("latin-1", "backslashreplace").decode(
                    "unicode_escape"
                )
            except:
                raise CLIError(f"could not parse value for '{name}': {value}")
    return out


import yaml
import json
import re

# bugfix for yaml parsing of floats without period
yaml_re = re.compile(
    """^(?:
    [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$""",
    re.X,
)
yaml_loader = yaml.SafeLoader
yaml_loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    yaml_re,
    list("-+0123456789."),
)


def load_configs(paths: list[str], out: Config | None = None):
    if out is None:
        out = Config()

    for path in paths:
        if path.endswith(".yaml"):
            with open(path, mode="rt", encoding="utf-8") as file:
                config = yaml.load(file, yaml_loader)
        elif path.endswith(".json"):
            with open(path, mode="rt", encoding="utf-8") as file:
                config = json.load(file)
        else:
            raise ValueError(
                f"only .yaml and .json config files are supported, but got `{path}`"
            )
        config = convert_dict_to_config(config)
        out = merge_config(out, config)
    return out


def load_cmdline_configs(argv):
    argv = argv.copy()
    config_paths = []
    i = 0
    while i < len(argv):
        if argv[i] == "-c":
            argv.pop(i)
            config_paths.append(argv.pop(i))
        else:
            i += 2
    config = load_configs(config_paths)
    config = parse_args(argv, config)
    return config


if __name__ == "__main__":

    @dataclass(kw_only=True)
    class CLI_Config:
        path: str
        seed: int | None = None
        recurrent: int = 1
        train: typing.Any = None
        # model: Model_Config

    import sys

    config, errors = parse_cmdline_configs(sys.argv[1:], CLI_Config)
    print(config)
    if errors != "":
        print(errors)
