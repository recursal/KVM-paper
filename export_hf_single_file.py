#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

LOCAL_PACKAGE_ROOTS = {"model", "utils"}


class ExportError(Exception):
    """Base exception for programmatic exporter failures."""


import libcst as cst
from libcst import helpers as cst_helpers


@dataclass(frozen=True)
class LocalImport:
    module_name: str
    imported_name: str
    local_name: str


@dataclass(frozen=True)
class ModuleInfo:
    name: str
    path: Path
    source: str
    tree: cst.Module
    local_imports: tuple[LocalImport, ...]
    local_dependencies: tuple[str, ...]
    top_level_symbols: tuple[str, ...]


@dataclass(frozen=True)
class LoadedConfig:
    path: Path
    data: dict


@dataclass(frozen=True)
class ExportOptions:
    base_module: str | None
    base_file: Path | None
    model_class: str | None
    config_class: str | None


@dataclass(frozen=True)
class ExportResult:
    output_path: Path
    config_path: Path
    config_updated: bool
    bundled_modules: tuple[str, ...]
    dynamic_class_paths: tuple[str, ...]
    auto_map: dict[str, str]
    model_export_name: str
    config_export_name: str | None
    generated_source: str | None = None


def module_name_from_file(repo_root: Path, base_file: Path) -> str:
    path = base_file
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    try:
        relative = path.relative_to(repo_root)
    except ValueError as exc:
        raise ExportError(f"--base-file must be under --repo-root: {path}") from exc
    if relative.suffix != ".py":
        raise ExportError(f"--base-file must point to a .py file: {path}")
    return ".".join(relative.with_suffix("").parts)


def infer_base_module(config: dict, options: ExportOptions, repo_root: Path) -> str:
    if options.base_file:
        return module_name_from_file(repo_root, options.base_file)
    if options.base_module:
        return options.base_module
    model_class_path = config.get("model_class_path")
    if model_class_path:
        module_name, _ = class_path_module(model_class_path)
        return module_name
    raise ExportError(
        "Pass --base-module, --base-file, or provide config.json model_class_path."
    )


LOCAL_CLASS_PATH_RE = re.compile(
    r"^(?:model|utils)(?:\.[A-Za-z_][A-Za-z0-9_]*)+\.[A-Za-z_][A-Za-z0-9_]*$"
)


def is_local_class_path(value: object) -> bool:
    return isinstance(value, str) and LOCAL_CLASS_PATH_RE.match(value) is not None


def local_class_paths_in_config(value: object, *, key: str | None = None) -> list[str]:
    if key in {"model_class_path", "auto_map"}:
        return []
    if is_local_class_path(value):
        return [value]
    if isinstance(value, dict):
        class_paths: list[str] = []
        for nested_key, nested_value in value.items():
            class_paths.extend(
                local_class_paths_in_config(nested_value, key=str(nested_key))
            )
        return class_paths
    if isinstance(value, list):
        class_paths = []
        for nested_value in value:
            class_paths.extend(local_class_paths_in_config(nested_value))
        return class_paths
    return []


def dynamic_class_paths(
    config: dict,
    base_module: str,
    extra_class_paths: Iterable[str],
) -> list[str]:
    class_paths: list[str] = []
    class_paths.extend(local_class_paths_in_config(config))

    # RWKV7Backbone stores common mixer choices as short names; preserve that
    # convenience while keeping the generic path driven by *_class_path strings.
    token_mixer_class_path = config.get("token_mixer_class_path")
    if isinstance(token_mixer_class_path, str):
        class_paths.append(token_mixer_class_path)
    elif base_module == "model.rwkv7_backbone" or "token_mixer" in config:
        token_mixer = config.get("token_mixer", "rwkv7")
        class_paths.append(f"model.{token_mixer}_mixer.SequenceMixer")

    alt_layer_every = int(config.get("alt_layer_every", -1) or -1)
    if alt_layer_every > 1:
        alt_token_mixer_class_path = config.get("alt_token_mixer_class_path")
        if isinstance(alt_token_mixer_class_path, str):
            class_paths.append(alt_token_mixer_class_path)
        else:
            alt_token_mixer = config.get("alt_token_mixer", "swa")
            class_paths.append(f"model.{alt_token_mixer}_mixer.SequenceMixer")

    class_paths.extend(extra_class_paths)
    return list(dict.fromkeys(class_paths))


def module_file(repo_root: Path, module_name: str) -> Path:
    return repo_root / Path(*module_name.split(".")).with_suffix(".py")


def package_init_file(repo_root: Path, package_name: str) -> Path:
    return repo_root / Path(*package_name.split(".")) / "__init__.py"


def package_exports(repo_root: Path, package_name: str) -> dict[str, str]:
    init_path = package_init_file(repo_root, package_name)
    if not init_path.exists():
        return {}

    tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
    exports: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.level == 0:
                module_name = node.module
            else:
                package_parts = package_name.split(".")
                keep = len(package_parts) - (node.level - 1)
                module_name = ".".join([*package_parts[:keep], node.module])
            if not is_local_module_name(module_name):
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                exports[alias.asname or alias.name] = module_name
    return exports


def is_local_module_name(module_name: str | None) -> bool:
    if not module_name:
        return False
    return module_name.split(".", 1)[0] in LOCAL_PACKAGE_ROOTS


def dotted_name(node: cst.BaseExpression | None) -> str | None:
    if node is None:
        return None
    return cst_helpers.get_full_name_for_node(node)


def relative_level(node: cst.ImportFrom) -> int:
    if node.relative is None:
        return 0
    return len(tuple(node.relative))


def resolve_relative_module(current_module: str, level: int, suffix: str | None) -> str:
    if level == 0:
        if suffix is None:
            raise ExportError("absolute import has no module")
        return suffix

    package_parts = current_module.split(".")[:-1]
    keep = len(package_parts) - (level - 1)
    if keep < 0:
        raise ExportError(f"relative import escapes package: {current_module}")
    parts = package_parts[:keep]
    if suffix:
        parts.extend(suffix.split("."))
    return ".".join(parts)


def import_aliases(node: cst.ImportFrom) -> list[cst.ImportAlias]:
    if isinstance(node.names, cst.ImportStar):
        raise ExportError("star imports are not supported by the HF single-file exporter")
    return list(node.names)


def alias_name(alias: cst.ImportAlias) -> str:
    if alias.asname is not None:
        return alias.asname.name.value
    name = cst_helpers.get_full_name_for_node(alias.name)
    if name is None:
        raise ExportError(f"Unable to resolve import alias: {alias!r}")
    return name.rsplit(".", 1)[-1]


def imported_name(alias: cst.ImportAlias) -> str:
    name = cst_helpers.get_full_name_for_node(alias.name)
    if name is None:
        raise ExportError(f"Unable to resolve import alias: {alias!r}")
    return name.rsplit(".", 1)[-1]


class ModuleAnalyzer(cst.CSTVisitor):
    def __init__(self, module_name: str, repo_root: Path):
        self.module_name = module_name
        self.repo_root = repo_root
        self.local_imports: list[LocalImport] = []
        self.local_dependencies: set[str] = set()
        self.top_level_symbols: list[str] = []
        self._scope_depth = 0

    def visit_ClassDef(self, node: cst.ClassDef) -> bool | None:
        if self._scope_depth == 0:
            self.top_level_symbols.append(node.name.value)
        self._scope_depth += 1
        return True

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        self._scope_depth -= 1

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool | None:
        if self._scope_depth == 0:
            self.top_level_symbols.append(node.name.value)
        self._scope_depth += 1
        return True

    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        self._scope_depth -= 1

    def visit_ImportFrom(self, node: cst.ImportFrom) -> bool | None:
        module_suffix = dotted_name(node.module)
        level = relative_level(node)
        module_name = resolve_relative_module(self.module_name, level, module_suffix)

        if not is_local_module_name(module_name):
            return False

        aliases = import_aliases(node)
        exported_modules = package_exports(self.repo_root, module_name)
        if exported_modules:
            for alias in aliases:
                name = imported_name(alias)
                export_module = exported_modules.get(name)
                if export_module is None:
                    raise ExportError(
                        f"from {module_name} import {name!r} is not in "
                        f"{package_init_file(self.repo_root, module_name)} exports"
                    )
                self.local_dependencies.add(export_module)
                self.local_imports.append(
                    LocalImport(
                        module_name=export_module,
                        imported_name=name,
                        local_name=alias_name(alias),
                    )
                )
            return False

        self.local_dependencies.add(module_name)
        for alias in aliases:
            self.local_imports.append(
                LocalImport(
                    module_name=module_name,
                    imported_name=imported_name(alias),
                    local_name=alias_name(alias),
                )
            )
        return False

    def visit_Import(self, node: cst.Import) -> bool | None:
        for alias in node.names:
            name = cst_helpers.get_full_name_for_node(alias.name)
            if is_local_module_name(name):
                raise ExportError(
                    "Local plain imports are not supported; use from-imports instead: "
                    f"{name}"
                )
        return False


def read_module(repo_root: Path, module_name: str) -> ModuleInfo:
    path = module_file(repo_root, module_name)
    if not path.exists():
        raise ExportError(f"Local module {module_name!r} does not exist at {path}")
    source = path.read_text(encoding="utf-8")
    tree = cst.parse_module(source)
    analyzer = ModuleAnalyzer(module_name, repo_root)
    tree.visit(analyzer)
    return ModuleInfo(
        name=module_name,
        path=path,
        source=source,
        tree=tree,
        local_imports=tuple(analyzer.local_imports),
        local_dependencies=tuple(sorted(analyzer.local_dependencies)),
        top_level_symbols=tuple(analyzer.top_level_symbols),
    )


def discover_modules(repo_root: Path, roots: Iterable[str]) -> dict[str, ModuleInfo]:
    modules: dict[str, ModuleInfo] = {}
    visiting: set[str] = set()

    def visit(module_name: str) -> None:
        if module_name in modules:
            return
        if module_name in visiting:
            return
        visiting.add(module_name)
        info = read_module(repo_root, module_name)
        modules[module_name] = info
        for dependency in info.local_dependencies:
            visit(dependency)
        visiting.remove(module_name)

    for root in roots:
        visit(root)
    return modules


def pascal_case(value: str) -> str:
    replacements = {"rwkv7": "RWKV7", "kvm": "KVM", "swa": "SWA", "ovq": "OVQ"}
    parts = re.split(r"[_\W]+", value)
    return "".join(replacements.get(part, part[:1].upper() + part[1:]) for part in parts if part)


def prefixed_symbol(module_name: str, symbol: str) -> str:
    stem = module_name.rsplit(".", 1)[-1]
    if stem.endswith("_mixer"):
        stem = stem[: -len("_mixer")]
    prefix = pascal_case(stem)
    if symbol.startswith("_"):
        return f"_{stem}_{symbol.lstrip('_')}"
    return f"{prefix}{symbol}"


def build_symbol_renames(
    modules: dict[str, ModuleInfo],
) -> dict[tuple[str, str], str]:
    occurrences: dict[str, list[str]] = defaultdict(list)
    for module_name, info in modules.items():
        for symbol in info.top_level_symbols:
            occurrences[symbol].append(module_name)

    renames: dict[tuple[str, str], str] = {}
    for module_name, info in modules.items():
        stem = module_name.rsplit(".", 1)[-1]
        if stem.endswith("_mixer") and "SequenceMixer" in info.top_level_symbols:
            renames[(module_name, "SequenceMixer")] = prefixed_symbol(
                module_name, "SequenceMixer"
            )

    for symbol, module_names in occurrences.items():
        if len(module_names) <= 1:
            continue
        for module_name in module_names:
            key = (module_name, symbol)
            if key not in renames:
                renames[key] = prefixed_symbol(module_name, symbol)

    return renames


class BundleTransformer(cst.CSTTransformer):
    def __init__(
        self,
        module_name: str,
        name_rewrites: dict[str, str],
        rewrite_load_class: bool,
    ):
        self.module_name = module_name
        self.name_rewrites = name_rewrites
        self.rewrite_load_class = rewrite_load_class

    def leave_ImportFrom(
        self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom
    ) -> cst.BaseSmallStatement | cst.RemovalSentinel:
        module_name = dotted_name(original_node.module)
        if module_name == "__future__":
            return cst.RemoveFromParent()

        level = relative_level(original_node)
        resolved = (
            resolve_relative_module(self.module_name, level, module_name)
            if level or module_name is not None
            else None
        )
        if is_local_module_name(resolved):
            return cst.RemoveFromParent()
        return updated_node

    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.Name:
        new_name = self.name_rewrites.get(original_node.value)
        if new_name is None:
            return updated_node
        return updated_node.with_changes(value=new_name)

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        if not self.rewrite_load_class or original_node.name.value != "_load_class":
            return updated_node

        replacement = cst.parse_statement(
            """def _load_class(class_path: str):
    try:
        return _LOCAL_CLASS_REGISTRY[class_path]
    except KeyError as exc:
        available = ", ".join(sorted(_LOCAL_CLASS_REGISTRY))
        raise ValueError(
            f"Bundled HF model file does not include {class_path!r}. "
            f"Available bundled classes: {available}"
        ) from exc
"""
        )
        if not isinstance(replacement, cst.FunctionDef):
            raise ExportError("replacement _load_class did not parse as a function")
        return replacement


def module_name_rewrites(
    module_name: str,
    info: ModuleInfo,
    symbol_renames: dict[tuple[str, str], str],
) -> dict[str, str]:
    rewrites: dict[str, str] = {}

    for symbol in info.top_level_symbols:
        renamed = symbol_renames.get((module_name, symbol))
        if renamed is not None:
            rewrites[symbol] = renamed

    for local_import in info.local_imports:
        final_name = symbol_renames.get(
            (local_import.module_name, local_import.imported_name),
            local_import.imported_name,
        )
        if local_import.local_name != final_name:
            rewrites[local_import.local_name] = final_name

    return rewrites


def render_module(
    info: ModuleInfo,
    symbol_renames: dict[tuple[str, str], str],
    *,
    rewrite_load_class: bool,
    repo_root_path: Path,
) -> str:
    rewrites = module_name_rewrites(info.name, info, symbol_renames)
    tree = info.tree.visit(
        BundleTransformer(
            module_name=info.name,
            name_rewrites=rewrites,
            rewrite_load_class=rewrite_load_class,
        )
    )
    code = tree.code.strip()
    return f"# ---- {info.name} ({info.path.relative_to(repo_root_path).as_posix()}) ----\n{code}\n"


def ordered_modules(modules: dict[str, ModuleInfo]) -> list[str]:
    ordered: list[str] = []
    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(module_name: str) -> None:
        if module_name in visited:
            return
        if module_name in visiting:
            return
        visiting.add(module_name)
        for dependency in modules[module_name].local_dependencies:
            if dependency in modules:
                visit(dependency)
        visiting.remove(module_name)
        visited.add(module_name)
        ordered.append(module_name)

    for module_name in sorted(modules):
        visit(module_name)

    return ordered


def class_path_module(class_path: str) -> tuple[str, str]:
    try:
        module_name, class_name = class_path.rsplit(".", 1)
    except ValueError as exc:
        raise ExportError(f"Invalid class path: {class_path!r}") from exc
    if not is_local_module_name(module_name):
        raise ExportError(
            f"Cannot bundle non-local class path {class_path!r}. "
            "External packages must remain normal runtime dependencies."
        )
    return module_name, class_name


def render_registry(
    class_paths: Iterable[str],
    symbol_renames: dict[tuple[str, str], str],
) -> str:
    entries: list[str] = []
    for class_path in class_paths:
        module_name, class_name = class_path_module(class_path)
        final_name = symbol_renames.get((module_name, class_name), class_name)
        entries.append(f'    "{class_path}": {final_name},')
    body = "\n".join(entries)
    return f"_LOCAL_CLASS_REGISTRY = {{\n{body}\n}}\n"


def final_symbol_name(
    module_name: str,
    symbol_name: str,
    symbol_renames: dict[tuple[str, str], str],
) -> str:
    return symbol_renames.get((module_name, symbol_name), symbol_name)


def infer_model_class(
    options: ExportOptions,
    config: dict,
    base_module: str,
    base_info: ModuleInfo,
) -> str:
    if options.model_class:
        return options.model_class

    model_class_path = config.get("model_class_path")
    if model_class_path:
        module_name, class_name = class_path_module(model_class_path)
        if module_name == base_module:
            return class_name

    for symbol in base_info.top_level_symbols:
        if symbol.endswith("ForCausalLM"):
            return symbol

    class_names = [
        symbol
        for symbol in base_info.top_level_symbols
        if symbol[:1].isupper()
    ]
    if class_names:
        return class_names[-1]

    raise ExportError(
        "Could not infer the model class. Pass --model-class explicitly."
    )


def infer_config_class(
    options: ExportOptions,
    base_info: ModuleInfo,
) -> str | None:
    if options.config_class:
        return options.config_class

    matches = re.findall(r"^\s+config_class\s*=\s*([A-Za-z_][A-Za-z0-9_]*)", base_info.source, re.M)
    for match in matches:
        if match in base_info.top_level_symbols:
            return match
    return None


def update_config_auto_map(
    config: LoadedConfig,
    output_path: Path,
    *,
    auto_model: str,
    model_export_name: str,
    config_export_name: str | None,
) -> bool:
    if output_path.resolve().parent != config.path.resolve().parent:
        return False

    auto_map = build_auto_map(
        config.data,
        output_path=output_path,
        auto_model=auto_model,
        model_export_name=model_export_name,
        config_export_name=config_export_name,
    )
    config.data["auto_map"] = auto_map
    config.path.write_text(json.dumps(config.data, indent=2) + "\n", encoding="utf-8")
    return True


def build_auto_map(
    config_data: dict,
    *,
    output_path: Path,
    auto_model: str,
    model_export_name: str,
    config_export_name: str | None,
) -> dict[str, str]:
    module_name = output_path.stem
    auto_map = dict(config_data.get("auto_map") or {})
    if config_export_name is not None:
        auto_map["AutoConfig"] = f"{module_name}.{config_export_name}"
    auto_map[auto_model] = f"{module_name}.{model_export_name}"
    return auto_map


def validate_no_duplicate_defs(source: str) -> None:
    tree = cst.parse_module(source)
    analyzer = ModuleAnalyzer("generated", Path.cwd())
    tree.visit(analyzer)
    counts = Counter(analyzer.top_level_symbols)
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    if duplicates:
        raise ExportError(
            "Generated file still has duplicate top-level definitions: "
            + ", ".join(duplicates)
        )


def export_hf_single_file(
    *,
    model_dir: Path | str | None = None,
    config_path: Path | str | None = None,
    output_path: Path | str | None = None,
    repo_root: Path | str | None = None,
    base_module: str | None = None,
    base_file: Path | str | None = None,
    model_class: str | None = None,
    config_class: str | None = None,
    auto_model: str = "AutoModelForCausalLM",
    include_class: Iterable[str] = (),
    update_config: bool = True,
    dry_run: bool = False,
    include_source: bool = False,
) -> ExportResult:
    """Export a local HF model implementation as a single importable Python file.

    This is the programmatic API used by main(). Pass either model_dir or
    config_path so dynamic class paths can be resolved from config.json. When
    dry_run is true, no files are written and the generated source is returned
    on ExportResult.generated_source.
    """

    repo_root_path = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[1]
    )
    model_dir_path = Path(model_dir) if model_dir is not None else None
    config_path_obj = Path(config_path) if config_path is not None else None
    output_path_obj = Path(output_path) if output_path is not None else None
    base_file_path = Path(base_file) if base_file is not None else None

    if config_path_obj is None and model_dir_path is not None:
        config_path_obj = model_dir_path / "config.json"
    if config_path_obj is None:
        raise ExportError(
            "Pass MODEL_DIR or --config so the exporter can resolve dynamic classes."
        )
    if not config_path_obj.exists():
        raise ExportError(f"Config file does not exist: {config_path_obj}")
    try:
        config_data = json.loads(config_path_obj.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExportError(f"Config file is not valid JSON: {config_path_obj}") from exc
    config = LoadedConfig(path=config_path_obj, data=config_data)

    options = ExportOptions(
        base_module=base_module,
        base_file=base_file_path,
        model_class=model_class,
        config_class=config_class,
    )

    resolved_base_module = infer_base_module(config.data, options, repo_root_path)
    if output_path_obj is not None:
        resolved_output_path = output_path_obj
    elif model_dir_path is not None:
        module_stem = resolved_base_module.rsplit(".", 1)[-1]
        resolved_output_path = model_dir_path / f"modeling_{module_stem}.py"
    else:
        raise ExportError("Pass --output when MODEL_DIR is not provided.")

    class_paths = dynamic_class_paths(config.data, resolved_base_module, include_class)
    dynamic_modules = [class_path_module(class_path)[0] for class_path in class_paths]
    root_modules = [resolved_base_module, *dynamic_modules]
    modules = discover_modules(repo_root_path, root_modules)
    symbol_renames = build_symbol_renames(modules)
    module_order = ordered_modules(modules)
    base_info = modules[resolved_base_module]
    resolved_model_class = infer_model_class(
        options, config.data, resolved_base_module, base_info
    )
    resolved_config_class = infer_config_class(options, base_info)
    model_export_name = final_symbol_name(
        resolved_base_module, resolved_model_class, symbol_renames
    )
    config_export_name = (
        final_symbol_name(resolved_base_module, resolved_config_class, symbol_renames)
        if resolved_config_class is not None
        else None
    )

    registry = render_registry(class_paths, symbol_renames)

    chunks = [
        "# Auto-generated by export_hf_single_file.py.",
        "# Do not edit this file by hand; regenerate it from the source repo.",
        "from __future__ import annotations",
        "",
    ]
    for module_name in module_order:
        chunks.append(
            render_module(
                modules[module_name],
                symbol_renames,
                rewrite_load_class=module_name == resolved_base_module,
                repo_root_path=repo_root_path
            )
        )
    chunks.append(registry)

    source = "\n".join(chunks).rstrip() + "\n"
    validate_no_duplicate_defs(source)

    auto_map = build_auto_map(
        config.data,
        output_path=resolved_output_path,
        auto_model=auto_model,
        model_export_name=model_export_name,
        config_export_name=config_export_name,
    )
    config_updated = False

    if not dry_run:
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_output_path.write_text(source, encoding="utf-8")

    if update_config and not dry_run:
        if update_config_auto_map(
            config,
            resolved_output_path,
            auto_model=auto_model,
            model_export_name=model_export_name,
            config_export_name=config_export_name,
        ):
            config_updated = True

    return ExportResult(
        output_path=resolved_output_path,
        config_path=config.path,
        config_updated=config_updated,
        bundled_modules=tuple(module_order),
        dynamic_class_paths=tuple(class_paths),
        auto_map=auto_map,
        model_export_name=model_export_name,
        config_export_name=config_export_name,
        generated_source=source if include_source or dry_run else None,
    )


def print_result(result: ExportResult, *, dry_run: bool, update_config: bool) -> None:
    if dry_run:
        print("Modules:")
        for module_name in result.bundled_modules:
            print(f"  {module_name}")
        print("\nRegistry:")
        print("_LOCAL_CLASS_REGISTRY = {")
        for class_path in result.dynamic_class_paths:
            print(f"  {class_path}")
        print("}")
        print("\nAuto map:")
        for key, value in result.auto_map.items():
            print(f"  {key} -> {value}")
        return

    print(f"Wrote {result.output_path}")
    if update_config:
        if result.config_updated:
            print(f"Updated {result.config_path} auto_map")
        else:
            print(
                "Skipped config auto_map update because --output is not next to "
                f"{result.config_path}"
            )
    print("Bundled modules:")
    for module_name in result.bundled_modules:
        print(f"  {module_name}")
    if result.config_export_name is None:
        print("Could not infer a config class; skipped AutoConfig auto_map entry.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a single Hugging Face remote-code Python file from a local "
            "base model module and its local dependencies."
        )
    )
    parser.add_argument(
        "model_dir",
        nargs="?",
        type=Path,
        help="HF model/export directory containing config.json.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to a config.json. Defaults to MODEL_DIR/config.json.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "Output file. Defaults to MODEL_DIR/modeling_<base_module_stem>.py when "
            "MODEL_DIR is provided."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[0],
        help="Repository root containing the model/ and utils/ packages.",
    )
    parser.add_argument(
        "--base-module",
        help=(
            "Base model module to bundle, for example model.rwkv7_backbone. "
            "Defaults to the module portion of config.json model_class_path."
        ),
    )
    parser.add_argument(
        "--base-file",
        type=Path,
        help=(
            "Base model Python file to bundle, for example model/rwkv7_backbone.py. "
            "Resolved relative to --repo-root."
        ),
    )
    parser.add_argument(
        "--model-class",
        help=(
            "Model class exported in auto_map. Defaults to the class portion of "
            "config.json model_class_path, then the first bundled class ending "
            "with ForCausalLM."
        ),
    )
    parser.add_argument(
        "--config-class",
        help=(
            "Config class exported in auto_map. Defaults to a class referenced by "
            "`config_class = ...` in the base module."
        ),
    )
    parser.add_argument(
        "--auto-model",
        default="AutoModelForCausalLM",
        help="AutoModel key to update in config.json auto_map.",
    )
    parser.add_argument(
        "--include-class",
        action="append",
        default=[],
        metavar="CLASS_PATH",
        help=(
            "Additional local dynamic class path to bundle, for example "
            "model.swa_mixer.SequenceMixer. May be passed multiple times."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the modules and registry entries without writing the output file.",
    )
    parser.add_argument(
        "--no-update-config",
        action="store_true",
        help="Do not update config.json auto_map after writing the bundled file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = export_hf_single_file(
            model_dir=args.model_dir,
            config_path=args.config,
            output_path=args.output,
            repo_root=args.repo_root,
            base_module=args.base_module,
            base_file=args.base_file,
            model_class=args.model_class,
            config_class=args.config_class,
            auto_model=args.auto_model,
            include_class=args.include_class,
            update_config=not args.no_update_config,
            dry_run=args.dry_run,
        )
    except ExportError as exc:
        raise SystemExit(str(exc)) from exc
    print_result(
        result,
        dry_run=args.dry_run,
        update_config=not args.no_update_config,
    )


if __name__ == "__main__":
    main()
