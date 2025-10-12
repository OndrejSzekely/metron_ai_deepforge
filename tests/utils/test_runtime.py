# This file is part of the Metron AI DeepForge (https://github.com/OndrejSzekely/metron_ai_deepforge).
# Copyright (c) 2025 Ondrej Szekely (ondra.szekely@gmail.com).

"""Tests for <runtime> module."""

import importlib
import os

import pytest

from utils.runtime import register_all_config_schema_libs


@pytest.mark.unit
def test_check_register_lib_func_exists():
    """Test existance of <register_lib()> function across all Config Schema libraries
    in <config_schema> package."""

    # GIVEN: A list of all Config Schema modules <config_schema_modules_list> in <config_schema> package
    config_schema_package_name = "config_schema"
    config_schema_modules_list = []
    config_schema_package = importlib.import_module(config_schema_package_name)
    config_schema_package_path = config_schema_package.__path__[0]
    prefix_cut = len(config_schema_package_path) - len(config_schema_package_name)
    for package_path, _, file_names in os.walk(config_schema_package_path):
        for file_name in file_names:
            if file_name.endswith(".py") and not file_name.startswith("_"):
                module_path = os.path.join(package_path, file_name)[prefix_cut:-3].replace("/", ".")
                config_schema_modules_list.append(module_path)

    # WHEN: Checking all modules in <config_schema_modules_list> for <register_lib()> function existence
    modules_register_lib_existence_check = []
    for module_path in config_schema_modules_list:
        module = importlib.import_module(module_path)
        try:
            module.__getattribute__("register_lib")
            modules_register_lib_existence_check.append(True)
        except AttributeError:
            modules_register_lib_existence_check.append(False)

    # THEN: All modules in <config_schema_modules_list> have <register_lib()> function defined
    assert all(modules_register_lib_existence_check)


@pytest.mark.unit
def test_register_all_config_schema_libs_decorator():
    """Test <register_all_config_schema_libs> decorator from <utils.runtime> module."""

    # GIVEN: <register_all_config_schema_libs> decorator from <utils.runtime> module and a dummy main function
    @register_all_config_schema_libs
    def dummy_main():
        return True

    # WHEN: Calling the decorated <dummy_main()> function
    # THEN: The decorated <dummy_main()> function runs without raising an unexpected AttributeError
    try:
        dummy_main()
    except AttributeError as e:
        pytest.fail(f"Decorator <register_all_config_schema_libs> raised an unexpected AttributeError: {e}")
