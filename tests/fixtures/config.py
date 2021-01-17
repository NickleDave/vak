"""fixtures relating to .toml configuration files"""
import pytest
import toml


@pytest.fixture
def test_configs_root(test_data_root):
    """Path that points to test_data/configs

    Two types of config files in this directory:
    1) those used by the src/scripts/test_data/test_data_generate.py script.
       All configs that start with ``test_`` prefix.
    2) those used by tests that are static, e.g., ``invalid_section_config.toml``

    This fixture facilitates access to type (2), e.g. in test_config/test_parse
    """
    return test_data_root.joinpath('configs')


@pytest.fixture
def invalid_section_config_path(test_configs_root):
    return test_configs_root.joinpath('invalid_section_config.toml')


@pytest.fixture
def invalid_option_config_path(test_configs_root):
    return test_configs_root.joinpath('invalid_option_config.toml')


@pytest.fixture
def generated_test_configs_root(generated_test_data_root):
    return generated_test_data_root.joinpath('configs')


# ---- path to config files ----
@pytest.fixture
def all_generated_configs(generated_test_configs_root):
    return sorted(generated_test_configs_root.glob('test*toml'))


@pytest.fixture
def all_generated_train_configs(generated_test_configs_root):
    return sorted(generated_test_configs_root.glob('test_train*toml'))


@pytest.fixture
def all_generated_learncurve_configs(generated_test_configs_root):
    return sorted(generated_test_configs_root.glob('test_learncurve*toml'))


@pytest.fixture
def all_generated_eval_configs(generated_test_configs_root):
    return sorted(generated_test_configs_root.glob('test_eval*toml'))


@pytest.fixture
def all_generated_predict_configs(generated_test_configs_root):
    return sorted(generated_test_configs_root.glob('test_predict*toml'))


def _return_toml(toml_path):
    """return config files loaded into dicts with toml library
    used to test functions that parse config sections, taking these dicts as inputs"""
    with toml_path.open('r') as fp:
        config_toml = toml.load(fp)
    return config_toml


# ----  config toml from paths ----
@pytest.fixture
def all_generated_configs_toml(all_generated_configs):
    return [_return_toml(config) for config in all_generated_configs]


@pytest.fixture
def all_generated_train_configs_toml(all_generated_train_configs):
    return [_return_toml(config) for config in all_generated_train_configs]


@pytest.fixture
def all_generated_learncurve_configs_toml(all_generated_learncurve_configs):
    return [_return_toml(config) for config in all_generated_learncurve_configs]


@pytest.fixture
def all_generated_eval_configs_toml(all_generated_eval_configs):
    return [_return_toml(config) for config in all_generated_eval_configs]


@pytest.fixture
def all_generated_predict_configs_toml(all_generated_predict_configs):
    return [_return_toml(config) for config in all_generated_predict_configs]


# ---- config toml + path pairs ----
@pytest.fixture
def all_generated_configs_toml_path_pairs(all_generated_configs):
    """zip of tuple pairs: (dict, pathlib.Path)
    where ``Path`` is path to .toml config file and ``dict`` is
    the .toml config from that path
    loaded into a dict with the ``toml`` library
    """
    return zip(
        [_return_toml(config) for config in all_generated_configs],
        all_generated_configs,
    )


@pytest.fixture
def all_generated_train_configs_toml_path_pairs(all_generated_train_configs):
    """zip of tuple pairs: (dict, pathlib.Path)
    where ``Path`` is path to .toml config file and ``dict`` is
    the .toml config from that path
    loaded into a dict with the ``toml`` library
    """
    return zip(
        [_return_toml(config) for config in all_generated_train_configs],
        all_generated_train_configs,
    )


@pytest.fixture
def all_generated_learncurve_configs_toml_path_pairs(all_generated_learncurve_configs):
    """zip of tuple pairs: (dict, pathlib.Path)
    where ``Path`` is path to .toml config file and ``dict`` is
    the .toml config from that path
    loaded into a dict with the ``toml`` library
    """
    return zip(
        [_return_toml(config) for config in all_generated_learncurve_configs],
        all_generated_learncurve_configs,
    )


@pytest.fixture
def all_generated_eval_configs_toml_path_pairs(all_generated_eval_configs):
    """zip of tuple pairs: (dict, pathlib.Path)
    where ``Path`` is path to .toml config file and ``dict`` is
    the .toml config from that path
    loaded into a dict with the ``toml`` library
    """
    return zip(
        [_return_toml(config) for config in all_generated_eval_configs],
        all_generated_eval_configs,
    )


@pytest.fixture
def all_generated_predict_configs_toml_path_pairs(all_generated_predict_configs):
    """zip of tuple pairs: (dict, pathlib.Path)
    where ``Path`` is path to .toml config file and ``dict`` is
    the .toml config from that path
    loaded into a dict with the ``toml`` library
    """
    return zip(
        [_return_toml(config) for config in all_generated_predict_configs],
        all_generated_predict_configs,
    )
