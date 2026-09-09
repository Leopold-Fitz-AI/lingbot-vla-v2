import importlib.util
from pathlib import Path

import pytest


def configuration():
    pytest.importorskip("transformers")
    path = Path(__file__).resolve().parents[1] / "lingbotvla/models/vla/lingbot_vla/configuration_lingbot_vla.py"
    spec = importlib.util.spec_from_file_location("recap_configuration_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LingbotVLAV2Config


def test_deployment_flags_and_initializer_survive_config_roundtrip():
    cls = configuration()
    config = cls(
        recap_training_backend="deployment",
        recap_adapter_initialization="orthogonal_matched_v1",
        recap_adapter_init_seed=971,
        recap_prompt_enabled=False,
        enable_visual_distillation=False,
        attention_implementation="eager",
    )
    restored = cls.from_dict(config.to_dict())
    assert restored.use_cache
    assert restored.recap_training_backend == "deployment"
    assert restored.recap_adapter_initialization == "orthogonal_matched_v1"
    assert restored.recap_adapter_init_seed == 971
    assert not restored.recap_prompt_enabled
    assert not restored.enable_visual_distillation
    assert restored.attention_implementation == "eager"


def test_missing_new_flags_retain_legacy_defaults():
    config = configuration()()
    assert config.recap_training_backend == "legacy"
    assert config.recap_adapter_initialization == "legacy_sin_v1"
    assert not config.use_cache
