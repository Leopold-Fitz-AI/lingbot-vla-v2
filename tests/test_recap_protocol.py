import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from lingbotvla.recap.protocol import load_environment_seed_map


def test_loads_exact_environment_seed_map():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "seeds.json"
        path.write_text(
            json.dumps({"schema_version": 1, "tasks": {"click_bell": [1, 3]}})
        )
        assert load_environment_seed_map(path) == {"click_bell": [1, 3]}


def test_rejects_duplicate_environment_seeds():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "seeds.json"
        path.write_text(
            json.dumps({"schema_version": 1, "tasks": {"click_bell": [1, 1]}})
        )
        with pytest.raises(ValueError, match="duplicates"):
            load_environment_seed_map(path)
