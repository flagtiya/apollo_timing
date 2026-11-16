import json
from pathlib import Path
import tomllib

from apollo_timing import __version__


def test_distribution_runtime_profile_and_readme_versions_agree():
    root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((root / "pyproject.toml").read_text())
    profile = json.loads((root / "src/apollo_timing/release_profile.json").read_text())
    assert metadata["project"]["version"] == __version__ == profile["distribution_version"]
    for name in ("README.md", "README.zh-CN.md"):
        assert (root / name).read_text().splitlines()[0] == "# APOLLO-Timing " + __version__
