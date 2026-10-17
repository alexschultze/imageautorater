"""Packaging rules that protect the install on other people's machines."""
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_pyproject_does_not_pin_torch_to_an_index():
    """uv applies [tool.uv.sources] / [[tool.uv.index]] of a package to everyone who installs it from git. A torch entry
    there silently overrides --torch-backend=auto and installs the CPU-only PyTorch on NVIDIA machines (this really
    happened: `uv pip install git+... --torch-backend=auto` gave +cpu until the entry was removed)."""
    uv = PYPROJECT.get("tool", {}).get("uv", {})
    sources = uv.get("sources", {})
    assert not any(name in sources for name in ("torch", "torchvision", "torchaudio")), sources
    assert not [i for i in uv.get("index", []) if "pytorch" in str(i).lower()], uv.get("index")


def test_ci_chooses_the_torch_build_on_the_command_line_and_does_not_use_uv_sync():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "--torch-backend=cpu" in ci and "uv sync" not in ci


def test_setuptools_stays_below_the_release_without_pkg_resources():
    assert "setuptools<82" in PYPROJECT["project"]["dependencies"]


def test_the_commands_users_are_told_to_run_exist():
    scripts = PYPROJECT["project"]["scripts"]
    assert set(scripts) == {"imageautorater-rate", "imageautorater-write-xmp", "imageautorater-write-exif", "imageautorater-gallery"}
