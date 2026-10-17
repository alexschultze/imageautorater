"""The CUDA self-repair: detect an NVIDIA GPU + CPU-only PyTorch, ask first, install before PyTorch is loaded."""
import shutil
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

import imageautorater.pipeline as pl

GPU = ["NVIDIA GeForce RTX 2060, 610.74"]


def fake_dist(tmp_path, monkeypatch, version_py):
    """Pretend 'torch' is installed with this torch/version.py."""
    from importlib import metadata
    if version_py is not None:
        f = tmp_path / "torch" / "version.py"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(version_py, encoding="utf-8")
    monkeypatch.setattr(metadata, "distribution", lambda name: NS(locate_file=lambda rel: tmp_path / rel))


def test_cpu_and_cuda_builds_are_told_apart_without_importing_torch(tmp_path, monkeypatch):
    fake_dist(tmp_path, monkeypatch, "__version__ = '2.14.1+cpu'\ndebug = False\ncuda: Optional[str] = None\nhip: Optional[str] = None\n")
    assert pl.torch_cuda_build() == (True, None)
    fake_dist(tmp_path, monkeypatch, "__version__ = '2.14.1+cu132'\ncuda: Optional[str] = '13.2'\nhip: Optional[str] = None\n")
    assert pl.torch_cuda_build() == (True, "13.2")
    fake_dist(tmp_path, monkeypatch, "something unexpected\n")
    assert pl.torch_cuda_build() == (True, "unknown")               # unknown layout: never touched
    (tmp_path / "torch" / "version.py").unlink()
    assert pl.torch_cuda_build() == (True, "unknown")
    from importlib import metadata

    def missing(name):
        raise metadata.PackageNotFoundError(name)
    monkeypatch.setattr(metadata, "distribution", missing)
    assert pl.torch_cuda_build() == (False, None)


def test_the_newest_cuda_build_the_driver_can_run_is_chosen():
    pick = pl.choose_cuda_index
    assert pick("13.3") == "cu132" and pick("13.2") == "cu132" and pick("13.0") == "cu130"
    assert pick("12.9") == "cu129" and pick("12.7") == "cu126" and pick("12.4") == "cu124" and pick("11.8") == "cu118"
    assert pick("10.2") == "cu118"                                    # older than anything offered: the oldest build
    assert pick(None) == "cu126" and pick("garbage") == "cu126"


def test_the_install_command_uses_uv_when_present_else_pip_with_the_right_index():
    py = "C:/env/Scripts/python.exe"
    assert pl.gpu_fix_command(py, "C:/uv.exe", "13.3") == [
        "C:/uv.exe", "pip", "install", "--python", py, "--reinstall", "torch", "torchvision", "--torch-backend=auto"]
    cmd = pl.gpu_fix_command(py, None, "12.8")
    assert cmd[:4] == [py, "-m", "pip", "install"] and "--force-reinstall" in cmd
    assert cmd[-2:] == ["--index-url", "https://download.pytorch.org/whl/cu128"]


@pytest.fixture
def world(tmp_path, monkeypatch):
    """An NVIDIA machine with a CPU-only PyTorch; subprocess.run records instead of installing."""
    calls = []
    monkeypatch.setattr(pl, "nvidia_gpus", lambda: GPU)
    monkeypatch.setattr(pl, "torch_cuda_build", lambda: (True, None))
    monkeypatch.setattr(pl, "driver_cuda_version", lambda: "13.3")
    monkeypatch.setattr(shutil, "which", lambda name: "C:/uv.exe" if name == "uv" else None)

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return NS(returncode=0, stdout="True 13.2", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    world_ = NS(calls=calls, tmp=tmp_path)
    return world_


def terminal(monkeypatch, answer):
    monkeypatch.setattr(sys, "stdin", NS(isatty=lambda: True))
    asked = []
    monkeypatch.setattr("builtins.input", lambda prompt="": asked.append(prompt) or answer)
    return asked


def test_with_the_flag_it_installs_without_asking_and_verifies(world):
    assert pl.maybe_fix_gpu(True, None) is True
    install, verify = world.calls
    assert install[:2] == ["C:/uv.exe", "pip"] and "--torch-backend=auto" in install and install[install.index("--python") + 1] == sys.executable
    assert verify[:2] == [sys.executable, "-c"] and "torch.cuda.is_available" in verify[2]   # checked in a fresh process


def test_in_a_terminal_it_asks_and_respects_the_answer(world, monkeypatch):
    asked = terminal(monkeypatch, "n")
    assert pl.maybe_fix_gpu(None, None) is False and world.calls == [] and "2-3 GB" in asked[0]
    terminal(monkeypatch, "")                                         # just Enter = yes
    assert pl.maybe_fix_gpu(None, None) is True and len(world.calls) == 2


def test_without_a_terminal_it_never_installs_on_its_own(world, monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", NS(isatty=lambda: False))
    assert pl.maybe_fix_gpu(None, None) is False and world.calls == []
    assert "--fix-gpu" in capsys.readouterr().out


def test_it_stays_out_of_the_way_when_there_is_nothing_to_fix(world, monkeypatch):
    assert pl.maybe_fix_gpu(False, None) is False                                   # --no-fix-gpu
    assert pl.maybe_fix_gpu(None, "cpu") is False                                    # --device cpu: the user chose
    monkeypatch.setattr(pl, "torch_cuda_build", lambda: (True, "13.2"))              # already a CUDA build
    assert pl.maybe_fix_gpu(True, None) is False
    monkeypatch.setattr(pl, "torch_cuda_build", lambda: (True, "unknown"))           # cannot tell: never touch
    assert pl.maybe_fix_gpu(True, None) is False
    monkeypatch.setattr(pl, "torch_cuda_build", lambda: (False, None))               # no PyTorch at all
    assert pl.maybe_fix_gpu(True, None) is False
    monkeypatch.setattr(pl, "torch_cuda_build", lambda: (True, None))
    monkeypatch.setattr(pl, "nvidia_gpus", lambda: [])                               # no NVIDIA GPU (Intel / AMD / none)
    assert pl.maybe_fix_gpu(True, None) is False and world.calls == []


def test_a_failed_install_does_not_stop_the_run(world, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: world.calls.append(cmd) or NS(returncode=1, stdout="", stderr=""))
    assert pl.maybe_fix_gpu(True, None) is False and len(world.calls) == 1          # no verification after a failure


def test_without_uv_pip_is_used_with_the_index_matching_the_driver(world, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(pl, "driver_cuda_version", lambda: "12.9")
    pl.maybe_fix_gpu(True, None)
    assert world.calls[0][-2:] == ["--index-url", "https://download.pytorch.org/whl/cu129"]


def test_the_fix_runs_before_anything_imports_pytorch(stubbed, tmp_path, monkeypatch):
    """The command line calls the fix first thing, with the --fix-gpu / --device options it was given."""
    from conftest import make_photo
    seen = []
    monkeypatch.setattr(pl, "maybe_fix_gpu", lambda choice, device: seen.append((choice, device)) or False)
    p = tmp_path / "P"
    make_photo(p, "a.png", "A")
    monkeypatch.setattr("sys.argv", ["imageautorater-rate", str(p), "--faces-only", "--faces-dir", str(tmp_path / "_f"), "--fix-gpu"])
    pl.main()
    assert seen == [(True, None)]
