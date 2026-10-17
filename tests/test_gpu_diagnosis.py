"""When an NVIDIA GPU exists but is not used, the program must say so and say why (a CPU-only PyTorch is the usual cause)."""
from types import SimpleNamespace as NS

import pytest

import imageautorater.pipeline as pl


class Dev:
    def __init__(self, kind, index=None):
        self.type, self.index = kind, index

    def __str__(self):
        return self.type if self.index is None else f"{self.type}:{self.index}"


def fake_torch(cuda=None, hip=None, version="2.5.0"):
    return NS(__version__=version, version=NS(cuda=cuda, hip=hip), cuda=NS(get_device_name=lambda i=0: "NVIDIA RTX 4070"))


GPUS = ["NVIDIA RTX 4070, 551.86"]


def test_cpu_only_pytorch_on_a_machine_with_an_nvidia_gpu_is_called_out_with_the_fix():
    line, why = pl.hardware_report(fake_torch(cuda=None, version="2.5.0+cpu"), Dev("cpu"), GPUS)
    assert "CPU-only build" in line and "device: cpu" in line
    assert "NVIDIA GPU is present (NVIDIA RTX 4070, 551.86)" in why and "CPU-only build" in why and "NOT used" in why
    assert "uv pip install --reinstall torch torchvision --torch-backend=auto" in why


def test_cuda_build_that_cannot_reach_the_gpu_points_at_the_driver():
    line, why = pl.hardware_report(fake_torch(cuda="12.8"), Dev("cpu"), GPUS)
    assert "CUDA 12.8" in line
    assert "torch.cuda.is_available() is False" in why and "driver" in why and "--reinstall" not in why


def test_working_gpu_is_named_and_not_warned_about():
    line, why = pl.hardware_report(fake_torch(cuda="12.8"), Dev("cuda", 0), [])
    assert why is None and "device: cuda:0 (NVIDIA RTX 4070)" in line


def test_a_cpu_machine_without_nvidia_gets_no_warning():
    line, why = pl.hardware_report(fake_torch(cuda=None, version="2.5.0+cpu"), Dev("cpu"), [])
    assert why is None and "CPU-only build" in line


def test_no_nvidia_driver_means_no_gpus_found(monkeypatch):
    import shutil
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert pl.nvidia_gpus() == []


def test_asking_for_cuda_without_cuda_stops_with_an_explanation(monkeypatch):
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("this machine really has CUDA")
    monkeypatch.setattr(pl, "nvidia_gpus", lambda: GPUS)
    with pytest.raises(SystemExit) as stop:
        pl.pick_device("cuda")
    assert "--device cuda was requested" in str(stop.value) and "--torch-backend=auto" in str(stop.value)


def test_the_hardware_is_reported_once_per_run(monkeypatch, capsys):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(pl, "_hardware_reported", False)
    monkeypatch.setattr(pl, "nvidia_gpus", lambda: [])
    pl.pick_device("cpu")
    pl.pick_device("cpu")
    assert capsys.readouterr().out.count("PyTorch ") == 1
