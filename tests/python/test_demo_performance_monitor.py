import importlib.util
import json
import sys
from pathlib import Path


def _load_demo_module():
    root = Path(__file__).resolve().parents[2]
    demo_path = root / "demos" / "demo_performance_monitor.py"
    spec = importlib.util.spec_from_file_location("demo_performance_monitor", demo_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_performance_demo_rigid_outputs(tmp_path):
    module = _load_demo_module()
    cfg = module.DemoConfig()
    cfg.scene = "rigid"
    cfg.warmup_steps = 2
    cfg.measured_steps = 5
    cfg.top_n = 3
    cfg.output_dir = str(tmp_path / "rigid_profile")
    outputs = module.run_demo(cfg)

    summary_path = Path(outputs["summary_json"])
    trace_path = Path(outputs["trace_json"])
    assert summary_path.exists()
    assert trace_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    all_phase_names = {item["name"] for item in summary["all_phase_stats"]}
    assert "world.total" in all_phase_names
    assert summary["last_frame_metrics"]["bodies"] == 1000.0


def test_performance_demo_fluid_outputs(tmp_path):
    module = _load_demo_module()
    cfg = module.DemoConfig()
    cfg.scene = "fluid"
    cfg.warmup_steps = 1
    cfg.measured_steps = 3
    cfg.top_n = 5
    cfg.output_dir = str(tmp_path / "fluid_profile")
    outputs = module.run_demo(cfg)

    summary = json.loads(Path(outputs["summary_json"]).read_text(encoding="utf-8"))
    all_phase_names = {item["name"] for item in summary["all_phase_stats"]}
    assert "fluid.total" in all_phase_names
    assert "world.total" in all_phase_names
    assert summary["last_frame_metrics"]["fluid_particles"] > 0.0
