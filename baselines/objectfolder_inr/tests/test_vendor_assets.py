from pathlib import Path


def test_vendored_objectfolder_runtime_is_complete():
    root = Path(__file__).resolve().parents[1] / "vendor" / "objectfolder"
    required = (
        "LICENSE",
        "README.md",
        "taxim_render.py",
        "basics/__init__.py",
        "basics/CalibData.py",
        "basics/sensorParams.py",
        "calibs/dataPack.npz",
        "calibs/depth_bg.npy",
        "calibs/polycalib.npz",
        "calibs/real_bg.npy",
    )
    missing = [relative for relative in required if not (root / relative).is_file()]
    assert not missing
