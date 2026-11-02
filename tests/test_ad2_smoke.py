import ast
from pathlib import Path

import pytest

SMOKE = Path(__file__).resolve().parents[1] / "scripts" / "smoke_ad2.py"


def aspect_ratio_changed(original_hw, target_hw):
    tree = ast.parse(SMOKE.read_text(encoding="utf-8"))
    func = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "aspect_ratio_changed"
    )
    namespace = {}
    exec(compile(ast.Module(body=[func], type_ignores=[]), str(SMOKE), "exec"), namespace)
    return namespace["aspect_ratio_changed"](original_hw, target_hw)


@pytest.mark.parametrize(
    ("original", "target", "changed"),
    [
        ((288, 288), (288, 288), False),
        ((10, 10), (256, 256), False),
        ((100, 200), (50, 100), False),
        ((2048, 2448), (288, 288), True),
        ((2048, 2448), (256, 256), True),
        ((100, 200), (50, 50), True),
    ],
)
def test_aspect_ratio_changed_detects_stretch(original, target, changed):
    assert aspect_ratio_changed(original, target) is changed


def test_tuple_versus_list_size_compare_is_not_an_aspect_check():
    original = (2048, 2448)
    target = (288, 288)
    assert (tuple(original) != list(target)) is True
    assert aspect_ratio_changed(original, target) is True
    square = (256, 256)
    assert (tuple(square) != list(target)) is True
    assert aspect_ratio_changed(square, target) is False


def test_smoke_script_uses_cross_product_not_tuple_list_compare():
    text = SMOKE.read_text(encoding="utf-8")
    assert 'tuple(sample["original_size_hw"]) != list(load_size)' not in text
    assert "height * target_w != width * target_h" in text
    assert "from lc_simplenet.geometry" not in text


def test_smoke_defaults_to_256_square_full_frame():
    text = SMOKE.read_text(encoding="utf-8")
    assert 'parser.add_argument("--image-height", type=int, default=256)' in text
    assert 'parser.add_argument("--image-width", type=int, default=256)' in text
    assert '"crop": None' in text
    assert '"geometry_frozen": False' in text
