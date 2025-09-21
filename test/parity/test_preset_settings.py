import pytest

from noisemaker import rng
from noisemaker.composer import Preset
from noisemaker.presets import PRESETS

from .utils import js_preset_settings

PRESET_NAMES = sorted(PRESETS().keys())


@pytest.mark.parametrize("name", PRESET_NAMES)
def test_preset_settings(name):
    seed = 1
    rng.set_seed(seed)
    preset = Preset(name, PRESETS())
    py_settings = {k: getattr(v, 'value', v) for k, v in preset.settings.items()}
    js_settings = js_preset_settings(name, seed)
    assert py_settings == js_settings

    rng.set_seed(seed)
    dsl_preset = Preset(name, PRESETS())
    dsl_settings = {k: getattr(v, 'value', v) for k, v in dsl_preset.settings.items()}
    assert dsl_settings == py_settings
