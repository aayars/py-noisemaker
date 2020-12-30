import os
import tempfile
import unittest

from noisemaker.composer import Effect, Preset

SHAPE = [256, 256, 3]

PRESETS = lambda: {
    "test-parent": {
        "settings": lambda: {
            "freq": 2,
            "sides": 4,
        },

        "generator": lambda settings: {
            "freq": settings['freq'],
            "octaves": 8,
        },

        "octaves": lambda settings: [
            Effect('kaleido', sides=settings['sides']),
        ],

        "post": lambda settings: [
            Effect('posterize'),
        ],
    },

    "test-child": {
        "layers": ["test-parent"],

        "settings": lambda: {
            "freq": 5,
        }
    },

    "test-grandchild": {
        "layers": ["test-child"],

        "settings": lambda: {
            "sides": 25,
        },

        "octaves": lambda settings: [
            Effect("derivative"),
        ],

        "post": lambda settings: [
            Effect("glitch"),
        ]
    },

    "test-invalid-layers": {
        "layers": ["does-not-exist"],
    },

}


class TestComposer(unittest.TestCase):
    def test_parent(self):
        """test-parent preset contains the expected values from the canned example."""

        preset = Preset('test-parent', PRESETS())

        assert preset.settings['freq'] == 2
        assert preset.settings['sides'] == 4

        assert preset.generator_kwargs['freq'] == 2
        assert preset.generator_kwargs['octaves'] == 8

        assert len(preset.octave_effects) == 1 and preset.octave_effects[0].func.__name__ == 'kaleido'
        assert len(preset.octave_effects[0].keywords) == 1 and preset.octave_effects[0].keywords['sides'] == 4

        assert len(preset.post_effects) == 1 and preset.post_effects[0].func.__name__ == 'posterize'
        assert len(preset.post_effects[0].keywords) == 0

    def test_child(self):
        """test-child preset contains the expected values from the canned example."""

        preset = Preset('test-child', PRESETS())

        assert preset.settings['freq'] == 5
        assert preset.settings['sides'] == 4

        assert preset.generator_kwargs['freq'] == 5
        assert preset.generator_kwargs['octaves'] == 8

        assert len(preset.octave_effects) == 1 and preset.octave_effects[0].func.__name__ == 'kaleido'
        assert len(preset.octave_effects[0].keywords) == 1 and preset.octave_effects[0].keywords['sides'] == 4

        assert len(preset.post_effects) == 1 and preset.post_effects[0].func.__name__ == 'posterize'
        assert len(preset.post_effects[0].keywords) == 0

    def test_grandchild(self):
        """test-grandchild preset contains the expected values from the canned example."""

        preset = Preset('test-grandchild', PRESETS())

        assert preset.settings['freq'] == 5
        assert preset.settings['sides'] == 25

        assert preset.generator_kwargs['freq'] == 5
        assert preset.generator_kwargs['octaves'] == 8

        assert len(preset.octave_effects) == 2
        assert preset.octave_effects[0].func.__name__ == 'kaleido'
        assert len(preset.octave_effects[0].keywords) == 1 and preset.octave_effects[0].keywords['sides'] == 25
        assert preset.octave_effects[1].func.__name__ == 'derivative'
        assert len(preset.octave_effects[1].keywords) == 0

        assert len(preset.post_effects) == 2
        assert preset.post_effects[0].func.__name__ == 'posterize'
        assert len(preset.post_effects[0].keywords) == 0
        assert preset.post_effects[1].func.__name__ == 'glitch'
        assert len(preset.post_effects[1].keywords) == 0

    def test_render(self):
        """Rendering an image to disk does not raise an exception."""

        preset = Preset('test-grandchild', PRESETS())

        with tempfile.TemporaryDirectory() as temp:
            preset.render(shape=SHAPE, filename=os.path.join(temp, "art.jpg"))

    def test_invalid_layers(self):
        """An invalid parent preset name raises an exception at preset creation time."""

        with self.assertRaises(ValueError):
            preset = Preset("test-invalid-layers", PRESETS())
