from pathlib import Path

from noisemaker.dsl.tokenizer import tokenize
from noisemaker.dsl.parser import parse, PRESET_KEYS


def test_presets_dsl_parses():
    path = Path(__file__).resolve().parent.parent / "dsl" / "presets.dsl"
    text = path.read_text()
    ast = parse(tokenize(text), enforce_preset_keys=False)
    presets = ast["body"]["properties"]
    for prop in presets:
        value = prop["value"]
        assert value["type"] == "ObjectExpr"
        keys = {p["key"] for p in value["properties"]}
        assert keys.issubset(PRESET_KEYS)
