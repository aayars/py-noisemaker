# Copilot Instructions for Noisemaker

This guide is for AI coding agents working on Noisemaker, a procedural noise generation playground in Python, JS, and WGSL shaders. Follow these project-specific conventions and workflows for maximum productivity.

## Architecture Overview
- **Python core**: Main implementation in `noisemaker/` (TensorFlow-based). Entry points: `cli.py`, `composer.py`, `generators.py`, `effects.py`.
- **Presets**: Shared DSL file at `dsl/presets.dsl` used by both Python and JS. Never move or duplicate this file.
- **JS port**: Located in `js/`. Follows strict parity with Python. See `js/doc/VANILLA_JS_PORT_SPEC.md` and `js/doc/PY_JS_PARITY_SPEC.md`.
- **Shaders**: WGSL shaders in `shaders/effects/`, demo viewer in `shaders/demo.html`. Shaders are independent from Python/JS code.

## Developer Workflows
- **Python setup**:
  1. `python3 -m venv venv && source venv/bin/activate`
  2. `pip install -r requirements.txt`
  3. `python setup.py develop && python setup.py install_scripts`
  4. Run CLI: `noisemaker --help`
- **Testing**: Run `pytest` before submitting Python changes. JS tests are in `test/` and use Node only for testing.
- **Docker**: See `docker/README.md` for running Noisemaker in containers. Output must be mounted to `/output`.

## Conventions & Patterns
- **Presets**: Only edit `dsl/presets.dsl` if explicitly requested. Do not create new preset locations.
- **Randomness**: Never simulate weighted randomness by repeating values in collections. Use explicit probability checks (e.g., `random() < p`).
- **JS/Node**: Node is allowed only for tests. Production JS is vanilla and browser-only.
- **Parity**: When porting, JS must match Python reference. Do not change Python for JS parity.
- **Shaders**:
  - All textures are 4-channel RGBA. Do not count channels; always assume 4.
  - WGSL struct members end with `,` (comma), not `;` (semicolon).
  - Controls in `shaders/demo.html` must match Python effect params (except "shape").

## Integration Points
- **Python CLI**: `noisemaker` command for image generation and effects.
- **Python API**: `Preset` class in `composer.py` for programmatic use.
- **JS**: Entry in `js/noisemaker/`, tests in `test/`.
- **Shaders**: Viewer in `shaders/demo.html`, tests in `shaders/tests/`.

## Shader Testing
- Resolve *all* console errors before returning a solution.
- The shader must run in the browser with no errors, or your turn is not over.

## Documentation
- Do not produce documentation unless requested.

## References
- Main docs: [README.md](../README.md)
- Python agent rules: [AGENTS.md](../AGENTS.md)
- Shader agent rules: [shaders/AGENTS.md](../shaders/AGENTS.md)
- Docker usage: [docker/README.md](../docker/README.md)

---

**Always fix forward. Never mask or hobble tests. Be an honest developer.**
