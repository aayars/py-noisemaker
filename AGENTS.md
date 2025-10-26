You're working on Noisemaker, a procedural noise generation algorithm playground, written in Python. Read the README.md.

## Bootstrapping the Python environment

1. Create a virtual environment and activate it:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies and set up the local package:

   ```bash
   pip install -r requirements.txt
   python setup.py develop
   python setup.py install_scripts
   ```

3. Verify the installation by running the CLI:

   ```bash
   noisemaker --help
   ```

4. Run the test suite before submitting changes (only if modifying Python code):

   ```bash
   pytest
   ```

## Docs

readthedocs content is for the *Python version only*

## Presets

Presets are located at /dsl/presets.dsl. This same file is used by both the Python and JS implementations. *Do not* invent new locations or modify presets unless explicitly requested.

## Javascript

JS port is under js/

*NO NODE* allowed except for tests.

Read and follow to the letter:
    - js/doc/VANILLA_JS_PORT_SPEC.md porting document
    - js/doc/PY_JS_PARITY_SPEC.md cross-language parity requirements

Never simulate weighted randomness by repeating values in collections passed to
`random_member`; use explicit probability checks instead (e.g., `random() < p`).

## Javascript/Python Parity Testing

- When the focus is JS, you may not change the reference python implementation.
- You may not disable, remove, or hobble tests.
- Do not mask, cover, or attempt to obscure actual problems.
- Always fix forward.
- Be an honest developer.

## Shaders

Shaders implementation is under shaders/
