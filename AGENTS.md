You're working on Noisemaker, a procedural noise generation algorithm playground, written in Python. Read the README.md.

For the Javascript port: Read the VANILLA_JS_PORT_SPEC.md porting document.

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
