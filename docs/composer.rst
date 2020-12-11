Noisemaker Composer
===================

Noisemaker Composer is a high-level interface for creating generative art with noise. The design is informed by lessons learned from the previous incarnation of presets in Noisemaker.

Limitations With Previous Approach
----------------------------------

Noisemaker's previous strategy for applying parameterized effects was to call into a single `large top-level monolithic function <api.html#noisemaker.effects.post_process>`_. This suffered from several limitations:

- Not dynamic enough. Always applied effects in the same hard-coded order.
- Did not allow repeated effects, unless bespoke logic to do so was shoehorned into the monolith.
- Indirect. Rather than providing direct access to `generators <api.html#module-noisemaker.generators>`_ and `effects <api.html#module-noisemaker.effects>`_, the old post-processing monolith acted as a middleman and strict gatekeeper.
- Error prone. Sloppy parameter handling, with no obviously rational way to add validation.
- `An unwieldy number of parameters <api.html#noisemaker.effects.post_process>`_, which is not a great way to program. Adding new features was difficult, and only compounded the problem over time.

Composer Presets
----------------

The current solution attempts to be flexible and highly composable. Composer Presets expose Noisemaker's lower-level `generator <api.html#module-noisemaker.generators>`_ and `effect <api.html#module-noisemaker.effects>`_ APIs, and are modeled using terse syntax which can be finely tailored per-preset. The intent behind this design was to provide a compact and maintainable interface which answers five key questions for each preset:

1) Which presets are being built on?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Effects and presets are intended to be combined with and riff off each other, but repetition in code is distateful, especially when we have to copy settings around. To minimize copied settings, Composer Presets may "extend" parent presets, and/or refer to other presets inline.

The lineage of ancestor presets is modeled in each preset's "extends" list, which is a flat list of preset names.

2) What are the meaningful variables?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each preset may need to reuse values, or tweak a value which was already set by a parent. To facilitate this, presets have an optional bank of settings which may be plugged in and overridden as needed.

Reusable settings are modeled in each preset's "settings" dictionary. Extending a preset inherits this dictionary, allowing preset authors to override or add key/value pairs. This is a free-form dictionary, and authors may stash any arbitrary values they need here.

3) What are the noise generation parameters?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Noisemaker's noise generator has several parameters, and these simply need to live somewhere. Noise generator parameters are modeled in each preset's "generator" dictionary. Generator parameters may be defined in this dictionary, or can be fed in from settings. Just as with "settings", extending a preset inherits this dictionary, enabling preset authors to override or add key/value pairs. Unlike "settings", the keys found in this dictionary are strictly validated and must be valid parameters to `noisemaker.generators.multires <api.html#noisemaker.generators.multires>`

4) Which effects should be applied to each octave?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Preset authors should be able to specify a list of effects which get applied to each octave of noise. Historically, the per-octave effects in Noisemaker were constrained by hard-coded logic. In Composer Presets, authors may specify an arbitrary list of effects.

Per-octave effects are modeled in each preset's "octaves" list, which specifies parameterized effects functions. Per-octave effect parameters may be defined in this list, or can be fed in from settings. Extending a preset inherits this list, allowing authors to append additional effects.

5) Which effects should be applied after flattening layers?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similar to how per-octave effects were originally implemented, post effects in Noisemaker were hard-coded and inflexible. Composer Presets aim to break this pattern by enabling preset authors to specify an ordered list of "final pass" effects.

Post-reduce effects are modeled in each preset's "post" section, which is a flat list of parameterized effects functions and presets. Post-processing effect parameters may be defined in this list, or can be fed in from settings. Extending a preset inherits this list, allowing authors to append additional effects and inline presets. A preset's post-processing list can contain effects as well as links to other presets, enabling powerful expression of nested macros.
