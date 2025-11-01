"""
Sphinx extension for embedding live Noisemaker.js examples.

This extension provides the `.. noisemaker-live::` directive which generates
interactive canvas elements that render Noisemaker presets in the browser.

Usage:
    .. noisemaker-live::
       :preset: acid
       :seed: 12345
       :width: 512
       :height: 512
       :caption: An acid preset example

The directive generates a canvas element with data attributes that the
JavaScript runtime (noisemaker-live.js) uses to render the example.
"""

from docutils import nodes
from docutils.parsers.rst import Directive, directives
from sphinx.application import Sphinx


class NoisemakerLiveDirective(Directive):
    """
    Directive for embedding live Noisemaker canvas examples.
    
    Options:
        preset: Name of the preset to render (required)
        seed: Random seed (default: 42)
        width: Canvas width in pixels (default: 512)
        height: Canvas height in pixels (default: 512)
        caption: Optional caption text
        time: Time parameter for animated presets (default: 0.0)
        frame: Frame parameter for animated presets (default: 0.0)
    """
    
    has_content = False
    required_arguments = 0
    optional_arguments = 0
    option_spec = {
        'preset': directives.unchanged_required,
        'seed': directives.nonnegative_int,
        'width': directives.nonnegative_int,
        'height': directives.nonnegative_int,
        'caption': directives.unchanged,
        'time': directives.unchanged,
        'frame': directives.unchanged,
    }
    
    def run(self):
        preset = self.options.get('preset')
        seed = self.options.get('seed', 42)
        width = self.options.get('width', 512)
        height = self.options.get('height', 512)
        caption = self.options.get('caption', '')
        time = self.options.get('time', '0.0')
        frame = self.options.get('frame', '0.0')
        
        # Create container div
        container_attrs = {
            'class': 'noisemaker-live-container',
        }
        container = nodes.container(**container_attrs)
        
        # Create canvas element with data attributes
        canvas_html = f'''
<div class="noisemaker-live-container">
    <div class="noisemaker-live-canvas-wrapper">
        <canvas class="noisemaker-live-canvas"
                data-preset="{preset}"
                data-seed="{seed}"
                data-width="{width}"
                data-height="{height}"
                data-time="{time}"
                data-frame="{frame}"
                width="{width}"
                height="{height}">
        </canvas>
        <div class="noisemaker-live-loading">Loading...</div>
        <div class="noisemaker-live-error" style="display: none;"></div>
        <button class="noisemaker-live-random" title="Generate with random seed">Random</button>
    </div>
    {f'<p class="noisemaker-live-caption">{caption}</p>' if caption else ''}
</div>
'''
        
        # Add raw HTML node
        raw = nodes.raw('', canvas_html, format='html')
        container += raw
        
        return [container]


def setup(app: Sphinx):
    """Register the directive with Sphinx."""
    app.add_directive('noisemaker-live', NoisemakerLiveDirective)
    
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
