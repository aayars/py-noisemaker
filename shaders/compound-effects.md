# Effect Call Tree Analysis

Extracted from Python source code by analyzing @effect decorated functions.

## Topological Order

Overall effect evaluation order, respecting dependencies:

[[ DONE ]] **adjust_brightness**
[[ DONE ]] **adjust_contrast**
[[ DONE ]] **adjust_hue**
[[ DONE ]] **adjust_saturation**
[[ DONE ]] **convolve**
[[ DONE ]] **bloom**
[[ DONE ]] **blur**
[[ DONE ]] **color_map**
[[ DONE ]] **fxaa**
[[ DONE ]] **glyph_map**
[[ DONE ]] **grain**
[[ SKIP ]] **jpeg_decimate**
[[ DONE ]] **normalize**
[[ DONE ]] **aberration**
[[ DONE ]] **conv_feedback**
[[ DONE ]] **density_map**
[[ DONE ]] **derivative**
[[ SKIP ]] **dla**
[[ DONE ]] **false_color**
[[ DONE ]] **lens_distortion**
[[ DONE ]] **normal_map**
[[ DONE ]] **on_screen_display**
[[ DONE ]] **palette**
[[ WIP ]] **pixel_sort**
[[ TODO ]] **posterize**
[[ TODO ]] **refract**
[[ TODO ]] **grime**
[[ TODO ]] **lens_warp**
[[ TODO ]] **degauss**
[[ TODO ]] **reindex**
[[ TODO ]] **reverb**
[[ TODO ]] **ridge**
[[ TODO ]] **ripple**
[[ TODO ]] **rotate**
[[ TODO ]] **scanline_error**
[[ TODO ]] **shadow**
[[ TODO ]] **erosion_worms**
[[ TODO ]] **simple_frame**
[[ TODO ]] **sine**
[[ TODO ]] **smoothstep**
[[ TODO ]] **snow**
[[ TODO ]] **sobel_operator**
[[ TODO ]] **glowing_edges**
[[ TODO ]] **outline**
[[ TODO ]] **spooky_ticker**
[[ TODO ]] **texture**
[[ TODO ]] **tint**
[[ TODO ]] **nebula**
[[ TODO ]] **value_refract**
[[ TODO ]] **vaseline**
[[ TODO ]] **vhs**
[[ TODO ]] **vignette**
[[ TODO ]] **crt**
[[ TODO ]] **voronoi**
[[ TODO ]] **kaleido**
[[ TODO ]] **lowpoly**
[[ TODO ]] **vortex**
[[ TODO ]] **warp**
[[ TODO ]] **clouds**
[[ TODO ]] **spatter**
[[ TODO ]] **wobble**
[[ TODO ]] **wormhole**
[[ TODO ]] **light_leak**
[[ TODO ]] **worms**
[[ TODO ]] **fibers**
[[ TODO ]] **scratches**
[[ TODO ]] **sketch**
[[ TODO ]] **stray_hair**
[[ TODO ]] **frame**

## Leaf Effects (No Dependencies)

These 27 effects don't call any other effects:

- **adjust_brightness** ← used by: spatter
- **adjust_contrast** ← used by: spatter
- **adjust_hue** ← used by: crt
- **adjust_saturation**
- **bloom** ← used by: glowing_edges, light_leak, vaseline
- **blur**
- **color_map** ← used by: false_color
- **fxaa**
- **glyph_map**
- **grain**
- **jpeg_decimate**
- **normalize** ← used by: aberration, conv_feedback, convolve, crt, density_map, derivative, false_color, frame, glowing_edges, grime, kaleido, lens_distortion, lowpoly, normal_map, reverb, shadow, sketch, sobel_operator, spatter, vignette, voronoi, vortex, wormhole, worms
- **on_screen_display** ← used by: spooky_ticker
- **palette**
- **pixel_sort**
- **posterize** ← used by: glowing_edges, simple_frame
- **reindex** ← used by: erosion_worms
- **ridge**
- **ripple**
- **rotate** ← used by: nebula
- **scanline_error**
- **sine**
- **smoothstep**
- **snow**
- **tint** ← used by: nebula
- **vhs**
- **wobble**

## Compound Effects (With Dependencies)

### aberration

Direct dependencies: normalize

```
aberration
└── normalize
```

### clouds

Direct dependencies: convolve, shadow, warp

```
clouds
├── convolve
│   └── normalize
├── shadow
│   ├── convolve
│   └── normalize
└── warp
    └── refract
        └── convolve
```

### conv_feedback

Direct dependencies: convolve, normalize

```
conv_feedback
├── convolve
│   └── normalize
└── normalize
```

### convolve

Direct dependencies: normalize

```
convolve
└── normalize
```

### crt

Direct dependencies: aberration, adjust_hue, lens_warp, normalize, vignette

```
crt
├── aberration
│   └── normalize
├── adjust_hue
├── lens_warp
│   └── refract
│       └── convolve
│           └── normalize
├── normalize
└── vignette
    └── normalize
```

### degauss

Direct dependencies: lens_warp

```
degauss
└── lens_warp
    └── refract
        └── convolve
            └── normalize
```

### density_map

Direct dependencies: normalize

```
density_map
└── normalize
```

### derivative

Direct dependencies: convolve, normalize

```
derivative
├── convolve
│   └── normalize
└── normalize
```

### dla

Direct dependencies: convolve

```
dla
└── convolve
    └── normalize
```

### erosion_worms

Direct dependencies: convolve, reindex, shadow

```
erosion_worms
├── convolve
│   └── normalize
├── reindex
└── shadow
    ├── convolve
    └── normalize
```

### false_color

Direct dependencies: color_map, normalize

```
false_color
├── color_map
└── normalize
```

### fibers

Direct dependencies: worms

```
fibers
└── worms
    └── normalize
```

### frame

Direct dependencies: aberration, grime, light_leak, normalize, scratches, shadow, stray_hair, vignette

```
frame
├── aberration
│   └── normalize
├── grime
│   ├── derivative
│   │   ├── convolve
│   │   │   └── normalize
│   │   └── normalize
│   ├── normalize
│   └── refract
│       └── convolve
├── light_leak
│   ├── bloom
│   ├── vaseline
│   │   └── bloom
│   ├── voronoi
│   │   ├── normalize
│   │   └── refract
│   └── wormhole
│       └── normalize
├── normalize
├── scratches
│   └── worms
│       └── normalize
├── shadow
│   ├── convolve
│   └── normalize
├── stray_hair
│   └── worms
└── vignette
    └── normalize
```

### glowing_edges

Direct dependencies: bloom, convolve, normalize, posterize, sobel_operator

```
glowing_edges
├── bloom
├── convolve
│   └── normalize
├── normalize
├── posterize
└── sobel_operator
    ├── convolve
    └── normalize
```

### grime

Direct dependencies: derivative, normalize, refract

```
grime
├── derivative
│   ├── convolve
│   │   └── normalize
│   └── normalize
├── normalize
└── refract
    └── convolve
```

### kaleido

Direct dependencies: normalize, voronoi

```
kaleido
├── normalize
└── voronoi
    ├── normalize
    └── refract
        └── convolve
            └── normalize
```

### lens_distortion

Direct dependencies: normalize

```
lens_distortion
└── normalize
```

### lens_warp

Direct dependencies: refract

```
lens_warp
└── refract
    └── convolve
        └── normalize
```

### light_leak

Direct dependencies: bloom, vaseline, voronoi, wormhole

```
light_leak
├── bloom
├── vaseline
│   └── bloom
├── voronoi
│   ├── normalize
│   └── refract
│       └── convolve
│           └── normalize
└── wormhole
    └── normalize
```

### lowpoly

Direct dependencies: normalize, voronoi

```
lowpoly
├── normalize
└── voronoi
    ├── normalize
    └── refract
        └── convolve
            └── normalize
```

### nebula

Direct dependencies: rotate, tint

```
nebula
├── rotate
└── tint
```

### normal_map

Direct dependencies: convolve, normalize

```
normal_map
├── convolve
│   └── normalize
└── normalize
```

### outline

Direct dependencies: sobel_operator

```
outline
└── sobel_operator
    ├── convolve
    │   └── normalize
    └── normalize
```

### refract

Direct dependencies: convolve

```
refract
└── convolve
    └── normalize
```

### reverb

Direct dependencies: normalize

```
reverb
└── normalize
```

### scratches

Direct dependencies: worms

```
scratches
└── worms
    └── normalize
```

### shadow

Direct dependencies: convolve, normalize

```
shadow
├── convolve
│   └── normalize
└── normalize
```

### simple_frame

Direct dependencies: posterize

```
simple_frame
└── posterize
```

### sketch

Direct dependencies: derivative, normalize, vignette, warp, worms

```
sketch
├── derivative
│   ├── convolve
│   │   └── normalize
│   └── normalize
├── normalize
├── vignette
│   └── normalize
├── warp
│   └── refract
│       └── convolve
└── worms
    └── normalize
```

### sobel_operator

Direct dependencies: convolve, normalize

```
sobel_operator
├── convolve
│   └── normalize
└── normalize
```

### spatter

Direct dependencies: adjust_brightness, adjust_contrast, normalize, warp

```
spatter
├── adjust_brightness
├── adjust_contrast
├── normalize
└── warp
    └── refract
        └── convolve
            └── normalize
```

### spooky_ticker

Direct dependencies: on_screen_display

```
spooky_ticker
└── on_screen_display
```

### stray_hair

Direct dependencies: worms

```
stray_hair
└── worms
    └── normalize
```

### texture

Direct dependencies: shadow

```
texture
└── shadow
    ├── convolve
    │   └── normalize
    └── normalize
```

### value_refract

Direct dependencies: refract

```
value_refract
└── refract
    └── convolve
        └── normalize
```

### vaseline

Direct dependencies: bloom

```
vaseline
└── bloom
```

### vignette

Direct dependencies: normalize

```
vignette
└── normalize
```

### voronoi

Direct dependencies: normalize, refract

```
voronoi
├── normalize
└── refract
    └── convolve
        └── normalize
```

### vortex

Direct dependencies: convolve, normalize, refract

```
vortex
├── convolve
│   └── normalize
├── normalize
└── refract
    └── convolve
```

### warp

Direct dependencies: refract

```
warp
└── refract
    └── convolve
        └── normalize
```

### wormhole

Direct dependencies: normalize

```
wormhole
└── normalize
```

### worms

Direct dependencies: normalize

```
worms
└── normalize
```

