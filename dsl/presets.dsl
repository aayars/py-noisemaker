{
  "1969": {
    layers: ["symmetry", "voronoi", "posterize-outline", "distressed"],
    settings: {
      color_space: ColorSpace.rgb,
      dist_metric: DistanceMetric.euclidean,
      palette_on: false,
      voronoi_alpha: 0.5 + random() * 0.5,
      voronoi_diagram_type: VoronoiDiagramType.color_range,
      voronoi_point_corners: true,
      voronoi_point_distrib: PointDistribution.circular,
      voronoi_point_freq: random_int(3, 5) * 2,
      voronoi_nth: random_int(1, 3),
    },
  },

  "1976": {
    layers: ["voronoi", "grain", "saturation"],
    settings: {
      dist_metric: DistanceMetric.triangular,
      saturation_final: 0.25 + random() * 0.125,
      voronoi_diagram_type: VoronoiDiagramType.color_regions,
      voronoi_nth: 0,
      voronoi_point_distrib: PointDistribution.random,
      voronoi_point_freq: 2,
    },
  },

  "1985": {
    layers: ["reindex-post", "voronoi", "palette", "random-hue", "spatter-post", "be-kind-rewind", "spatter-final"],
    settings: {
      dist_metric: DistanceMetric.chebyshev,
      freq: random_int(10, 15),
      reindex_range: 0.2 + random() * 0.1,
      spline_order: InterpolationType.constant,
      voronoi_diagram_type: VoronoiDiagramType.range,
      voronoi_nth: 0,
      voronoi_point_distrib: PointDistribution.random,
      voronoi_refract: 0.2 + random() * 0.1,
    },
  },

  "2001": {
    layers: ["analog-glitch", "invert", "posterize", "vignette-bright", "aberration"],
    settings: {
      mask: ValueMask.bank_ocr,
      mask_repeat: random_int(9, 12),
      spline_order: InterpolationType.cosine,
      vignette_bright_alpha: 0.75 + random() * 0.25,
      posterize_levels: random_int(1, 2),
    },
  },

  "2d-chess": {
    layers: ["value-mask", "voronoi", "maybe-rotate"],
    settings: {
      corners: true,
      dist_metric: random_member(DistanceMetric.absolute_members()),
      freq: 8,
      mask: ValueMask.chess,
      spline_order: InterpolationType.constant,
      voronoi_alpha: 0.5 + random() * 0.5,
      voronoi_diagram_type: coin_flip() ? VoronoiDiagramType.color_range : random_member([
        VoronoiDiagramType.range,
        VoronoiDiagramType.color_range,
        VoronoiDiagramType.regions,
        VoronoiDiagramType.color_regions,
        VoronoiDiagramType.range_regions,
      ]),
      voronoi_nth: random_int(0, 1) * random_int(0, 63),
      voronoi_point_corners: true,
      voronoi_point_distrib: PointDistribution.square,
      voronoi_point_freq: 8,
    },
  },

  "aberration": {
    settings: {
      aberration_displacement: 0.0125 + random() * 0.000625,
    },
    final: [aberration(displacement: settings.aberration_displacement)],
  },

  "acid": {
    layers: ["basic", "reindex-post", "normalize"],
    settings: {
      color_space: ColorSpace.rgb,
      freq: random_int(10, 15),
      octaves: 8,
      reindex_range: 1.25 + random() * 1.25,
    },
  },

  "acid-droplets": {
    layers: ["multires", "reflect-octaves", "density-map", "random-hue", "bloom", "shadow", "saturation"],
    settings: {
      freq: random_int(8, 12),
      hue_range: 0,
      lattice_drift: 1.0,
      mask: ValueMask.sparse,
      mask_static: true,
      palette_on: false,
      reflect_range: 7.5 + random() * 3.5,
    },
  },

  "acid-grid": {
    layers: ["voronoi-refract", "sobel", "funhouse", "bloom"],
    settings: {
      dist_metric: DistanceMetric.euclidean,
      lattice_drift: coin_flip(),
      voronoi_alpha: 0.333 + random() * 0.333,
      voronoi_diagram_type: VoronoiDiagramType.color_range,
      voronoi_point_distrib: random_member(PointDistribution.grid_members()),
      voronoi_point_freq: 4,
      voronoi_point_generations: 2,
      warp_range: 0.125 + random() * 0.0625,
    },
  },

  "acid-wash": {
    layers: ["basic", "funhouse", "ridge", "shadow", "saturation"],
    settings: {
      freq: random_int(4, 6),
      hue_range: 1.0,
      ridges: true,
      warp_octaves: 8,
    },
  },

  "activation-signal": {
    layers: ["value-mask", "glitchin-out"],
    settings: {
      color_space: random_member(ColorSpace.color_members()),
      freq: 4,
      mask: ValueMask.white_bear,
      spline_order: InterpolationType.constant,
    },
  },

  "aesthetic": {
    layers: ["basic", "maybe-derivative-post", "spatter-post", "maybe-invert", "be-kind-rewind", "spatter-final"],
    settings: {
      corners: true,
      distrib: random_member([ValueDistribution.column_index, ValueDistribution.ones, ValueDistribution.row_index]),
      freq: random_int(3, 5) * 2,
      mask: ValueMask.chess,
      spline_order: InterpolationType.constant,
    },
  },

  "alien-terrain": {
    layers: ["multires-ridged", "invert", "voronoi", "derivative-octaves", "invert", "erosion-worms", "bloom", "shadow", "grain", "saturation"],
    settings: {
      grain_contrast: 1.5,
      deriv_alpha: 0.25 + random() * 0.125,
      dist_metric: DistanceMetric.euclidean,
      erosion_worms_alpha: 0.05 + random() * 0.025,
      erosion_worms_density: random_int(150, 200),
      erosion_worms_inverse: true,
      erosion_worms_xy_blend: 0.333 + random() * 0.16667,
      freq: random_int(3, 5),
      hue_rotation: 0.875,
      hue_range: 0.25 + random() * 0.25,
      palette_on: false,
      voronoi_alpha: 0.5 + random() * 0.25,
      voronoi_diagram_type: VoronoiDiagramType.flow,
      voronoi_point_freq: 10,
      voronoi_point_distrib: PointDistribution.random,
      voronoi_refract: 0.25 + random() * 0.125,
    },
  },

  "alien-glyphs": {
    layers: ["entities", "maybe-rotate", "smoothstep-narrow", "posterize", "grain", "saturation"],
    settings: {
      corners: true,
      mask: random_member([ValueMask.arecibo_num, ValueMask.arecibo_bignum, ValueMask.arecibo_nucleotide]),
      mask_repeat: random_int(6, 12),
      refract_range: 0.025 + random() * 0.0125,
      refract_signed_range: false,
      refract_y_from_offset: true,
      spline_order: random_member([InterpolationType.linear, InterpolationType.cosine]),
    },
  },

  "alien-transmission": {
    layers: ["analog-glitch", "sobel", "glitchin-out"],
    settings: {
      mask: random_member(ValueMask.procedural_members()),
    },
  },

  "analog-glitch": {
    layers: ["value-mask"],
    settings: {
      mask: random_member([ValueMask.alphanum_hex, ValueMask.lcd, ValueMask.fat_lcd]),
      mask_repeat: random_int(20, 30),
    },
    generator: {
      freq: mask_freq(settings.mask, settings.mask_repeat),
    },
  },

  "arcade-carpet": {
    layers: ["multires-alpha", "funhouse", "posterize", "nudge-hue", "carpet", "bloom", "contrast-final"],
    settings: {
      color_space: ColorSpace.rgb,
      distrib: ValueDistribution.exp,
      hue_range: 1,
      mask: ValueMask.sparser,
      mask_static: true,
      octaves: 2,
      palette_on: false,
      posterize_levels: 3,
      warp_freq: random_int(25, 25),
      warp_range: 0.03 + random() * 0.015,
      warp_octaves: 1,
    },
    generator: {
      freq: settings.warp_freq,
    },
  },

  "are-you-human": {
    layers: ["multires", "value-mask", "funhouse", "density-map", "saturation", "maybe-invert", "aberration", "snow"],
    settings: {
      freq: 15,
      hue_range: random() * 0.25,
      hue_rotation: random(),
      mask: ValueMask.truetype,
    },
  },

  "band-together": {
    layers: ["basic", "reindex-post", "funhouse", "shadow", "normalize", "grain"],
    settings: {
      freq: random_int(6, 12),
      reindex_range: random_int(8, 12),
      warp_range: 0.333 + random() * 0.16667,
      warp_octaves: 8,
      warp_freq: random_int(2, 3),
    },
  },

  "basic": {
    unique: true,
    layers: ["maybe-palette"],
    settings: {
      brightness_distrib: null,
      color_space: random_member(ColorSpace.color_members()),
      corners: false,
      distrib: ValueDistribution.uniform,
      freq: [random_int(2, 4), random_int(2, 4)],
      hue_distrib: null,
      hue_range: random() * 0.25,
      hue_rotation: random(),
      lattice_drift: 0.0,
      mask: null,
      mask_inverse: false,
      mask_static: false,
      octave_blending: OctaveBlending.falloff,
      octaves: 1,
      ridges: false,
      saturation: 1.0,
      saturation_distrib: null,
      sin: 0.0,
      spline_order: InterpolationType.bicubic,
    },
    generator: {
      brightness_distrib: settings.brightness_distrib,
      color_space: settings.color_space,
      corners: settings.corners,
      distrib: settings.distrib,
      freq: settings.freq,
      hue_distrib: settings.hue_distrib,
      hue_range: settings.hue_range,
      hue_rotation: settings.hue_rotation,
      lattice_drift: settings.lattice_drift,
      mask: settings.mask,
      mask_inverse: settings.mask_inverse,
      mask_static: settings.mask_static,
      octave_blending: settings.octave_blending,
      octaves: settings.octaves,
      ridges: settings.ridges,
      saturation: settings.saturation,
      saturation_distrib: settings.saturation_distrib,
      sin: settings.sin,
      spline_order: settings.spline_order,
    },
  },

  "basic-low-poly": {
    layers: ["basic", "low-poly", "grain", "saturation"],  },

  "basic-voronoi": {
    layers: ["basic", "voronoi"],
    settings: {
      voronoi_diagram_type: random_member([
        VoronoiDiagramType.color_range,
        VoronoiDiagramType.color_regions,
        VoronoiDiagramType.range_regions,
        VoronoiDiagramType.color_flow,
      ]),
    },
  },

  "basic-voronoi-refract": {
    layers: ["basic", "voronoi"],
    settings: {
      dist_metric: random_member(DistanceMetric.absolute_members()),
      hue_range: 0.25 + random() * 0.5,
      voronoi_diagram_type: VoronoiDiagramType.range,
      voronoi_nth: 0,
      voronoi_refract: 1.0 + random() * 0.5,
    },
  },

  "basic-water": {
    layers: ["multires", "refract-octaves", "reflect-octaves", "ripple"],
    settings: {
      color_space: ColorSpace.hsv,
      distrib: ValueDistribution.uniform,
      freq: random_int(7, 10),
      hue_range: 0.05 + random() * 0.05,
      hue_rotation: 0.5125 + random() * 0.025,
      lattice_drift: 1.0,
      octaves: 4,
      palette_on: false,
      reflect_range: 0.16667 + random() * 0.16667,
      refract_range: 0.25 + random() * 0.125,
      refract_y_from_offset: true,
      ripple_range: 0.005 + random() * 0.0025,
      ripple_kink: random_int(2, 4),
      ripple_freq: random_int(2, 4),
    },
  },

  "be-kind-rewind": {
    final: [vhs(), preset("crt")],
  },

  "benny-lava": {
    layers: ["basic", "posterize", "funhouse", "distressed"],
    settings: {
      distrib: ValueDistribution.column_index,
      posterize_levels: 1,
      warp_range: 1 + random() * 0.5,
    },
  },

  "berkeley": {
    layers: ["multires-ridged", "reindex-octaves", "sine-octaves", "ridge", "shadow", "grain", "saturation"],
    settings: {
      freq: random_int(12, 16),
      palette_on: false,
      reindex_range: 0.75 + random() * 0.25,
      sine_range: 2.0 + random() * 2.0,
    },
  },

  "big-data-startup": {
    layers: ["glyphic"],
    settings: {
      mask: ValueMask.script,
      hue_rotation: random(),
      hue_range: 0.0625 + random() * 0.5,
      posterize_levels: random_int(2, 4),
    },
  },

  "bit-by-bit": {
    layers: ["value-mask", "bloom", "crt"],
    settings: {
      mask: random_member([ValueMask.alphanum_binary, ValueMask.alphanum_hex, ValueMask.alphanum_numeric]),
      mask_repeat: random_int(20, 40),
    },
  },

  "bitmask": {
    layers: ["multires-low", "value-mask", "bloom"],
    settings: {
      mask: random_member(ValueMask.procedural_members()),
      mask_repeat: random_int(7, 15),
      ridges: true,
    },
  },

  "blacklight-fantasy": {
    layers: ["voronoi", "funhouse", "posterize", "sobel", "invert", "bloom", "grain", "nudge-hue", "contrast-final"],
    settings: {
      color_space: ColorSpace.rgb,
      dist_metric: random_member(DistanceMetric.absolute_members()),
      posterize_levels: 3,
      voronoi_refract: 0.5 + random() * 1.25,
      warp_octaves: random_int(1, 4),
      warp_range: random_int(0, 1) * random(),
    },
  },

  "bloom": {
    settings: {
      bloom_alpha: 0.025 + random() * 0.0125,
    },
    final: [bloom(alpha: settings.bloom_alpha)],
  },

  "blotto": {
    layers: ["basic", "random-hue", "spatter-post", "maybe-palette", "maybe-invert"],
    settings: {
      color_space: random_member(ColorSpace.color_members()),
      distrib: ValueDistribution.ones,
      spatter_post_color: false,
    },
  },

  "branemelt": {
    layers: ["multires", "sine-octaves", "reflect-octaves", "bloom", "shadow", "grain", "saturation"],
    settings: {
      color_space: ColorSpace.oklab,
      freq: random_int(6, 12),
      palette_on: false,
      reflect_range: 0.025 + random() * 0.0125,
      shadow_alpha: 0.666 + random() * 0.333,
      sine_range: random_int(48, 64),
    },
  },

  "branewaves": {
    layers: ["value-mask", "ripple", "bloom"],
    settings: {
      mask: random_member(ValueMask.grid_members()),
      mask_repeat: random_int(5, 10),
      ridges: true,
      ripple_freq: 2,
      ripple_kink: 1.5 + random() * 2,
      ripple_range: 0.15 + random() * 0.15,
      spline_order: random_member([InterpolationType.linear, InterpolationType.cosine, InterpolationType.bicubic]),
    },
  },

  "brightness-post": {
    settings: {
      brightness_post: 0.125 + random() * 0.0625,
    },
    post: [adjust_brightness(amount: settings.brightness_post)],
  },

  "brightness-final": {
    settings: {
      brightness_final: 0.125 + random() * 0.0625,
    },
    final: [adjust_brightness(amount: settings.brightness_final)],
  },

  "bringing-hexy-back": {
    layers: ["voronoi", "funhouse", "maybe-invert", "bloom"],
    settings: {
      color_space: random_member(ColorSpace.color_members()),
      dist_metric: DistanceMetric.euclidean,
      hue_range: 0.25 + random() * 0.75,
      voronoi_alpha: 0.333 + random() * 0.333,
      voronoi_diagram_type: VoronoiDiagramType.range_regions,
      voronoi_nth: 0,
      voronoi_point_distrib: coin_flip() ? PointDistribution.v_hex : PointDistribution.h_hex,
      voronoi_point_freq: random_int(4, 7) * 2,
      warp_range: 0.05 + random() * 0.25,
      warp_octaves: random_int(1, 4),
    },
    generator: {
      freq: settings.voronoi_point_freq,
    },
  },

  "broken": {
    layers: ["multires-low", "reindex-octaves", "posterize", "glowing-edges", "grain", "saturation"],
    settings: {
      color_space: ColorSpace.rgb,
      freq: random_int(3, 4),
      lattice_drift: 2,
      posterize_levels: 3,
      reindex_range: random_int(3, 4),
      speed: 0.025,
    },
  },

  "bubble-machine": {
    layers: ["basic", "posterize", "wormhole", "reverb", "outline", "maybe-invert"],
    settings: {
      corners: true,
      distrib: ValueDistribution.uniform,
      freq: random_int(3, 6) * 2,
      mask: random_member([ValueMask.h_hex, ValueMask.v_hex]),
      posterize_levels: random_int(8, 16),
      reverb_iterations: random_int(1, 3),
      reverb_octaves: random_int(3, 5),
      spline_order: random_member([InterpolationType.linear, InterpolationType.cosine, InterpolationType.bicubic]),
      wormhole_stride: 0.1 + random() * 0.05,
      wormhole_kink: 0.5 + random() * 4,
    },
  },

  "bubble-multiverse": {
    layers: ["voronoi", "refract-post", "density-map", "random-hue", "bloom", "shadow"],
    settings: {
      dist_metric: DistanceMetric.euclidean,
      refract_range: 0.125 + random() * 0.05,
      speed: 0.05,
      voronoi_alpha: 1.0,
      voronoi_diagram_type: VoronoiDiagramType.flow,
      voronoi_point_freq: 10,
      voronoi_refract: 0.625 + random() * 0.25,
    },
  },

  "carpet": {
    layers: ["worms", "grime"],
    settings: {
      worms_alpha: 0.25 + random() * 0.25,
      worms_behavior: WormBehavior.chaotic,
      worms_stride: 0.333 + random() * 0.333,
      worms_stride_deviation: 0.25,
    },
  },

  "celebrate": {
    layers: ["basic", "posterize", "distressed"],
    settings: {
      brightness_distrib: ValueDistribution.ones,
      hue_range: 1,
      posterize_levels: random_int(3, 5),
      speed: 0.025,
    },
  },

  "cell-reflect": {
    layers: ["voronoi", "reflect-post", "derivative-post", "density-map", "maybe-invert", "bloom", "grain", "saturation"],
    settings: {
      dist_metric: random_member(DistanceMetric.absolute_members()),
      palette_name: null,
      reflect_range: random_int(2, 4) * 5,
      saturation_final: 0.5 + random() * 0.25,
      voronoi_alpha: 0.333 + random() * 0.333,
      voronoi_diagram_type: VoronoiDiagramType.color_range,
      voronoi_nth: coin_flip(),
      voronoi_point_distrib: random_member([
        PointDistribution.random,
        PointDistribution.spiral,
        PointDistribution.circular,
        PointDistribution.concentric,
        PointDistribution.rotating,
      ]),
      voronoi_point_freq: random_int(2, 3),
    },
  },

  "cell-refract": {
    layers: ["voronoi", "ridge"],
    settings: {
      color_space: random_member(ColorSpace.color_members()),
      dist_metric: random_member(DistanceMetric.absolute_members()),
      ridges: true,
      voronoi_diagram_type: VoronoiDiagramType.range,
      voronoi_point_freq: random_int(3, 4),
      voronoi_refract: random_int(8, 12) * 0.5,
    },
  },

  "cell-refract-2": {
    layers: ["voronoi", "refract-post", "derivative-post", "density-map", "saturation"],
    settings: {
      dist_metric: random_member(DistanceMetric.absolute_members()),
      refract_range: random_int(1, 3) * 0.25,
      voronoi_alpha: 0.333 + random() * 0.333,
      voronoi_diagram_type: VoronoiDiagramType.color_range,
      voronoi_point_distrib: random_member([
        PointDistribution.random,
        PointDistribution.spiral,
        PointDistribution.circular,
        PointDistribution.concentric,
        PointDistribution.rotating,
      ]),
      voronoi_point_freq: random_int(2, 3),
    },
  },

  "cell-worms": {
    layers: ["multires-low", "voronoi", "worms", "density-map", "random-hue", "saturation"],
    settings: {
      freq: random_int(3, 7),
      hue_range: 0.125 + random() * 0.875,
      voronoi_alpha: 0.75,
      voronoi_point_distrib: random_member(PointDistribution, ValueMask.nonprocedural_members()),
      voronoi_point_freq: random_int(2, 4),
      worms_density: 1500,
      worms_kink: random_int(16, 32),
      worms_stride_deviation: 0,
    },
  },

  "chalky": {
    layers: ["basic", "refract-post", "octave-warp-post", "outline", "grain", "lens"],
    settings: {
      color_space: ColorSpace.oklab,
      freq: random_int(2, 3),
      octaves: random_int(2, 3),
      outline_invert: true,
      refract_range: 0.1 + random() * 0.05,
      ridges: true,
      warp_octaves: 8,
      warp_range: 0.0333 + random() * 0.016667,
    },
  },

  "chunky-knit": {
    layers: ["jorts", "random-hue", "contrast-final"],
    settings: {
      angle: random() * 360.0,
      glyph_map_alpha: 0.333 + random() * 0.16667,
      glyph_map_mask: ValueMask.waffle,
      glyph_map_zoom: 16.0,
    },
  },

  "classic-desktop": {
    layers: ["basic", "lens-warp"],
    settings: {
      hue_range: 0.333 + random() * 0.333,
      lattice_drift: random(),
    },
  },

  "cloudburst": {
    layers: ["multires", "reflect-octaves", "octave-warp-octaves", "refract-post", "invert", "grain"],
    settings: {
      color_space: ColorSpace.hsv,
      distrib: ValueDistribution.exp,
      freq: 2,
      hue_range: 0.05 - random() * 0.025,
      hue_rotation: 0.1 - random() * 0.025,
      lattice_drift: 0.75,
      palette_on: false,
      reflect_range: 0.125 + random() * 0.0625,
      refract_range: 0.1 + random() * 0.05,
      saturation_distrib: ValueDistribution.ones,
      speed: 0.075,
    },
  },

  "clouds": {
    layers: ["bloom", "grain"],
    post: [clouds()],
  },

  "concentric": {
    layers: ["wobble", "voronoi", "contrast-post", "maybe-palette"],
    settings: {
      color_space: ColorSpace.rgb,
      dist_metric: random_member(DistanceMetric.absolute_members()),
      distrib: ValueDistribution.ones,
      freq: 2,
      mask: ValueMask.h_bar,
      speed: 0.75,
      spline_order: InterpolationType.constant,
      voronoi_diagram_type: VoronoiDiagramType.range,
      voronoi_refract: random_int(8, 16),
      voronoi_point_drift: 0,
      voronoi_point_freq: random_int(1, 2),
    },
  },

  "conference": {
    layers: ["value-mask", "sobel", "maybe-rotate", "maybe-invert", "grain"],
    settings: {
      mask: ValueMask.halftone,
      mask_repeat: random_int(4, 12),
      spline_order: InterpolationType.cosine,
    },
  },

  "contrast-post": {
    settings: {
      contrast_post: 1.25 + random() * 0.25,
    },
    post: [adjust_contrast(amount: settings.contrast_post)],
  },

  "contrast-final": {
    settings: {
      contrast_final: 1.25 + random() * 0.25,
    },
    final: [adjust_contrast(amount: settings.contrast_final)],
  },

  "cool-water": {
    layers: ["basic-water", "funhouse", "bloom", "lens"],
    settings: {
      warp_range: 0.0625 + random() * 0.0625,
      warp_freq: random_int(2, 3),
    },
  },

  "corner-case": {
    layers: ["multires-ridged", "maybe-rotate", "grain", "saturation", "vignette-dark"],
    settings: {
      corners: true,
      lattice_drift: coin_flip(),
      spline_order: InterpolationType.constant,
    },
  },

  "deep-field": {
    layers: ["multires", "refract-octaves", "octave-warp-octaves", "bloom", "lens"],
    settings: {
      distrib: ValueDistribution.uniform,
      freq: random_int(8, 10),
      hue_range: 1,
      mask: ValueMask.sparser,
      mask_static: true,
      lattice_drift: 1,
      octave_blending: OctaveBlending.alpha,
      octaves: 5,
      palette_on: false,
      speed: 0.05,
      refract_range: 0.2 + random() * 0.1,
      warp_freq: 2,
      warp_signed_range: true,
    },
  },

  "deeper": {
    layers: ["multires-alpha", "funhouse", "lens", "contrast-final"],
    settings: {
      hue_range: 0.75,
      octaves: 6,
      ridges: true,
    },
  },

  "degauss": {
    final: [degauss(displacement: .06 + random() * 0.03), preset("crt")],
  },

  "density-map": {
    layers: ["grain"],
    post: [density_map(), convolve(kernel: ValueMask.conv2d_invert)],
  },

  "density-wave": {
    layers: [random_member(["basic", "symmetry"]), "reflect-post", "density-map", "invert", "bloom"],
    settings: {
      reflect_range: random_int(3, 8),
      saturation: random_int(0, 1),
    },
  },

  "derivative-octaves": {
    settings: {
      deriv_alpha: 1.0,
      dist_metric: random_member(DistanceMetric.absolute_members()),
    },
    octaves: [derivative(dist_metric: settings.dist_metric, alpha: settings.deriv_alpha)],
    post: [fxaa()],
  },

  "derivative-post": {
    settings: {
      deriv_alpha: 1.0,
      dist_metric: random_member(DistanceMetric.absolute_members()),
    },
    post: [
      derivative(dist_metric: settings.dist_metric, alpha: settings.deriv_alpha),
      fxaa(),
    ],
  },

  "dexter": {
    layers: ["spatter-final"],
    settings: {
      spatter_final_color: [
        0.35 + random() * 0.15,
        0.025 + random() * 0.0125,
        0.075 + random() * 0.0375,
      ],
    },
  },

  "different": {
    layers: ["multires", "sine-octaves", "reflect-octaves", "reindex-octaves", "funhouse", "lens"],
    settings: {
      freq: [random_int(4, 6), random_int(4, 6)],
      reflect_range: 7.5 + random() * 5.0,
      reindex_range: 0.25 + random() * 0.25,
      sine_range: random_int(7, 12),
      speed: 0.025,
      warp_range: 0.0375 * random() * 0.0375,
    },
  },

  "distressed": {
    layers: ["grain", "filthy", "saturation"],
  },

  "distance": {
    layers: ["multires", "derivative-octaves", "bloom", "shadow", "contrast-final", "maybe-rotate", "lens"],
    settings: {
      dist_metric: random_member(DistanceMetric.absolute_members()),
      distrib: ValueDistribution.exp,
      freq: [random_int(4, 5), random_int(2, 3)],
      lattice_drift: 1,
      saturation: 0.0625 + random() * 0.125,
    },
  },

  "dla": {
    layers: ["basic", "contrast-final"],
    settings: {
      dla_alpha: 0.875 + random() * 0.125,
      dla_padding: random_int(1, 8),
      dla_seed_density: 0.1 + random() * 0.05,
      dla_density: 0.2 + random() * 0.1,
    },
    post: [
      dla(
        alpha: settings.dla_alpha,
        padding: settings.dla_padding,
        seed_density: settings.dla_seed_density,
        density: settings.dla_density,
      ),
    ],
  },

  "dla-forest": {
    layers: ["dla", "reverb", "contrast-final", "bloom"],
    settings: {
      dla_padding: random_int(2, 8),
      reverb_iterations: random_int(2, 4),
    },
  },

  "domain-warp": {
    layers: ["multires-ridged", "refract-post", "vaseline", "grain", "vignette-dark", "saturation"],
    settings: {
      refract_range: 0.5 + random() * 0.5,
    },
  },

  "dropout": {
    layers: ["basic", "maybe-rotate", "derivative-post", "maybe-invert", "grain"],
    settings: {
      color_space: random_member(ColorSpace.color_members()),
      distrib: ValueDistribution.ones,
      freq: [random_int(4, 6), random_int(2, 4)],
      mask: ValueMask.dropout,
      octave_blending: OctaveBlending.reduce_max,
      octaves: random_int(4, 6),
      spline_order: InterpolationType.constant,
    },
  },

  "eat-static": {
    layers: ["basic", "be-kind-rewind", "scanline-error", "crt"],
    settings: {
      freq: 512,
      saturation: 0,
      speed: 2.0,
    },
  },

  "educational-video-film": {
    layers: ["basic", "be-kind-rewind"],
    settings: {
      color_space: ColorSpace.oklab,
      ridges: true,
    },
  },

  "electric-worms": {
    layers: ["voronoi", "worms", "density-map", "glowing-edges", "lens"],
    settings: {
      dist_metric: random_member(DistanceMetric.all()),
      freq: random_int(3, 6),
      lattice_drift: 1,
      voronoi_alpha: 0.25 + random() * 0.25,
      voronoi_diagram_type: VoronoiDiagramType.color_range,
      voronoi_nth: random_int(0, 3),
      voronoi_point_freq: random_int(3, 6),
      voronoi_point_distrib: PointDistribution.random,
      worms_alpha: 0.666 + random() * 0.333,
      worms_behavior: WormBehavior.random,
      worms_density: 1000,
      worms_duration: 1,
      worms_kink: random_int(7, 9),
      worms_stride: 1.0,
      worms_stride_deviation: 0,
      worms_quantize: coin_flip(),
    },
  },

  "emboss": {
    post: [convolve(kernel: ValueMask.conv2d_emboss)],
  },

  "emo": {
    layers: ["value-mask", "voronoi", "contrast-final", "maybe-rotate", "saturation", "tint", "lens"],
    settings: {
      contrast_final: 4.0,
      dist_metric: random_member([DistanceMetric.manhattan, DistanceMetric.chebyshev]),
      mask: ValueMask.emoji,
      spline_order: InterpolationType.cosine,
      voronoi_diagram_type: VoronoiDiagramType.range,
      voronoi_refract: 0.125 + random() * 0.25,
    },
  },

  "emu": {
    layers: ["value-mask", "voronoi", "saturation", "distressed"],
    settings: {
      dist_metric: random_member(DistanceMetric.all()),
      distrib: ValueDistribution.ones,
      mask: stash("mask", random_member(enum_range(ValueMask.emoji_00, ValueMask.emoji_26))),
      mask_repeat: 1,
      spline_order: InterpolationType.constant,
      voronoi_alpha: 1.0,
      voronoi_diagram_type: VoronoiDiagramType.range,
      voronoi_point_distrib: stash("mask"),
      voronoi_refract: 0.125 + random() * 0.125,
      voronoi_refract_y_from_offset: false,
    },
  },

  "entities": {
    layers: ["value-mask", "refract-octaves", "normalize"],
    settings: {
      hue_range: 2.0 + random() * 2.0,
      mask: ValueMask.invaders_square,
      mask_repeat: random_int(3, 4) * 2,
      refract_range: 0.1 + random() * 0.05,
      refract_signed_range: false,
      refract_y_from_offset: true,
      spline_order: InterpolationType.cosine,
    },
  },

  "entity": {
    layers: ["entities", "sobel", "invert", "bloom", "random-hue", "lens"],
    settings: {
      corners: true,
      distrib: ValueDistribution.ones,
      hue_range: 1.0 + random() * 0.5,
      mask_repeat: 1,
      refract_range: 0.025 + random() * 0.0125,
      refract_signed_range: true,
      refract_y_from_offset: false,
      speed: 0.05,
    },
  },

  "erosion-worms": {
    settings: {
      erosion_worms_alpha: 0.5 + random() * 0.5,
      erosion_worms_contraction: 0.5 + random() * 0.5,
      erosion_worms_density: random_int(25, 100),
      erosion_worms_inverse: false,
      erosion_worms_iterations: random_int(25, 100),
      erosion_worms_quantize: false,
      erosion_worms_xy_blend: 0.75 + random() * 0.25,
    },
    post: [
      erosion_worms(
        alpha: settings.erosion_worms_alpha,
        contraction: settings.erosion_worms_contraction,
        density: settings.erosion_worms_density,
        inverse: settings.erosion_worms_inverse,
        iterations: settings.erosion_worms_iterations,
        quantize: settings.erosion_worms_quantize,
        xy_blend: settings.erosion_worms_xy_blend,
      ),
      normalize(),
    ],
  },

  "escape-velocity": {
    layers: ["multires-low", "erosion-worms", "lens"],
    settings: {
      color_space: random_member(ColorSpace.color_members()),
      distrib: random_member([ValueDistribution.exp, ValueDistribution.uniform]),
      erosion_worms_contraction: 0.2 + random() * 0.1,
      erosion_worms_iterations: random_int(625, 1125),
    },
  },

  "falsetto": {
    final: [false_color()],
  },

  "fargate": {
    layers: ["serene", "contrast-post", "crt", "saturation"],
    settings: {
      brightness_distrib: ValueDistribution.uniform,
      freq: 3,
      octaves: 3,
      refract_range: 0.015 + random() * 0.0075,
      saturation_distrib: ValueDistribution.uniform,
      speed: 0 - 0.25,
      value_distrib: ValueDistribution.center_circle,
      value_freq: 3,
      value_refract_range: 0.015 + random() * 0.0075,
    },
  },

  "fast-eddies": {
    layers: ["basic", "voronoi", "worms", "contrast-final", "saturation"],
    settings: {
      dist_metric: DistanceMetric.euclidean,
      hue_range: 0.25 + random() * 0.75,
      hue_rotation: random(),
      octaves: random_int(1, 3),
      palette_on: false,
      ridges: coin_flip(),
      voronoi_alpha: 0.5 + random() * 0.5,
      voronoi_diagram_type: VoronoiDiagramType.flow,
      voronoi_point_freq: random_int(2, 6),
      voronoi_refract: 1.0,
      worms_alpha: 0.5 + random() * 0.5,
      worms_behavior: WormBehavior.chaotic,
      worms_density: 1000,
      worms_duration: 6,
      worms_kink: random_int(125, 375),
      worms_stride: 1.0,
      worms_stride_deviation: 0.0,
    },
  },

  "fibers": {
    final: [fibers()],
  },

  "figments": {
    layers: ["multires-low", "voronoi", "funhouse", "wormhole", "bloom", "contrast-final", "lens"],
    settings: {
      freq: 2,
      hue_range: 2,
      lattice_drift: 1,
      speed: 0.025,
      voronoi_diagram_type: VoronoiDiagramType.flow,
      voronoi_refract: 0.333 + random() * 0.333,
      wormhole_stride: 0.02 + random() * 0.01,
      wormhole_kink: 4,
    },
  },

  "filthy": {
    layers: ["grime", "scratches", "stray-hair"],
  },

  "fireball": {
    layers: ["basic", "periodic-refract", "refract-post", "refract-post", "bloom", "lens", "contrast-final"],
    settings: {
      contrast_final: 2.5,
      distrib: ValueDistribution.center_circle,
      hue_rotation: 0.925,
      freq: 1,
      refract_range: 0.025 + random() * 0.0125,
      refract_y_from_offset: false,
      value_distrib: ValueDistribution.center_circle,
      value_freq: 1,
      value_refract_range: 0.05 + random() * 0.025,
      speed: 0.05,
    },
  },

  "financial-district": {
    layers: ["voronoi", "bloom", "contrast-final", "saturation"],
    settings: {
      dist_metric: DistanceMetric.manhattan,
      voronoi_diagram_type: VoronoiDiagramType.range_regions,
      voronoi_point_distrib: PointDistribution.random,
      voronoi_nth: random_int(1, 3),
      voronoi_point_freq: 2,
    },
  },

  "fossil-hunt": {
    layers: ["voronoi", "refract-octaves", "posterize-outline", "grain", "saturation"],
    settings: {
      freq: random_int(3, 5),
      lattice_drift: 1.0,
      posterize_levels: random_int(3, 5),
      refract_range: random_int(2, 4) * 0.5,
      refract_y_from_offset: true,
      voronoi_alpha: 0.5,
      voronoi_diagram_type: VoronoiDiagramType.color_range,
      voronoi_point_freq: 10,
    },
  },

  "fractal-forms": {
    layers: ["fractal-seed"],
    settings: {
      worms_kink: random_int(256, 512),
    },
  },

  "fractal-seed": {
    layers: ["multires-low", "worms", "density-map", "random-hue", "bloom", "shadow", "contrast-final", "saturation", "aberration"],
    settings: {
      freq: random_int(2, 3),
      hue_range: 1.0 + random() * 3.0,
      ridges: coin_flip(),
      speed: 0.05,
      palette_on: false,
      worms_behavior: random_member([WormBehavior.chaotic, WormBehavior.random]),
      worms_alpha: 0.9 + random() * 0.1,
      worms_density: random_int(750, 1250),
      worms_duration: random_int(2, 3),
      worms_kink: 1.0,
      worms_stride: 1.0,
      worms_stride_deviation: 0.0,
    },
  },

  "fractal-smoke": {
    layers: ["fractal-seed"],
    settings: {
      worms_behavior: WormBehavior.random,
      worms_stride: random_int(96, 192),
    },
  },

  "fractile": {
    layers: ["symmetry", "voronoi", "reverb", "contrast-post", "palette", "random-hue", "maybe-rotate", "lens"],
    settings: {
      dist_metric: random_member(DistanceMetric.absolute_members()),
      reverb_iterations: random_int(2, 4),
      reverb_octaves: random_int(2, 4),
      voronoi_alpha: 0.5 + random() * 0.5,
      voronoi_diagram_type: VoronoiDiagramType.color_range,
      voronoi_nth: random_int(0, 2),
      voronoi_point_distrib: random_member(PointDistribution.grid_members()),
      voronoi_point_freq: random_int(2, 3),
    },
  },

  "fundamentals": {
    layers: ["voronoi", "derivative-post", "density-map", "grain", "saturation"],
    settings: {
      dist_metric: random_member([DistanceMetric.manhattan, DistanceMetric.chebyshev]),
      freq: random_int(3, 5),
      voronoi_diagram_type: VoronoiDiagramType.color_range,
      voronoi_nth: random_int(3, 5),
      voronoi_point_freq: random_int(3, 5),
      voronoi_refract: 0.125 + random() * 0.0625,
    },
  },

  "funhouse": {
    settings: {
      warp_freq: [random_int(2, 4), random_int(2, 4)],
      warp_octaves: random_int(1, 4),
      warp_range: 0.25 + random() * 0.125,
      warp_signed_range: false,
      warp_spline_order: InterpolationType.bicubic,
    },
    post: [
      warp(
        displacement: settings.warp_range,
        freq: settings.warp_freq,
        octaves: settings.warp_octaves,
        signed_range: settings.warp_signed_range,
        spline_order: settings.warp_spline_order,
      ),
    ],
  },

  "funky-glyphs": {
    layers: ["value-mask", "refract-post", "contrast-final", "maybe-rotate", "saturation", "lens", "grain"],
    settings: {
      distrib: random_member([ValueDistribution.ones, ValueDistribution.uniform]),
      mask: random_member(ValueMask.glyph_members()),
      mask_repeat: random_int(1, 6),
      octaves: random_int(1, 2),
      refract_range: 0.125 + random() * 0.125,
      refract_signed_range: false,
      refract_y_from_offset: true,
      spline_order: random_member([InterpolationType.linear, InterpolationType.cosine, InterpolationType.bicubic]),
    },
  },

  "galalaga": {
    layers: ["value-mask", "contrast-final", "glitchin-out"],
    settings: {
      distrib: ValueDistribution.uniform,
      hue_range: random() * 2.5,
      mask: ValueMask.invaders_square,
      mask_repeat: 4,
      spline_order: InterpolationType.constant,
    },
    post: [
      glyph_map(colorize: true, mask: ValueMask.invaders_square, zoom: 32.0),
      glyph_map(colorize: true, mask: random_member([ValueMask.invaders_square, ValueMask.rgb]), zoom: 4.0),
      normalize(),
    ],
  },

  "game-show": {
    layers: ["basic", "maybe-rotate", "posterize", "be-kind-rewind"],
    settings: {
      freq: random_int(8, 16) * 2,
      mask: random_member([ValueMask.h_tri, ValueMask.v_tri]),
      posterize_levels: random_int(2, 5),
      spline_order: InterpolationType.cosine,
    },
  },

  "glacial": {
    layers: ["fractal-smoke"],
    settings: {
      worms_quantize: true,
    },
  },

  "glitchin-out": {
    layers: ["corrupt"],
    final: [glitch(), preset("crt"), preset("bloom")],
  },

  "globules": {
    layers: ["multires-low", "reflect-octaves", "density-map", "shadow", "lens"],
    settings: {
      distrib: ValueDistribution.ones,
      freq: random_int(3, 6),
      hue_range: 0.25 + random() * 0.5,
      lattice_drift: 1,
      mask: ValueMask.sparse,
      mask_static: true,
      octaves: random_int(3, 6),
      palette_on: false,
      reflect_range: 2.5,
      saturation: 0.175 + random() * 0.175,
      speed: 0.125,
    },
  },

  "glom": {
    layers: ["basic", "refract-octaves", "reflect-octaves", "refract-post", "reflect-post", "funhouse", "bloom", "shadow", "contrast-post", "lens"],
    settings: {
      distrib: ValueDistribution.uniform,
      freq: [2, 2],
      hue_range: 0.25 + random() * 0.125,
      lattice_drift: 1,
      octaves: 2,
      reflect_range: 0.625 + random() * 0.375,
      refract_range: 0.333 + random() * 0.16667,
      refract_signed_range: false,
      refract_y_from_offset: true,
      speed: 0.025,
      warp_range: 0.0625 + random() * 0.030625,
      warp_octaves: 1,
    },
  },

  "glowing-edges": {
    final: [glowing_edges()],
  },

  "glyph-map": {
    layers: ["basic"],
    settings: {
      glyph_map_alpha: 1.0,
      glyph_map_colorize: coin_flip(),
      glyph_map_spline_order: InterpolationType.constant,
      glyph_map_mask: random_member(masks.square_masks()),
      glyph_map_zoom: random_int(6, 10),
    },
    post: [
      glyph_map(
        alpha: settings.glyph_map_alpha,
        colorize: settings.glyph_map_colorize,
        mask: settings.glyph_map_mask,
        spline_order: settings.glyph_map_spline_order,
        zoom: settings.glyph_map_zoom,
      ),
    ],
  },

  "glyphic": {
    layers: ["value-mask", "posterize", "palette", "saturation", "maybe-rotate", "maybe-invert", "distressed"],
    settings: {
      corners: true,
      mask: random_member(ValueMask.procedural_members()),
      octave_blending: OctaveBlending.reduce_max,
      octaves: random_int(3, 5),
      posterize_levels: 1,
      saturation: 0,
      spline_order: InterpolationType.cosine,
    },
    generator: {
      freq: masks.mask_shape(settings.mask),
    },
  },

  "grain": {
    unique: true,
    settings: {
      grain_alpha: 0.0333 + random() * 0.01666,
      grain_brightness: 0.0125 + random() * 0.00625,
      grain_contrast: 1.025 + random() * 0.0125,
    },
    final: [
      grain(alpha: settings.grain_alpha),
      preset("brightness-final", {brightness_final: settings.grain_brightness}),
      preset("contrast-final", {contrast_final: settings.grain_contrast}),
    ],
  },

  "graph-paper": {
    layers: ["wobble", "voronoi", "derivative-post", "maybe-rotate", "lens", "crt", "bloom", "contrast-final"],
    settings: {
      color_space: ColorSpace.rgb,
      corners: true,
      distrib: ValueDistribution.ones,
      dist_metric: DistanceMetric.euclidean,
      freq: random_int(3, 4) * 2,
      mask: ValueMask.chess,
      spline_order: InterpolationType.constant,
      voronoi_alpha: 0.5 + random() * 0.25,
      voronoi_refract: 0.75 + random() * 0.375,
      voronoi_refract_y_from_offset: true,
      voronoi_diagram_type: VoronoiDiagramType.flow,
    },
  },

  "grass": {
    layers: ["multires", "worms", "grain"],
    settings: {
      color_space: ColorSpace.hsv,
      freq: random_int(6, 12),
      hue_rotation: 0.25 + random() * 0.05,
      lattice_drift: 1,
      palette_on: false,
      saturation: 0.625 + random() * 0.25,
      worms_behavior: random_member([WormBehavior.chaotic, WormBehavior.meandering]),
      worms_alpha: 0.9,
      worms_density: 50 + random() * 25,
      worms_drunkenness: 0.125,
      worms_duration: 1.125,
      worms_stride: 0.875,
      worms_stride_deviation: 0.125,
      worms_kink: 0.125 + random() * 0.5,
    },
  },

  "grayscale": {
    final: [adjust_saturation(amount: 0)],
  },

  "griddy": {
    layers: ["basic", "sobel", "invert", "bloom"],
    settings: {
      freq: random_int(3, 9),
      mask: ValueMask.chess,
      octaves: random_int(3, 8),
      spline_order: InterpolationType.constant,
    },
  },

  "grime": {
    final: [grime()],
  },

  "groove-is-stored-in-the-heart": {
    layers: ["basic", "posterize", "ripple", "distressed"],
    settings: {
      distrib: ValueDistribution.column_index,
      posterize_levels: random_int(1, 2),
      ripple_range: 0.75 + random() * 0.375,
    },
  },

  "halt-catch-fire": {
    layers: ["multires-low", "pixel-sort", "maybe-rotate", "glitchin-out"],
    settings: {
      freq: 2,
      hue_range: 0.05,
      lattice_drift: 1,
      spline_order: InterpolationType.constant,
    },
  },

  "hearts": {
    layers: ["value-mask", "skew", "posterize", "crt"],
    settings: {
      distrib: ValueDistribution.ones,
      hue_distrib: coin_flip() ? null : random_member([ValueDistribution.column_index, ValueDistribution.row_index]),
      hue_rotation: 0.925,
      mask: ValueMask.mcpaint_19,
      mask_repeat: random_int(8, 12),
      posterize_levels: random_int(1, 2),
    },
  },

  "hotel-carpet": {
    layers: ["basic", "ripple", "carpet", "grain"],
    settings: {
      ripple_kink: 0.5 + random() * 0.25,
      ripple_range: 0.666 + random() * 0.333,
      spline_order: InterpolationType.constant,
    },
  },

  "hsv-gradient": {
    layers: ["basic", "maybe-rotate", "grain", "saturation"],
    settings: {
      color_space: ColorSpace.hsv,
      hue_range: 0.5 + random() * 2.0,
      lattice_drift: 1.0,
      palette_on: false,
    },
  },

  "kaleido": {
    layers: ["voronoi-refract", "wobble"],
    settings: {
      color_space: ColorSpace.hsv,
      freq: random_int(8, 12),
      hue_range: 0.5 + random() * 2.5,
      kaleido_point_corners: true,
      kaleido_point_distrib: PointDistribution.random,
      kaleido_point_freq: 1,
      kaleido_sdf_sides: random_int(0, 10),
      kaleido_sides: random_int(3, 16),
      kaleido_blend_edges: false,
      palette_on: false,
      speed: 0.125,
      voronoi_point_freq: random_int(8, 12),
    },
    post: [
      kaleido(
        blend_edges: settings.kaleido_blend_edges,
        point_corners: settings.kaleido_point_corners,
        point_distrib: settings.kaleido_point_distrib,
        point_freq: settings.kaleido_point_freq,
        sdf_sides: settings.kaleido_sdf_sides,
        sides: settings.kaleido_sides,
      ),
    ],
  },

  "knotty-clouds": {
    layers: ["basic", "voronoi", "worms"],
    settings: {
      voronoi_alpha: 0.125 + random() * 0.25,
      voronoi_diagram_type: VoronoiDiagramType.color_range,
      voronoi_point_freq: random_int(6, 10),
      worms_alpha: 0.666 + random() * 0.333,
      worms_behavior: WormBehavior.obedient,
      worms_density: 1000,
      worms_duration: 1,
      worms_kink: 4,
    },
  },

  "later": {
    layers: ["value-mask", "multires", "wobble", "voronoi", "funhouse", "glowing-edges", "crt", "vignette-dark"],
    settings: {
      dist_metric: DistanceMetric.euclidean,
      freq: random_int(2, 5),
      mask: random_member(ValueMask.procedural_members()),
      spline_order: InterpolationType.constant,
      voronoi_diagram_type: VoronoiDiagramType.flow,
      voronoi_point_distrib: PointDistribution.random,
      voronoi_point_freq: random_int(3, 6),
      voronoi_refract: 1.0 + random() * 0.5,
      warp_freq: random_int(2, 4),
      warp_spline_order: InterpolationType.bicubic,
      warp_octaves: 2,
      warp_range: 0.05 + random() * 0.025,
    },
  },

  "lattice-noise": {
    layers: ["basic", "derivative-octaves", "derivative-post", "density-map", "shadow", "grain", "saturation", "vignette-dark"],
    settings: {
      dist_metric: random_member(DistanceMetric.absolute_members()),
      freq: random_int(2, 5),
      lattice_drift: 1.0,
      octaves: random_int(2, 3),
      ridges: coin_flip(),
    },
  },

  "lcd": {
    layers: ["value-mask", "invert", "skew", "shadow", "vignette-bright", "grain"],
    settings: {
      mask: random_member([ValueMask.lcd, ValueMask.lcd_binary]),
      mask_repeat: random_int(8, 12),
      saturation: 0.0,
    },
  },

  "lens": {
    layers: ["lens-distortion", "aberration", "vaseline", "tint", "vignette-dark"],
    settings: {
      lens_brightness: 0.05 + random() * 0.025,
      lens_contrast: 1.05 + random() * 0.025,
    },
    final: [
      preset("brightness-final", {brightness_final: settings.lens_brightness}),
      preset("contrast-final", {contrast_final: settings.lens_contrast}),
    ],
  },

  "lens-distortion": {
    final: [
      lens_distortion(displacement: (0.125 + random() * 0.0625) * (coin_flip() ? 1 : 0 - 1)),
    ],
  },

  "lens-warp": {
    post: [
      lens_warp(displacement: 0.125 + random() * 0.0625),
      lens_distortion(displacement: 0.25 + random() * 0.125 * (coin_flip() ? 1 : 0 - 1)),
    ],
  },

  "light-leak": {
    layers: ["vignette-bright"],
    settings: {
      light_leak_alpha: 0.25 + random() * 0.125,
    },
    final: [
      light_leak(alpha: settings.light_leak_alpha),
    ],
  },

  "look-up": {
    layers: ["multires-alpha", "brightness-post", "contrast-post", "contrast-final", "saturation", "lens", "bloom"],
    settings: {
      brightness_post: 0 - 0.075,
      color_space: ColorSpace.hsv,
      contrast_final: 1.5,
      distrib: ValueDistribution.exp,
      freq: random_int(30, 40),
      hue_range: 0.333 + random() * 0.333,
      lattice_drift: 0,
      mask: ValueMask.sparsest,
      octaves: 10,
      ridges: true,
      saturation: 0.5,
      speed: 0.025,
    },
  },

  "low-poly": {
    settings: {
      lowpoly_distrib: random_member(PointDistribution.circular_members()),
      lowpoly_freq: random_int(10, 20),
    },
    post: [
      lowpoly(
        distrib: settings.lowpoly_distrib,
        freq: settings.lowpoly_freq,
      ),
    ],
  },

  "low-poly-regions": {
    layers: ["voronoi", "low-poly"],
    settings: {
      voronoi_diagram_type: VoronoiDiagramType.color_regions,
      voronoi_point_freq: random_int(2, 3),
    },
  },

  "lsd": {
    layers: ["basic", "refract-post", "invert", "random-hue", "lens", "grain"],
    settings: {
      brightness_distrib: ValueDistribution.ones,
      freq: random_int(3, 4),
      hue_range: random_int(3, 4),
      speed: 0.025,
    },
  },

  "magic-smoke": {
    layers: ["multires", "worms", "lens"],
    settings: {
      octaves: random_int(2, 3),
      worms_alpha: 1,
      worms_behavior: random_member([WormBehavior.obedient, WormBehavior.crosshatch]),
      worms_density: 750,
      worms_duration: 0.25,
      worms_kink: random_int(1, 3),
      worms_stride: random_int(64, 256),
    },
  },

  "maybe-derivative-post": {
    post: random_member([[], [preset("derivative-post")]]),
  },

  "maybe-invert": {
    post: random_member([[], [preset("invert")]]),
  },

  "maybe-palette": {
    settings: {
      palette_alpha: 0.5 + random() * 0.5,
      palette_name: random_member(PALETTES),
    },
    post: random_member([
      [],
      [],
      [],
      [],
      [],
      [palette(name: settings.palette_name, alpha: settings.palette_alpha)],
      [palette(name: settings.palette_name, alpha: settings.palette_alpha)],
      [palette(name: settings.palette_name, alpha: settings.palette_alpha)],
    ]),
  },

  "maybe-rotate": {
    settings: {
      angle: random() * 360.0,
    },
    post: random_member([[], [rotate(angle: settings.angle)]]),
  },

  "maybe-skew": {
    final: random_member([[], [preset("skew")]]),
  },

  "mcpaint": {
    layers: ["glyph-map", "skew", "grain", "vignette-dark", "brightness-final", "contrast-final", "saturation"],
    settings: {
      corners: true,
      freq: random_int(2, 8),
      glyph_map_colorize: false,
      glyph_map_mask: ValueMask.mcpaint,
      glyph_map_zoom: random_int(2, 4),
      spline_order: InterpolationType.cosine,
    },
  },

  "moire-than-a-feeling": {
    layers: ["basic", "wormhole", "density-map", "invert", "contrast-post"],
    settings: {
      octaves: random_int(1, 2),
      saturation: 0,
      wormhole_kink: 128,
      wormhole_stride: 0.0005,
    },
  },

  "molten-glass": {
    layers: ["basic", "sine-octaves", "octave-warp-post", "brightness-post", "contrast-post", "bloom", "shadow", "normalize", "lens"],
    settings: {
      hue_range: random() * 3.0,
    },
  },

  "multires": {
    layers: ["basic"],
    settings: {
      octaves: random_int(6, 8),
    },
  },

  "multires-alpha": {
    layers: ["multires"],
    settings: {
      distrib: ValueDistribution.exp,
      lattice_drift: 1,
      octave_blending: OctaveBlending.alpha,
      octaves: 5,
      palette_on: false,
    },
  },

  "multires-low": {
    layers: ["basic"],
    settings: {
      octaves: random_int(2, 4),
    },
  },

  "multires-ridged": {
    layers: ["multires"],
    settings: {
      lattice_drift: random(),
      ridges: true,
    },
  },

  "muppet-fur": {
    layers: ["basic", "worms", "rotate", "bloom", "lens"],
    settings: {
      color_space: random_member([ColorSpace.oklab, ColorSpace.hsv]),
      freq: random_int(2, 3),
      hue_range: random() * 0.25,
      hue_rotation: random(),
      lattice_drift: random() * 0.333,
      palette_on: false,
      worms_alpha: 0.875 + random() * 0.125,
      worms_behavior: WormBehavior.unruly,
      worms_density: random_int(500, 1250),
      worms_drunkenness: random() * 0.025,
      worms_duration: 2.0 + random() * 1.0,
      worms_stride: 1.0,
      worms_stride_deviation: 0.0,
    },
  }

}
