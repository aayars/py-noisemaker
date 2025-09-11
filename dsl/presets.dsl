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
  }
}
