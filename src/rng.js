let _seed = 0x12345678;

export function setSeed(s) {
  _seed = (s >>> 0) || 0;
}

export function getSeed() {
  return _seed >>> 0;
}

export function random() {
  _seed = (_seed * 1664525 + 1013904223) >>> 0;
  return _seed / 0x100000000;
}
