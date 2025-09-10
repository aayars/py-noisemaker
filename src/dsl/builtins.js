import * as constants from '../constants.js';

export const surfaces = Object.freeze({
  synth1: 'synth1',
  synth2: 'synth2',
  mixer: 'mixer',
  post1: 'post1',
  post2: 'post2',
  post3: 'post3',
  final: 'final',
});

export const operations = Object.create(null);

export const enums = constants;

export const defaultContext = {
  surfaces,
  operations,
  enums,
};
