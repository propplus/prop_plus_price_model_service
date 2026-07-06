import assert from 'node:assert/strict';
import test from 'node:test';

import { buildFeatureTensor } from '../src/features.js';

test('buildFeatureTensor reconstructs district_rank from district_code mapping', () => {
  const tensor = buildFeatureTensor(
    {
      property_type_id: 3,
      district_code: 1001,
      area_sqm: 400,
    },
    ['area_sqm', 'district_rank', 'ptype_3'],
    { '1001': 7 }
  );

  assert.deepEqual(Array.from(tensor), [400, 7, 1]);
});

test('buildFeatureTensor uses explicit unknown rank for unmapped districts', () => {
  const tensor = buildFeatureTensor(
    {
      district_code: 9999,
    },
    ['district_rank'],
    { '1001': 7 }
  );

  assert.deepEqual(Array.from(tensor), [0]);
});
