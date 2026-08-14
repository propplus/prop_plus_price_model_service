import assert from 'node:assert/strict';
import test from 'node:test';

import {
  UNSEEN_RANK,
  assertSchemaSupported,
  buildFeatureTensor,
  rankArtifactsFor,
} from '../src/features.js';

// ---------------------------------------------------------------------------
// Cross-repo contract with prop_plus_price_trainer (FEAT-004).
// The trainer builds its matrix in train.py::_build_features(); every
// expectation below quotes the line of train.py it mirrors. When the trainer
// changes, these tests must change with it — that is the point of them.
//
// Values are chosen to be exactly representable in float32 (x.0 / x.5 / x.25)
// so equality assertions stay honest and don't hide precision drift.
// ---------------------------------------------------------------------------

// --- missing values: NaN, not a magic number -------------------------------
// train.py::_explode_landmarks — "Missing distances stay NaN"; the 99999
// sentinel was removed in 2bd4cdb. 99999 means "genuinely 100 km away" to the
// model, which is a different statement from "we don't know".

test('missing landmark distance is NaN, not the old 99999 sentinel', () => {
  const tensor = buildFeatureTensor({ landmarks: {} }, ['dist_bts_m']);
  assert.ok(Number.isNaN(tensor[0]), `expected NaN, got ${tensor[0]}`);
});

test('present landmark distance is passed through, with or without the _m key suffix', () => {
  assert.deepEqual(
    Array.from(buildFeatureTensor({ landmarks: { bts: 250.5 } }, ['dist_bts_m'])),
    [250.5]
  );
  assert.deepEqual(
    Array.from(buildFeatureTensor({ landmarks: { bts_m: 250.5 } }, ['dist_bts_m'])),
    [250.5]
  );
});

// train.py::_build_features — pd.to_numeric(..., errors="coerce") yields NaN,
// so an absent bedroom count must not arrive as "a 0-bedroom home".
test('absent numeric fields are NaN, not 0', () => {
  const tensor = buildFeatureTensor({}, [
    'area_sqm',
    'bedrooms_count',
    'bathrooms_count',
    'completion_year',
  ]);
  for (const [i, name] of ['area_sqm', 'bedrooms', 'bathrooms', 'completion_year'].entries()) {
    assert.ok(Number.isNaN(tensor[i]), `${name}: expected NaN, got ${tensor[i]}`);
  }
});

test('unparseable numeric fields are NaN, not 0', () => {
  const tensor = buildFeatureTensor({ area_sqm: 'ห้าสิบตารางเมตร' }, ['area_sqm']);
  assert.ok(Number.isNaN(tensor[0]));
});

// train.py::_build_features — is_off_plan/.fillna(False).astype(int): these two
// are the deliberate exception. Absent means False, not unknown.
test('absent booleans are 0 — NOT NaN (trainer does fillna(False))', () => {
  assert.deepEqual(
    Array.from(buildFeatureTensor({}, ['is_off_plan', 'is_price_negotiable'])),
    [0, 0]
  );
  assert.deepEqual(
    Array.from(
      buildFeatureTensor({ is_off_plan: true, is_price_negotiable: true }, [
        'is_off_plan',
        'is_price_negotiable',
      ])
    ),
    [1, 1]
  );
});

// --- lat/lng: new in FEAT-004, previously fell through to a silent 0 --------
// A silent 0/0 puts every listing in the Gulf of Guinea.

test('lat/lng are passed through', () => {
  assert.deepEqual(
    Array.from(buildFeatureTensor({ lat: 13.5, lng: 100.25 }, ['lat', 'lng'])),
    [13.5, 100.25]
  );
});

test('absent lat/lng are NaN, not 0 (0,0 is a real place in the ocean)', () => {
  const tensor = buildFeatureTensor({}, ['lat', 'lng']);
  assert.ok(Number.isNaN(tensor[0]) && Number.isNaN(tensor[1]));
});

// --- rank encoders ---------------------------------------------------------
// train.py::_smoothed_rank — keys unseen in the train fraction map to rank 0
// "at train and serving time alike", and df[key].map(rank).fillna(0) makes a
// null key rank 0 too. So here 0 is correct and NaN would be wrong.

test('all three rank encoders resolve from their own map', () => {
  const tensor = buildFeatureTensor(
    {
      district_code: 1001,
      project_id: '3f6b9c1e-0000-4000-8000-000000000001',
      developer_name: 'แสนสิริ',
    },
    ['district_rank', 'project_rank', 'developer_rank'],
    {
      district_rank: { 1001: 7 },
      project_rank: { '3f6b9c1e-0000-4000-8000-000000000001': 12 },
      developer_rank: { แสนสิริ: 3 },
    }
  );
  assert.deepEqual(Array.from(tensor), [7, 12, 3]);
});

test('rank map keys are compared as strings (trainer writes JSON object keys)', () => {
  // train.py::_district_code_key stringifies ints; JSON has no numeric keys.
  const tensor = buildFeatureTensor({ district_code: 1001 }, ['district_rank'], {
    district_rank: { 1001: 7 },
  });
  assert.deepEqual(Array.from(tensor), [7]);
});

test('unseen and absent keys get UNSEEN_RANK (0), not NaN', () => {
  assert.deepEqual(
    Array.from(
      buildFeatureTensor({ district_code: 9999 }, ['district_rank'], {
        district_rank: { 1001: 7 },
      })
    ),
    [UNSEEN_RANK]
  );
  // developer_name is not yet sent by propplus-backend — must degrade to
  // "unseen developer", never to NaN or to a crash.
  assert.deepEqual(
    Array.from(buildFeatureTensor({}, ['developer_rank'], { developer_rank: { x: 4 } })),
    [UNSEEN_RANK]
  );
});

test('a caller-supplied district_rank still wins over the map', () => {
  const tensor = buildFeatureTensor({ district_rank: 5, district_code: 1001 }, ['district_rank'], {
    district_rank: { 1001: 7 },
  });
  assert.deepEqual(Array.from(tensor), [5]);
});

test('an empty rank map is tolerated (rank artifacts absent)', () => {
  assert.deepEqual(Array.from(buildFeatureTensor({ district_code: 1001 }, ['district_rank'])), [0]);
});

// --- categorical -----------------------------------------------------------
// train.py: pd.get_dummies(..., prefix=["ptype","tenure","fquota"], dummy_na=True)

test('one-hot columns match on value, and _nan fires only when the value is absent', () => {
  const schema = ['ptype_3', 'ptype_7', 'ptype_nan', 'tenure_freehold', 'fquota_nan'];
  assert.deepEqual(
    Array.from(buildFeatureTensor({ property_type_id: 3, tenure: 'freehold' }, schema)),
    [1, 0, 0, 1, 1]
  );
  assert.deepEqual(Array.from(buildFeatureTensor({ tenure: 'freehold' }, schema)), [0, 0, 1, 1, 1]);
});

test('view_* is multi-hot with 0 for absent (trainer fillna(0))', () => {
  assert.deepEqual(
    Array.from(
      buildFeatureTensor({ view_type: ['city', 'river'] }, ['view_city', 'view_river', 'view_pool'])
    ),
    [1, 1, 0]
  );
  assert.deepEqual(Array.from(buildFeatureTensor({}, ['view_city'])), [0]);
});

// --- fail loud on an unknown column ----------------------------------------
// This is the guard for the whole class of bug FEAT-004 exposed: lat, lng,
// project_rank and developer_rank were all silently encoded as 0 for months
// because the old `let v = 0` default swallowed anything it didn't recognise.
// Refusing to serve is the correct failure here — a wrong price looks right.

test('an unrecognised schema column throws instead of silently encoding 0', () => {
  assert.throws(() => buildFeatureTensor({}, ['area_sqm', 'floor_level']), /floor_level/);
});

test('assertSchemaSupported names every unknown column at once', () => {
  assert.doesNotThrow(() =>
    assertSchemaSupported(['area_sqm', 'lat', 'dist_bts_m', 'ptype_1', 'view_city'])
  );
  assert.throws(() => assertSchemaSupported(['area_sqm', 'floor_level', 'noise_db']), (err) => {
    assert.match(err.message, /floor_level/);
    assert.match(err.message, /noise_db/);
    return true;
  });
});

test('dist_ and view_ prefixes are not matched loosely', () => {
  assert.throws(() => assertSchemaSupported(['dist_bts']), /dist_bts/); // no _m suffix
  assert.throws(() => assertSchemaSupported(['dist_m']), /dist_m/); // empty landmark kind
});

// --- artifacts to download -------------------------------------------------

test('rankArtifactsFor asks only for the rank files the schema actually uses', () => {
  assert.deepEqual(rankArtifactsFor(['area_sqm', 'district_rank']), {
    district_rank: 'district_rank.json',
  });
  assert.deepEqual(rankArtifactsFor(['district_rank', 'project_rank', 'developer_rank']), {
    district_rank: 'district_rank.json',
    project_rank: 'project_rank.json',
    developer_rank: 'developer_rank.json',
  });
  assert.deepEqual(rankArtifactsFor(['area_sqm']), {});
});

// --- ordering --------------------------------------------------------------

test('values follow the schema order, not the input order', () => {
  // train.py sorts columns: X.reindex(sorted(X.columns), axis=1)
  const tensor = buildFeatureTensor({ area_sqm: 100.5, lat: 13.5 }, ['lat', 'area_sqm']);
  assert.deepEqual(Array.from(tensor), [13.5, 100.5]);
});

test('an empty schema yields an empty tensor', () => {
  assert.equal(buildFeatureTensor({}, []).length, 0);
});
