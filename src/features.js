// Cross-repo contract with `prop_plus_price_trainer` (FEAT-004).
//
// `feature_schema.json` is an ordered column list produced by the trainer's
// `train.py::_build_features()`. Everything in this file exists to reproduce
// that function's encoding exactly at serving time. **Change both repos
// together** — a mismatch here does not crash and does not log; it just
// returns a wrong price that looks perfectly reasonable.
//
// Missing values are `NaN`, never a sentinel and never 0. LightGBM treats NaN
// as "unknown" and routes it down its own learned branch; 99999 or 0 are
// statements of fact ("100 km from the BTS", "at 0°N 0°E in the ocean") and
// the model believes them. The 99999 landmark sentinel was removed from the
// trainer in `2bd4cdb` (2026-08-06).
//
// The two deliberate exceptions, both mirroring the trainer:
//   - `is_off_plan` / `is_price_negotiable` → 0 when absent (`fillna(False)`)
//   - `*_rank` → 0 when the key is unseen or absent (`_smoothed_rank` maps
//     unseen keys to rank 0 "at train and serving time alike")

/** Rank-encoded column → the request field whose value keys its map. */
export const RANK_COLUMNS = Object.freeze({
  district_rank: 'district_code',
  project_rank: 'project_id',
  developer_rank: 'developer_name',
});

/** Rank-encoded column → the R2 artifact the trainer writes beside the model. */
export const RANK_ARTIFACT_FILES = Object.freeze({
  district_rank: 'district_rank.json',
  project_rank: 'project_rank.json',
  developer_rank: 'developer_rank.json',
});

/** Rank for a key the trainer never saw in its train fraction. */
export const UNSEEN_RANK = 0;

const NUMERIC_COLUMNS = new Set([
  'area_sqm',
  'bedrooms_count',
  'bathrooms_count',
  'completion_year',
  'lat',
  'lng',
]);

const BOOLEAN_COLUMNS = new Set(['is_off_plan', 'is_price_negotiable']);

/** `pd.get_dummies(prefix=[...], dummy_na=True)` → prefix maps to a request field. */
const ONEHOT_PREFIXES = Object.freeze({
  ptype_: 'property_type_id',
  tenure_: 'tenure',
  fquota_: 'foreign_quota_status',
});

const LANDMARK_RE = /^dist_(.+)_m$/;
const VIEW_RE = /^view_(.+)$/;

/**
 * Decide how one schema column is encoded. Returns `null` for a column this
 * service does not know — the caller must refuse to serve rather than guess.
 */
export function classifyColumn(col) {
  if (NUMERIC_COLUMNS.has(col)) return { kind: 'numeric' };
  if (BOOLEAN_COLUMNS.has(col)) return { kind: 'boolean' };
  if (Object.hasOwn(RANK_COLUMNS, col)) return { kind: 'rank', keyField: RANK_COLUMNS[col] };

  for (const [prefix, field] of Object.entries(ONEHOT_PREFIXES)) {
    if (col.startsWith(prefix) && col.length > prefix.length) {
      return { kind: 'onehot', field, suffix: col.slice(prefix.length) };
    }
  }

  const landmark = LANDMARK_RE.exec(col);
  if (landmark) return { kind: 'landmark', landmarkKind: landmark[1] };

  const view = VIEW_RE.exec(col);
  if (view) return { kind: 'view', value: view[1] };

  return null;
}

/**
 * Fail loudly when the trainer's schema contains a column this service cannot
 * encode. Before FEAT-004 the encoder fell through to `let v = 0`, so `lat`,
 * `lng`, `project_rank` and `developer_rank` were silently encoded as 0 from
 * the day the trainer added them. A refused request is recoverable; a
 * confidently wrong price is not.
 */
export function assertSchemaSupported(featureSchema) {
  const unknown = featureSchema.filter((col) => classifyColumn(col) === null);
  if (unknown.length === 0) return;
  throw new Error(
    `feature_schema.json has ${unknown.length} column(s) this service cannot encode: ` +
      `${unknown.join(', ')} — src/features.js is out of sync with the trainer ` +
      `(prop_plus_price_trainer::_build_features). Refusing to serve rather than encoding them as 0.`
  );
}

/** The rank artifacts that must be downloaded for a given schema (may be empty). */
export function rankArtifactsFor(featureSchema) {
  const wanted = {};
  for (const [col, filename] of Object.entries(RANK_ARTIFACT_FILES)) {
    if (featureSchema.includes(col)) wanted[col] = filename;
  }
  return wanted;
}

/**
 * @param features   request payload from propplus-backend
 * @param featureSchema ordered column names from the trainer
 * @param rankMaps   `{ district_rank: {key: rank}, ... }` loaded from R2
 */
export function buildFeatureTensor(features, featureSchema, rankMaps = {}) {
  assertSchemaSupported(featureSchema);

  const out = new Float32Array(featureSchema.length);
  const viewTypes = Array.isArray(features.view_type) ? features.view_type : [];
  const landmarks = features.landmarks || {};

  for (let i = 0; i < featureSchema.length; i++) {
    const col = featureSchema[i];
    const spec = classifyColumn(col);

    switch (spec.kind) {
      case 'numeric':
        out[i] = coerceNumeric(features[col]);
        break;
      case 'boolean':
        out[i] = features[col] ? 1 : 0;
        break;
      case 'rank':
        out[i] = resolveRank(col, spec.keyField, features, rankMaps);
        break;
      case 'onehot':
        out[i] = matchOneHot(spec, features);
        break;
      case 'landmark': {
        const d = landmarks[spec.landmarkKind] ?? landmarks[`${spec.landmarkKind}_m`];
        out[i] = coerceNumeric(d);
        break;
      }
      case 'view':
        out[i] = viewTypes.includes(spec.value) ? 1 : 0;
        break;
    }
  }
  return out;
}

/** `pd.to_numeric(errors="coerce")` — anything unparseable becomes NaN, not 0. */
function coerceNumeric(v) {
  if (v === null || v === undefined || v === '') return NaN;
  const n = Number(v);
  return Number.isFinite(n) ? n : NaN;
}

function resolveRank(col, keyField, features, rankMaps) {
  // A caller that already knows the rank may pass it straight through.
  const explicit = Number(features[col]);
  if (features[col] != null && Number.isFinite(explicit)) return explicit;

  const key = features[keyField];
  if (key === null || key === undefined || key === '') return UNSEEN_RANK;

  const rank = Number((rankMaps?.[col] || {})[String(key)]);
  return Number.isFinite(rank) ? rank : UNSEEN_RANK;
}

function matchOneHot(spec, features) {
  const value = features[spec.field];
  if (spec.suffix === 'nan') return value == null ? 1 : 0;
  return String(value) === spec.suffix ? 1 : 0;
}
