export const LANDMARK_SENTINEL = 99999.0;

export function buildFeatureTensor(features, featureSchema, districtRankMap = {}) {
  const out = new Float32Array(featureSchema.length);
  const propertyTypeId = features.property_type_id;
  const tenure = features.tenure;
  const fquota = features.foreign_quota_status;
  const districtRank = resolveDistrictRank(features, districtRankMap);
  const viewTypes = Array.isArray(features.view_type) ? features.view_type : [];
  const landmarks = features.landmarks || {};

  for (let i = 0; i < featureSchema.length; i++) {
    const col = featureSchema[i];
    let v = 0;
    if (col === 'area_sqm') v = num(features.area_sqm);
    else if (col === 'bedrooms_count') v = num(features.bedrooms_count);
    else if (col === 'bathrooms_count') v = num(features.bathrooms_count);
    else if (col === 'completion_year') v = num(features.completion_year);
    else if (col === 'is_off_plan') v = features.is_off_plan ? 1 : 0;
    else if (col === 'is_price_negotiable') v = features.is_price_negotiable ? 1 : 0;
    else if (col === 'district_rank') v = districtRank;
    else if (col.startsWith('ptype_')) v = matchOneHot(col, 'ptype_', propertyTypeId);
    else if (col.startsWith('tenure_')) v = matchOneHot(col, 'tenure_', tenure);
    else if (col.startsWith('fquota_')) v = matchOneHot(col, 'fquota_', fquota);
    else if (col.startsWith('view_')) v = viewTypes.includes(col.slice('view_'.length)) ? 1 : 0;
    else if (col.startsWith('dist_') && col.endsWith('_m')) {
      const kind = col.slice('dist_'.length, col.length - 2);
      const d = landmarks[kind] ?? landmarks[`${kind}_m`];
      v = Number.isFinite(d) ? Number(d) : LANDMARK_SENTINEL;
    }
    out[i] = v;
  }
  return out;
}

function resolveDistrictRank(features, districtRankMap) {
  if (Number.isFinite(features.district_rank)) return Number(features.district_rank);
  const districtCode = features.district_code;
  if (districtCode == null) return 0;
  const rank = districtRankMap[String(districtCode)];
  return Number.isFinite(rank) ? Number(rank) : 0;
}

function num(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : 0;
}

function matchOneHot(col, prefix, value) {
  const suffix = col.slice(prefix.length);
  if (suffix === 'nan') return value == null ? 1 : 0;
  return String(value) === suffix ? 1 : 0;
}
