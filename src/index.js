import http from 'node:http';
import crypto from 'node:crypto';
import onnxruntime from 'onnxruntime-node';
import { assertSchemaSupported, buildFeatureTensor, rankArtifactsFor } from './features.js';

const PORT = Number(process.env.PORT) || 3002;
const SERVICE_TOKEN = process.env.PRICE_MODEL_SERVICE_TOKEN;
const ADMIN_TOKEN = process.env.PRICE_MODEL_ADMIN_TOKEN;
const MODEL_R2_KEY = process.env.MODEL_R2_KEY;
const FEATURE_SCHEMA_R2_KEY = process.env.FEATURE_SCHEMA_R2_KEY;
const SHAP_GLOBAL_R2_KEY = process.env.SHAP_GLOBAL_R2_KEY;
const R2_ACCOUNT_ID = process.env.R2_ACCOUNT_ID;
const R2_ACCESS_KEY_ID = process.env.R2_ACCESS_KEY_ID;
const R2_SECRET_ACCESS_KEY = process.env.R2_SECRET_ACCESS_KEY;
const R2_BUCKET_NAME = process.env.R2_BUCKET_NAME;
const R2_REGION = 'auto';

let session = null;
let inputName = null;
let featureSchema = null;
let rankMaps = {};
let shapGlobal = null;
let modelVersion = null;
let modelReady = false;

// ---------- Minimal SigV4 GET for R2 ----------
function sha256Hex(input) {
  return crypto.createHash('sha256').update(input).digest('hex');
}
function hmac(key, value) {
  return crypto.createHmac('sha256', key).update(value).digest();
}

async function downloadFromR2(key) {
  if (!R2_ACCOUNT_ID || !R2_ACCESS_KEY_ID || !R2_SECRET_ACCESS_KEY || !R2_BUCKET_NAME) {
    throw new Error('Missing R2 credentials env vars');
  }
  const host = `${R2_ACCOUNT_ID}.r2.cloudflarestorage.com`;
  const path = `/${R2_BUCKET_NAME}/${key.split('/').map(encodeURIComponent).join('/')}`;
  const url = `https://${host}${path}`;

  const now = new Date();
  const amzDate = now.toISOString().replace(/[:-]|\.\d{3}/g, '');
  const dateStamp = amzDate.slice(0, 8);
  const service = 's3';
  const payloadHash = sha256Hex('');

  const canonicalHeaders =
    `host:${host}\n` +
    `x-amz-content-sha256:${payloadHash}\n` +
    `x-amz-date:${amzDate}\n`;
  const signedHeaders = 'host;x-amz-content-sha256;x-amz-date';
  const canonicalRequest = `GET\n${path}\n\n${canonicalHeaders}\n${signedHeaders}\n${payloadHash}`;

  const credentialScope = `${dateStamp}/${R2_REGION}/${service}/aws4_request`;
  const stringToSign =
    `AWS4-HMAC-SHA256\n${amzDate}\n${credentialScope}\n${sha256Hex(canonicalRequest)}`;

  const kDate = hmac(`AWS4${R2_SECRET_ACCESS_KEY}`, dateStamp);
  const kRegion = hmac(kDate, R2_REGION);
  const kService = hmac(kRegion, service);
  const kSigning = hmac(kService, 'aws4_request');
  const signature = crypto.createHmac('sha256', kSigning).update(stringToSign).digest('hex');

  const authorization =
    `AWS4-HMAC-SHA256 Credential=${R2_ACCESS_KEY_ID}/${credentialScope}, ` +
    `SignedHeaders=${signedHeaders}, Signature=${signature}`;

  const resp = await fetch(url, {
    method: 'GET',
    headers: {
      'x-amz-date': amzDate,
      'x-amz-content-sha256': payloadHash,
      Authorization: authorization,
    },
  });
  if (!resp.ok) {
    const text = await resp.text().catch(() => '');
    throw new Error(`R2 GET ${key} failed: ${resp.status} ${text}`);
  }
  return Buffer.from(await resp.arrayBuffer());
}

// ---------- Model lifecycle ----------
function deriveVersion(key) {
  if (!key) return null;
  const parts = key.split('/');
  return parts.length >= 3 ? parts[parts.length - 2] : null;
}

async function initializeModel() {
  if (!MODEL_R2_KEY || !FEATURE_SCHEMA_R2_KEY || !SHAP_GLOBAL_R2_KEY) {
    throw new Error('Missing MODEL_R2_KEY / FEATURE_SCHEMA_R2_KEY / SHAP_GLOBAL_R2_KEY');
  }
  console.log(`Loading model from R2: ${MODEL_R2_KEY}`);
  const [modelBuf, schemaBuf, shapBuf] = await Promise.all([
    downloadFromR2(MODEL_R2_KEY),
    downloadFromR2(FEATURE_SCHEMA_R2_KEY),
    downloadFromR2(SHAP_GLOBAL_R2_KEY),
  ]);

  session = await onnxruntime.InferenceSession.create(modelBuf);
  inputName = session.inputNames[0];
  featureSchema = JSON.parse(schemaBuf.toString('utf-8'));
  // Fail at load, not per-request: if the trainer added a column this service
  // cannot encode, every prediction from now on would be silently wrong.
  assertSchemaSupported(featureSchema);
  rankMaps = await loadRankMaps(featureSchema);
  shapGlobal = JSON.parse(shapBuf.toString('utf-8'));
  modelVersion = deriveVersion(MODEL_R2_KEY);
  modelReady = true;
  console.log(
    `Model loaded: version=${modelVersion}, features=${featureSchema.length}, ` +
      `rank encoders=[${Object.keys(rankMaps).join(', ')}]`
  );
}

/**
 * Download the rank-encoder artifacts the schema actually uses
 * (district_rank.json / project_rank.json / developer_rank.json).
 * Missing artifact = hard failure: serving every listing as "unseen" would
 * quietly delete a feature the model was trained on.
 */
async function loadRankMaps(schema) {
  const wanted = rankArtifactsFor(schema);
  const maps = {};
  for (const [col, filename] of Object.entries(wanted)) {
    const key = deriveSiblingArtifactKey(FEATURE_SCHEMA_R2_KEY, filename);
    if (!key) {
      throw new Error(`Cannot derive ${filename} key from FEATURE_SCHEMA_R2_KEY`);
    }
    const buf = await downloadFromR2(key);
    const parsed = JSON.parse(buf.toString('utf-8'));
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      throw new Error(`Invalid rank artifact for ${col}: ${key}`);
    }
    maps[col] = parsed;
  }
  return maps;
}

function deriveSiblingArtifactKey(key, filename) {
  if (!key) return null;
  const idx = key.lastIndexOf('/');
  return idx === -1 ? filename : `${key.slice(0, idx + 1)}${filename}`;
}

// ---------- Feature encoding ----------
// The column vocabulary and the missing-value rules live in src/features.js —
// that file is the cross-repo contract with the trainer and is unit-tested.
// Anything the schema contains and features.js does not recognise makes
// initializeModel() throw, so the service never serves a half-understood model.

/** Local helper for output arithmetic only — NOT for building the tensor. */
function num(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : 0;
}

function topShapFeatures(limit = 5) {
  if (!shapGlobal) return [];
  return Object.entries(shapGlobal)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, limit)
    .map(([name, contribution]) => ({ name, contribution: Number(contribution) }));
}

async function predictPrice(features) {
  const tensor = buildFeatureTensor(features, featureSchema, rankMaps);
  const ortTensor = new onnxruntime.Tensor('float32', tensor, [1, featureSchema.length]);
  const result = await session.run({ [inputName]: ortTensor });
  const outName = session.outputNames[0];
  const out = result[outName];
  const logPred = Number(out.data[0]);
  // Trainer fits on log1p(price_per_sqm) → invert with expm1.
  const predictedPps = Math.expm1(logPred);
  const band = 0.15; // ±15% on log scale → v1 confidence band
  const low = Math.expm1(logPred - band);
  const high = Math.expm1(logPred + band);
  const area = num(features.area_sqm);
  const total = area > 0 ? predictedPps * area : null;
  return {
    predicted_price_per_sqm: predictedPps,
    predicted_total_price: total,
    confidence_low: low,
    confidence_high: high,
    top_shap_features: topShapFeatures(),
    model_version: modelVersion,
  };
}

// ---------- HTTP ----------
function sendJson(res, statusCode, body) {
  const payload = JSON.stringify(body);
  res.writeHead(statusCode, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': Buffer.byteLength(payload),
  });
  res.end(payload);
}

function parseBody(req) {
  return new Promise((resolve) => {
    const chunks = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => {
      if (chunks.length === 0) return resolve({});
      try {
        resolve(JSON.parse(Buffer.concat(chunks).toString('utf-8')));
      } catch {
        resolve({});
      }
    });
    req.on('error', () => resolve({}));
  });
}

const server = http.createServer(async (req, res) => {
  const { method, url } = req;

  if (method === 'GET' && url === '/health') {
    if (!modelReady) return sendJson(res, 503, { ok: false });
    return sendJson(res, 200, { ok: true, model_version: modelVersion });
  }

  if (method === 'POST' && url === '/predict-price') {
    const token = req.headers['x-price-model-token'];
    if (!SERVICE_TOKEN || token !== SERVICE_TOKEN)
      return sendJson(res, 401, { error: 'Unauthorized' });
    if (!modelReady) return sendJson(res, 503, { error: 'Model not ready' });
    const body = await parseBody(req);
    if (!body || typeof body.features !== 'object' || body.features === null)
      return sendJson(res, 400, { error: 'features is required' });
    try {
      const out = await predictPrice(body.features);
      return sendJson(res, 200, out);
    } catch (err) {
      return sendJson(res, 500, { error: err.message });
    }
  }

  if (method === 'POST' && url === '/reload') {
    const token = req.headers['x-price-model-admin-token'];
    if (!ADMIN_TOKEN || token !== ADMIN_TOKEN)
      return sendJson(res, 401, { error: 'Unauthorized' });
    try {
      await initializeModel();
      return sendJson(res, 200, { ok: true, model_version: modelVersion });
    } catch (err) {
      return sendJson(res, 500, { error: err.message });
    }
  }

  sendJson(res, 404, { error: 'Not found' });
});

async function main() {
  if (!SERVICE_TOKEN) {
    console.warn(
      'WARNING: PRICE_MODEL_SERVICE_TOKEN is not set. All /predict-price requests will be rejected with 401.'
    );
  }
  try {
    await initializeModel();
  } catch (err) {
    console.error('Initial model load failed (service will still start):', err);
  }
  server.listen(PORT, () => {
    console.log(`Price model service ready on port ${PORT}`);
  });
}

main().catch((err) => {
  console.error('Failed to start price model service:', err);
  process.exit(1);
});
