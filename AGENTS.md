# prop_plus_price_model_service — Agent Notes

Node HTTP service (deployed on Fly.io) that loads the hedonic LightGBM model
(ONNX) from R2 and serves price predictions to the Cloudflare Worker backend.
Own git repo, independent of the PropPlus root.

- Stack: Node 22, `onnxruntime-node`, plain `node:http` (no framework)
- Local dev: `npm install && cp .env.example .env && npm run dev` (port 3002)
- Deploy: `fly deploy` · hot-reload new model without redeploy: `POST /reload`
  with `x-price-model-admin-token`
- Endpoints and required env vars: see `README.md`

## Tests / CI

- `npm test` — node:test on `test/features.test.js` (pure functions only)
- CI (`.github/workflows/ci.yml`) installs with `npm ci --ignore-scripts`, so
  tests must never import `onnxruntime-node` or download the model

## Cautions

- `src/features.js` must mirror the trainer's feature schema — cross-repo
  contract with `prop_plus_price_trainer` (incl. missing-landmark sentinel vs
  NaN, keyed by model version); change BOTH repos together
- Do not commit `.env`

## Git Restrictions

Do not use `git add` or `git commit`.
