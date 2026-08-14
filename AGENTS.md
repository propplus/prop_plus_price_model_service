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

- `npm test` — node:test on `test/features.test.js` (pure functions only),
  **20 cases**; CI greps `# pass 20` / `# fail 0` / `# skipped 0` (ADR-030), so
  changing the suite means updating that number in the same commit
- CI (`.github/workflows/ci.yml`) installs with `npm ci --ignore-scripts`, so
  tests must never import `onnxruntime-node` or download the model

## Cautions

- `src/features.js` must mirror the trainer's feature schema — cross-repo
  contract with `prop_plus_price_trainer::_build_features()`; change BOTH repos
  together. A mismatch does not crash and does not log: it returns a wrong
  price that looks reasonable.
- **Missing values are `NaN`.** No sentinel, no 0. There is deliberately no
  per-model-version branching: `model_runs` was empty when this was decided
  (2026-08-14, FEAT-004), so no older model expecting the `99999` sentinel has
  ever existed. If you ever need to serve a pre-FEAT-004 model, that is a new
  decision — do not assume the compatibility shim used to be here.
  - two exceptions, both mirroring the trainer: `is_off_plan` /
    `is_price_negotiable` → `0` when absent (`fillna(False)`), and `*_rank` →
    `0` for an unseen or absent key (`_smoothed_rank` unseen ⇒ rank 0)
- **A schema column `features.js` does not recognise is a hard failure**
  (`assertSchemaSupported`, thrown at load, not per request). Do not "fix" it
  by restoring a default value — that default is exactly how `lat`, `lng`,
  `project_rank` and `developer_rank` were silently served as 0 for months.
- ⚠️ `propplus-backend` does **not** yet send `developer_name`, so
  `developer_rank` degrades to "unseen" (0) for every request. Correct but
  lossy — see FEAT-004 in the PropPlus registry.
- Do not commit `.env`

## Git Restrictions

Do not use `git add` or `git commit`.
