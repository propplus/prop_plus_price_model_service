# prop_plus_price_model_service

Node HTTP service that loads the PropPlus hedonic LightGBM model (exported as ONNX)
from R2 and serves price predictions to the Cloudflare Worker backend.

## Local dev

```
npm install
cp .env.example .env   # fill in tokens + R2 creds
npm run dev
```

## Deploy

```
fly deploy
```

## Hot reload (no redeploy) after a new training run

```
curl -X POST https://<host>/reload \
  -H "x-price-model-admin-token: <admin-token>"
```

## Endpoints

| Method | Path             | Auth header                     | Purpose                |
|--------|------------------|---------------------------------|------------------------|
| GET    | /health          | —                               | Readiness + version    |
| POST   | /predict-price   | `x-price-model-token`           | Predict price          |
| POST   | /reload          | `x-price-model-admin-token`     | Re-download model      |

## Required env vars

| Name                          | Purpose                                                   |
|-------------------------------|-----------------------------------------------------------|
| `PORT`                        | HTTP port (default `3002`)                                |
| `PRICE_MODEL_SERVICE_TOKEN`   | Shared secret for `/predict-price`                        |
| `PRICE_MODEL_ADMIN_TOKEN`     | Shared secret for `/reload`                               |
| `MODEL_R2_KEY`                | R2 object key for `model.onnx`                            |
| `FEATURE_SCHEMA_R2_KEY`       | R2 object key for `feature_schema.json`                   |
| `SHAP_GLOBAL_R2_KEY`          | R2 object key for `shap_global.json`                      |
| `R2_ACCOUNT_ID`               | Cloudflare R2 account id                                  |
| `R2_ACCESS_KEY_ID`            | R2 access key                                             |
| `R2_SECRET_ACCESS_KEY`        | R2 secret                                                 |
| `R2_BUCKET_NAME`              | R2 bucket (e.g. `propplus-ml-models`)                     |
