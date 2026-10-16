# Confidence Collapse

A machine-learning monitoring platform for uploading tabular data, training classifiers, inspecting actual evaluation and prediction results, comparing incoming data for drift, and reviewing evidence-backed alerts.

## Run & Operate

- `pnpm --filter @workspace/api-server run dev` — run the API server (port 5000)
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- Required env: `DATABASE_URL` — Postgres connection string

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- API: Express 5
- DB: PostgreSQL + Drizzle ORM
- Validation: Zod (`zod/v4`), `drizzle-zod`
- API codegen: Orval (from OpenAPI spec)
- Build: esbuild (CJS bundle)

## Where things live

- `artifacts/confidence-collapse` — React + Vite web application and responsive dashboard.
- `artifacts/api-server/src/routes/confidence-collapse.ts` — API routes for datasets, models, predictions, monitoring, alerts, reports, and settings.
- `artifacts/api-server/src/lib/confidence-collapse` — CSV validation, classifier training, metrics, predictions, and drift calculations.
- `lib/api-spec/openapi.yaml` — source of truth for the API contract and generated hooks.
- `lib/db/src/schema/confidence-collapse.ts` — PostgreSQL persistence schema.

## Architecture decisions

- Classification and monitoring use the existing TypeScript/Express service; no second Python service is required to run the app.
- Evaluation metrics use the held-out split; monitoring only reports performance when incoming ground-truth labels are present.
- Drift alerts are based on measured distribution changes and configured thresholds, never seeded or generated history.
- Uploaded tabular rows and fitted model trees are persisted in PostgreSQL; uploaded CSV content is not retained as a separate raw file.

## Product

Upload and preview CSV datasets, train Decision Tree and Random Forest classifiers, inspect class-level metrics and probabilities, run feature and prediction drift comparisons, resolve alerts, and export JSON or CSV reports.

## User preferences

- Keep predicted class, confidence, data drift, prediction drift, and measured performance degradation distinct.
- Use genuine backend results and show honest empty states instead of fabricated dashboard content.
- Keep the application name consistently as Confidence Collapse.

## Gotchas

- After editing `lib/api-spec/openapi.yaml`, run `pnpm --filter @workspace/api-spec run codegen`.
- CSV uploads are limited to 10 MB; the API's JSON parser limit is 12 MB.
- Database schema changes are applied to the development database with `pnpm --filter @workspace/db run push`.

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
