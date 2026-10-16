import {
  integer,
  jsonb,
  pgTable,
  real,
  text,
  timestamp,
  uuid,
} from "drizzle-orm/pg-core";

export const datasetsTable = pgTable("cc_datasets", {
  id: uuid("id").primaryKey(),
  name: text("name").notNull(),
  purpose: text("purpose").notNull(),
  targetColumn: text("target_column"),
  rows: jsonb("rows").notNull(),
  columns: jsonb("columns").notNull(),
  rowCount: integer("row_count").notNull(),
  columnCount: integer("column_count").notNull(),
  missingCellCount: integer("missing_cell_count").notNull(),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull(),
});

export const modelsTable = pgTable("cc_models", {
  id: uuid("id").primaryKey(),
  name: text("name").notNull(),
  datasetId: uuid("dataset_id")
    .notNull()
    .references(() => datasetsTable.id, { onDelete: "restrict" }),
  algorithm: text("algorithm").notNull(),
  targetColumn: text("target_column").notNull(),
  testSize: real("test_size").notNull(),
  randomSeed: integer("random_seed").notNull(),
  maxDepth: integer("max_depth").notNull(),
  estimators: integer("estimators").notNull(),
  rowCount: integer("row_count").notNull(),
  featureCount: integer("feature_count").notNull(),
  classes: jsonb("classes").notNull(),
  metrics: jsonb("metrics").notNull(),
  featureImportances: jsonb("feature_importances").notNull(),
  modelData: jsonb("model_data").notNull(),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull(),
});

export const predictionsTable = pgTable("cc_predictions", {
  id: uuid("id").primaryKey(),
  modelId: uuid("model_id")
    .notNull()
    .references(() => modelsTable.id, { onDelete: "cascade" }),
  rowIndex: integer("row_index").notNull(),
  predictedLabel: text("predicted_label").notNull(),
  actualLabel: text("actual_label"),
  confidence: real("confidence").notNull(),
  probabilities: jsonb("probabilities").notNull(),
  source: text("source").notNull(),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull(),
});

export const monitoringRunsTable = pgTable("cc_monitoring_runs", {
  id: uuid("id").primaryKey(),
  modelId: uuid("model_id")
    .notNull()
    .references(() => modelsTable.id, { onDelete: "restrict" }),
  datasetId: uuid("dataset_id")
    .notNull()
    .references(() => datasetsTable.id, { onDelete: "restrict" }),
  results: jsonb("results").notNull(),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull(),
});

export const alertsTable = pgTable("cc_alerts", {
  id: uuid("id").primaryKey(),
  modelId: uuid("model_id")
    .notNull()
    .references(() => modelsTable.id, { onDelete: "cascade" }),
  monitoringRunId: uuid("monitoring_run_id")
    .notNull()
    .references(() => monitoringRunsTable.id, { onDelete: "cascade" }),
  severity: text("severity").notNull(),
  title: text("title").notNull(),
  explanation: text("explanation").notNull(),
  recommendedAction: text("recommended_action").notNull(),
  status: text("status").notNull(),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull(),
});

export const appSettingsTable = pgTable("cc_settings", {
  id: text("id").primaryKey(),
  driftThreshold: real("drift_threshold").notNull(),
  predictionDriftThreshold: real("prediction_drift_threshold").notNull(),
  performanceDropThreshold: real("performance_drop_threshold").notNull(),
  updatedAt: timestamp("updated_at", { withTimezone: true }).notNull(),
});
