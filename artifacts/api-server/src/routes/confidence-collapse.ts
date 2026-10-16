import { randomUUID } from "node:crypto";
import { desc, eq } from "drizzle-orm";
import { Router, type IRouter } from "express";
import {
  GetDatasetParams,
  GetModelParams,
  GetModelReportParams,
  ListPredictionsParams,
  PredictWithModelBody,
  PredictWithModelParams,
  RunMonitoringBody,
  TrainModelBody,
  UpdateAlertBody,
  UpdateAlertParams,
  UpdateSettingsBody,
  UploadDatasetBody,
} from "@workspace/api-zod";
import {
  alertsTable,
  appSettingsTable,
  datasetsTable,
  modelsTable,
  monitoringRunsTable,
  predictionsTable,
} from "@workspace/db";
import { db } from "@workspace/db";
import { parseCsv, toDatasetSummary, type DataRow } from "../lib/confidence-collapse/csv";
import { InputError } from "../lib/confidence-collapse/errors";
import {
  isUsableLabel,
  predictRow,
  trainClassifier,
  type EvaluationMetrics,
  type SerializedModel,
} from "../lib/confidence-collapse/ml";
import {
  analyzeMonitoring,
  type MonitoringAnalysis,
} from "../lib/confidence-collapse/drift";

const router: IRouter = Router();

class ApiFailure extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

function handleError(res: import("express").Response, error: unknown) {
  if (error instanceof ApiFailure) {
    return res.status(error.status).json({ error: error.message });
  }
  if (error instanceof Error && error.name === "ZodError") {
    return res.status(400).json({ error: error.message });
  }
  if (error instanceof InputError) {
    return res.status(400).json({ error: error.message });
  }
  return res.status(500).json({ error: "An unexpected server error occurred." });
}

type DatasetRecord = typeof datasetsTable.$inferSelect;
type ModelRecord = typeof modelsTable.$inferSelect;
type AlertRecord = typeof alertsTable.$inferSelect;
type RunRecord = typeof monitoringRunsTable.$inferSelect;
type PredictionRecord = typeof predictionsTable.$inferSelect;
type StoredMonitoringResults = Omit<MonitoringAnalysis, "predictions" | "alerts">;

function datasetDetail(dataset: DatasetRecord) {
  return {
    ...toDatasetSummary(dataset),
    preview: (dataset.rows as DataRow[]).slice(0, 12),
  };
}

function modelJson(
  model: ModelRecord,
  datasetName: string,
) {
  return {
    id: model.id,
    name: model.name,
    datasetId: model.datasetId,
    datasetName,
    targetColumn: model.targetColumn,
    algorithm: model.algorithm as "decision_tree" | "random_forest",
    testSize: model.testSize,
    randomSeed: model.randomSeed,
    trainedAt: model.createdAt.toISOString(),
    rowCount: model.rowCount,
    featureCount: model.featureCount,
    classes: model.classes as string[],
    metrics: model.metrics as EvaluationMetrics,
    featureImportances: model.featureImportances as { feature: string; importance: number }[],
  };
}

function predictionJson(prediction: PredictionRecord) {
  return {
    id: prediction.id,
    modelId: prediction.modelId,
    rowIndex: prediction.rowIndex,
    predictedLabel: prediction.predictedLabel,
    actualLabel: prediction.actualLabel,
    confidence: prediction.confidence,
    probabilities: prediction.probabilities as Record<string, number>,
    source: prediction.source as "evaluation" | "manual",
    createdAt: prediction.createdAt.toISOString(),
  };
}

function alertJson(alert: AlertRecord) {
  return {
    id: alert.id,
    modelId: alert.modelId,
    monitoringRunId: alert.monitoringRunId,
    severity: alert.severity as "low" | "medium" | "high",
    title: alert.title,
    explanation: alert.explanation,
    recommendedAction: alert.recommendedAction,
    status: alert.status as "open" | "resolved",
    createdAt: alert.createdAt.toISOString(),
  };
}

function monitoringRunJson(
  run: RunRecord,
  modelName: string,
  datasetName: string,
  alertIds: string[],
) {
  const results = run.results as StoredMonitoringResults;
  return {
    id: run.id,
    modelId: run.modelId,
    modelName,
    datasetId: run.datasetId,
    datasetName,
    createdAt: run.createdAt.toISOString(),
    ...results,
    alertIds,
  };
}

async function getDatasetOrFail(id: string): Promise<DatasetRecord> {
  const [dataset] = await db.select().from(datasetsTable).where(eq(datasetsTable.id, id)).limit(1);
  if (!dataset) throw new ApiFailure("Dataset not found.", 404);
  return dataset;
}

async function getModelOrFail(id: string): Promise<ModelRecord> {
  const [model] = await db.select().from(modelsTable).where(eq(modelsTable.id, id)).limit(1);
  if (!model) throw new ApiFailure("Model not found.", 404);
  return model;
}

async function getModelJson(model: ModelRecord) {
  const dataset = await getDatasetOrFail(model.datasetId);
  return modelJson(model, dataset.name);
}

async function ensureSettings() {
  await db
    .insert(appSettingsTable)
    .values({
      id: "default",
      driftThreshold: 0.2,
      predictionDriftThreshold: 0.2,
      performanceDropThreshold: 0.1,
      updatedAt: new Date(),
    })
    .onConflictDoNothing();
  const [settings] = await db.select().from(appSettingsTable).where(eq(appSettingsTable.id, "default"));
  if (!settings) throw new Error("Monitoring settings could not be loaded.");
  return settings;
}

function settingsJson(settings: typeof appSettingsTable.$inferSelect) {
  return {
    driftThreshold: settings.driftThreshold,
    predictionDriftThreshold: settings.predictionDriftThreshold,
    performanceDropThreshold: settings.performanceDropThreshold,
  };
}

router.get("/dashboard/summary", async (_req, res) => {
  try {
    const [datasets, models, openAlerts, runs, activityAlerts] = await Promise.all([
      db.select().from(datasetsTable).orderBy(desc(datasetsTable.createdAt)),
      db
        .select({ model: modelsTable, datasetName: datasetsTable.name })
        .from(modelsTable)
        .innerJoin(datasetsTable, eq(modelsTable.datasetId, datasetsTable.id))
        .orderBy(desc(modelsTable.createdAt)),
      db.select().from(alertsTable).where(eq(alertsTable.status, "open")),
      db
        .select({ run: monitoringRunsTable, modelName: modelsTable.name, datasetName: datasetsTable.name })
        .from(monitoringRunsTable)
        .innerJoin(modelsTable, eq(monitoringRunsTable.modelId, modelsTable.id))
        .innerJoin(datasetsTable, eq(monitoringRunsTable.datasetId, datasetsTable.id))
        .orderBy(desc(monitoringRunsTable.createdAt))
        .limit(10),
      db.select().from(alertsTable).orderBy(desc(alertsTable.createdAt)).limit(10),
    ]);

    const latestRun = runs[0];
    const activity = [
      ...datasets.slice(0, 6).map((dataset) => ({
        id: `dataset-${dataset.id}`,
        kind: "dataset" as const,
        title: dataset.name,
        detail: `${dataset.rowCount.toLocaleString()} rows · dataset uploaded`,
        createdAt: dataset.createdAt,
      })),
      ...models.slice(0, 6).map(({ model, datasetName }) => ({
        id: `model-${model.id}`,
        kind: "model" as const,
        title: model.name,
        detail: `${datasetName} · ${(model.metrics as EvaluationMetrics).accuracy.toLocaleString(undefined, { style: "percent", maximumFractionDigits: 1 })} accuracy`,
        createdAt: model.createdAt,
      })),
      ...runs.slice(0, 6).map(({ run, modelName, datasetName }) => ({
        id: `monitoring-${run.id}`,
        kind: "monitoring" as const,
        title: `Monitoring · ${modelName}`,
        detail: `${datasetName} · drift score ${Number((run.results as StoredMonitoringResults).overallDriftScore).toFixed(3)}`,
        createdAt: run.createdAt,
      })),
      ...activityAlerts.slice(0, 6).map((alert) => ({
        id: `alert-${alert.id}`,
        kind: "alert" as const,
        title: alert.title,
        detail: `${alert.severity} · ${alert.status}`,
        createdAt: alert.createdAt,
      })),
    ]
      .sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime())
      .slice(0, 8)
      .map((item) => ({ ...item, createdAt: item.createdAt.toISOString() }));

    const [latestAccuracy] = models.length
      ? [(models[0].model.metrics as EvaluationMetrics).accuracy]
      : [null];
    const latestResults = latestRun?.run.results as StoredMonitoringResults | undefined;
    res.json({
      datasetCount: datasets.length,
      modelCount: models.length,
      openAlertCount: openAlerts.length,
      latestAccuracy,
      latestDriftScore: latestResults?.overallDriftScore ?? null,
      latestRunAt: latestRun?.run.createdAt.toISOString() ?? null,
      activity,
    });
  } catch (error) {
    handleError(res, error);
  }
});

router.get("/datasets", async (_req, res) => {
  try {
    const datasets = await db.select().from(datasetsTable).orderBy(desc(datasetsTable.createdAt));
    res.json(datasets.map(toDatasetSummary));
  } catch (error) {
    handleError(res, error);
  }
});

router.post("/datasets", async (req, res) => {
  try {
    const input = UploadDatasetBody.parse(req.body);
    const parsed = parseCsv(input.csvContent);
    const targetColumn = input.targetColumn ?? null;
    if (targetColumn && !parsed.columns.some((column) => column.name === targetColumn)) {
      throw new ApiFailure("The selected target column is not in this CSV.", 400);
    }
    const [dataset] = await db
      .insert(datasetsTable)
      .values({
        id: randomUUID(),
        name: input.name.trim(),
        purpose: input.purpose,
        targetColumn,
        rows: parsed.rows,
        columns: parsed.columns,
        rowCount: parsed.rowCount,
        columnCount: parsed.columnCount,
        missingCellCount: parsed.missingCellCount,
        createdAt: new Date(),
      })
      .returning();
    res.status(201).json(toDatasetSummary(dataset));
  } catch (error) {
    handleError(res, error);
  }
});

router.get("/datasets/:datasetId", async (req, res) => {
  try {
    const { datasetId } = GetDatasetParams.parse(req.params);
    res.json(datasetDetail(await getDatasetOrFail(datasetId)));
  } catch (error) {
    handleError(res, error);
  }
});

router.delete("/datasets/:datasetId", async (req, res) => {
  try {
    const { datasetId } = GetDatasetParams.parse(req.params);
    const [model] = await db.select({ id: modelsTable.id }).from(modelsTable).where(eq(modelsTable.datasetId, datasetId)).limit(1);
    if (model) throw new ApiFailure("This dataset is referenced by a trained model and cannot be deleted.", 409);
    const [deleted] = await db.delete(datasetsTable).where(eq(datasetsTable.id, datasetId)).returning({ id: datasetsTable.id });
    if (!deleted) throw new ApiFailure("Dataset not found.", 404);
    res.status(204).end();
  } catch (error) {
    handleError(res, error);
  }
});

router.get("/models", async (_req, res) => {
  try {
    const models = await db
      .select({ model: modelsTable, datasetName: datasetsTable.name })
      .from(modelsTable)
      .innerJoin(datasetsTable, eq(modelsTable.datasetId, datasetsTable.id))
      .orderBy(desc(modelsTable.createdAt));
    res.json(models.map(({ model, datasetName }) => modelJson(model, datasetName)));
  } catch (error) {
    handleError(res, error);
  }
});

router.post("/models/train", async (req, res) => {
  try {
    const input = TrainModelBody.parse(req.body);
    const dataset = await getDatasetOrFail(input.datasetId);
    if (dataset.purpose !== "reference") {
      throw new ApiFailure("Train models from a reference dataset, not an incoming dataset.", 400);
    }
    const rows = dataset.rows as DataRow[];
    const columns = dataset.columns as Parameters<typeof trainClassifier>[0]["columns"];
    const trained = trainClassifier({
      rows,
      columns,
      targetColumn: input.targetColumn,
      algorithm: input.algorithm,
      testSize: input.testSize,
      randomSeed: input.randomSeed,
      maxDepth: input.maxDepth,
      estimators: input.estimators,
    });
    const modelId = randomUUID();
    const [model] = await db.transaction(async (tx) => {
      const [created] = await tx
        .insert(modelsTable)
        .values({
          id: modelId,
          name: `${input.algorithm === "decision_tree" ? "Decision Tree" : "Random Forest"} · ${input.targetColumn}`,
          datasetId: dataset.id,
          algorithm: input.algorithm,
          targetColumn: input.targetColumn,
          testSize: input.testSize,
          randomSeed: input.randomSeed,
          maxDepth: input.maxDepth,
          estimators: input.algorithm === "random_forest" ? input.estimators : 1,
          rowCount: trained.rowCount,
          featureCount: trained.featureCount,
          classes: trained.classes,
          metrics: trained.metrics,
          featureImportances: trained.featureImportances,
          modelData: trained.serialized,
          createdAt: new Date(),
        })
        .returning();
      if (trained.evaluation.length > 0) {
        await tx.insert(predictionsTable).values(
          trained.evaluation.map((prediction) => ({
            id: randomUUID(),
            modelId,
            rowIndex: prediction.rowIndex,
            predictedLabel: prediction.predictedLabel,
            actualLabel: prediction.actualLabel,
            confidence: prediction.confidence,
            probabilities: prediction.probabilities,
            source: "evaluation",
            createdAt: new Date(),
          })),
        );
      }
      return [created];
    });
    res.status(201).json(modelJson(model, dataset.name));
  } catch (error) {
    handleError(res, error);
  }
});

router.get("/models/:modelId", async (req, res) => {
  try {
    const { modelId } = GetModelParams.parse(req.params);
    res.json(await getModelJson(await getModelOrFail(modelId)));
  } catch (error) {
    handleError(res, error);
  }
});

router.post("/models/:modelId/predict", async (req, res) => {
  try {
    const { modelId } = PredictWithModelParams.parse(req.params);
    const input = PredictWithModelBody.parse(req.body);
    const modelRecord = await getModelOrFail(modelId);
    const model = modelRecord.modelData as SerializedModel;
    const rows = input.rows as DataRow[];
    const predicted = rows.map((row, rowIndex) => {
      const prediction = predictRow(model, row);
      const actualLabel = isUsableLabel(row[modelRecord.targetColumn])
        ? String(row[modelRecord.targetColumn])
        : null;
      return { row, rowIndex, actualLabel, ...prediction };
    });
    const saved = await db
      .insert(predictionsTable)
      .values(
        predicted.map((prediction) => ({
        id: randomUUID(),
        modelId,
        rowIndex: prediction.rowIndex,
        predictedLabel: prediction.predictedLabel,
        actualLabel: prediction.actualLabel,
        confidence: prediction.confidence,
        probabilities: prediction.probabilities,
        source: "manual",
        createdAt: new Date(),
        })),
      )
      .returning();
    const distribution = Object.fromEntries(
      model.classes.map((label) => [
        label,
        predicted.filter((prediction) => prediction.predictedLabel === label).length,
      ]),
    );
    res.json({
      modelId,
      predictions: saved.map(predictionJson),
      distribution,
    });
  } catch (error) {
    handleError(res, error);
  }
});

router.get("/models/:modelId/predictions", async (req, res) => {
  try {
    const { modelId } = ListPredictionsParams.parse(req.params);
    await getModelOrFail(modelId);
    const predictions = await db
      .select()
      .from(predictionsTable)
      .where(eq(predictionsTable.modelId, modelId))
      .orderBy(desc(predictionsTable.createdAt))
      .limit(5_000);
    res.json(predictions.map(predictionJson));
  } catch (error) {
    handleError(res, error);
  }
});

router.get("/monitoring/runs", async (_req, res) => {
  try {
    const [runs, alerts] = await Promise.all([
      db
        .select({ run: monitoringRunsTable, modelName: modelsTable.name, datasetName: datasetsTable.name })
        .from(monitoringRunsTable)
        .innerJoin(modelsTable, eq(monitoringRunsTable.modelId, modelsTable.id))
        .innerJoin(datasetsTable, eq(monitoringRunsTable.datasetId, datasetsTable.id))
        .orderBy(desc(monitoringRunsTable.createdAt))
        .limit(250),
      db.select({ id: alertsTable.id, monitoringRunId: alertsTable.monitoringRunId }).from(alertsTable),
    ]);
    const alertIdsByRun = new Map<string, string[]>();
    alerts.forEach((alert) =>
      alertIdsByRun.set(alert.monitoringRunId, [
        ...(alertIdsByRun.get(alert.monitoringRunId) ?? []),
        alert.id,
      ]),
    );
    res.json(
      runs.map(({ run, modelName, datasetName }) =>
        monitoringRunJson(run, modelName, datasetName, alertIdsByRun.get(run.id) ?? []),
      ),
    );
  } catch (error) {
    handleError(res, error);
  }
});

router.post("/monitoring/runs", async (req, res) => {
  try {
    const input = RunMonitoringBody.parse(req.body);
    const modelRecord = await getModelOrFail(input.modelId);
    const referenceDataset = await getDatasetOrFail(modelRecord.datasetId);
    const incomingDataset = await getDatasetOrFail(input.datasetId);
    if (incomingDataset.purpose !== "incoming") {
      throw new ApiFailure("Choose a dataset uploaded for incoming monitoring.", 400);
    }
    const columns = incomingDataset.columns as Parameters<typeof trainClassifier>[0]["columns"];
    const missingFeatures = (modelRecord.modelData as SerializedModel).features
      .map((feature) => feature.name)
      .filter((feature) => !columns.some((column) => column.name === feature));
    if (missingFeatures.length > 0) {
      throw new ApiFailure(
        `Incoming dataset is missing required model features: ${missingFeatures.join(", ")}.`,
        400,
      );
    }
    const incompatibleFeatures = (modelRecord.modelData as SerializedModel).features
      .filter((feature) => columns.find((column) => column.name === feature.name)?.type !== feature.type)
      .map((feature) => feature.name);
    if (incompatibleFeatures.length > 0) {
      throw new ApiFailure(
        `Incoming feature types do not match the reference dataset: ${incompatibleFeatures.join(", ")}.`,
        400,
      );
    }
    const [settings] = await db.select().from(appSettingsTable).where(eq(appSettingsTable.id, "default"));
    const activeSettings =
      settings ??
      (await db
        .insert(appSettingsTable)
        .values({
          id: "default",
          driftThreshold: 0.2,
          predictionDriftThreshold: 0.2,
          performanceDropThreshold: 0.1,
          updatedAt: new Date(),
        })
        .returning())[0];
    const analysis = analyzeMonitoring({
      model: modelRecord.modelData as SerializedModel,
      referenceRows: referenceDataset.rows as DataRow[],
      incomingRows: incomingDataset.rows as DataRow[],
      columns: referenceDataset.columns as Parameters<typeof trainClassifier>[0]["columns"],
      targetColumn: modelRecord.targetColumn,
      baselineMetrics: modelRecord.metrics as EvaluationMetrics,
      thresholds: {
        driftThreshold: activeSettings.driftThreshold,
        predictionDriftThreshold: activeSettings.predictionDriftThreshold,
        performanceDropThreshold: activeSettings.performanceDropThreshold,
      },
    });
    const runId = randomUUID();
    const createdAt = new Date();
    const { predictions: _predictions, alerts: alertDrafts, ...storedResults } = analysis;
    const [run] = await db.transaction(async (tx) => {
      const [created] = await tx
        .insert(monitoringRunsTable)
        .values({
          id: runId,
          modelId: modelRecord.id,
          datasetId: incomingDataset.id,
          results: storedResults,
          createdAt,
        })
        .returning();
      if (alertDrafts.length > 0) {
        await tx.insert(alertsTable).values(
          alertDrafts.map((alert) => ({
            id: randomUUID(),
            modelId: modelRecord.id,
            monitoringRunId: runId,
            severity: alert.severity,
            title: alert.title,
            explanation: alert.explanation,
            recommendedAction: alert.recommendedAction,
            status: "open",
            createdAt,
          })),
        );
      }
      return [created];
    });
    const savedAlerts = await db
      .select({ id: alertsTable.id })
      .from(alertsTable)
      .where(eq(alertsTable.monitoringRunId, run.id));
    res.status(201).json(
      monitoringRunJson(run, modelRecord.name, incomingDataset.name, savedAlerts.map((alert) => alert.id)),
    );
  } catch (error) {
    handleError(res, error);
  }
});

router.get("/alerts", async (_req, res) => {
  try {
    const alerts = await db.select().from(alertsTable).orderBy(desc(alertsTable.createdAt)).limit(1_000);
    res.json(alerts.map(alertJson));
  } catch (error) {
    handleError(res, error);
  }
});

router.patch("/alerts/:alertId", async (req, res) => {
  try {
    const { alertId } = UpdateAlertParams.parse(req.params);
    const input = UpdateAlertBody.parse(req.body);
    const [alert] = await db
      .update(alertsTable)
      .set({ status: input.status })
      .where(eq(alertsTable.id, alertId))
      .returning();
    if (!alert) throw new ApiFailure("Alert not found.", 404);
    res.json(alertJson(alert));
  } catch (error) {
    handleError(res, error);
  }
});

router.get("/settings", async (_req, res) => {
  try {
    res.json(settingsJson(await ensureSettings()));
  } catch (error) {
    handleError(res, error);
  }
});

router.patch("/settings", async (req, res) => {
  try {
    const input = UpdateSettingsBody.parse(req.body);
    const [settings] = await db
      .insert(appSettingsTable)
      .values({
        id: "default",
        ...input,
        updatedAt: new Date(),
      })
      .onConflictDoUpdate({
        target: appSettingsTable.id,
        set: { ...input, updatedAt: new Date() },
      })
      .returning();
    res.json(settingsJson(settings));
  } catch (error) {
    handleError(res, error);
  }
});

router.get("/reports/:modelId", async (req, res) => {
  try {
    const { modelId } = GetModelReportParams.parse(req.params);
    const model = await getModelOrFail(modelId);
    const dataset = await getDatasetOrFail(model.datasetId);
    const [predictions, runs, alerts] = await Promise.all([
      db
        .select()
        .from(predictionsTable)
        .where(eq(predictionsTable.modelId, modelId))
        .orderBy(desc(predictionsTable.createdAt))
        .limit(5_000),
      db
        .select({ run: monitoringRunsTable, datasetName: datasetsTable.name, modelName: modelsTable.name })
        .from(monitoringRunsTable)
        .innerJoin(datasetsTable, eq(monitoringRunsTable.datasetId, datasetsTable.id))
        .innerJoin(modelsTable, eq(monitoringRunsTable.modelId, modelsTable.id))
        .where(eq(monitoringRunsTable.modelId, modelId))
        .orderBy(desc(monitoringRunsTable.createdAt)),
      db.select().from(alertsTable).where(eq(alertsTable.modelId, modelId)).orderBy(desc(alertsTable.createdAt)),
    ]);
    const runAlertRows = await db
      .select({ id: alertsTable.id, monitoringRunId: alertsTable.monitoringRunId })
      .from(alertsTable)
      .where(eq(alertsTable.modelId, modelId));
    const alertIdsByRun = new Map<string, string[]>();
    runAlertRows.forEach((alert) =>
      alertIdsByRun.set(alert.monitoringRunId, [
        ...(alertIdsByRun.get(alert.monitoringRunId) ?? []),
        alert.id,
      ]),
    );
    res.json({
      model: modelJson(model, dataset.name),
      predictions: predictions.map(predictionJson),
      monitoringRuns: runs.map(({ run, modelName, datasetName }) =>
        monitoringRunJson(run, modelName, datasetName, alertIdsByRun.get(run.id) ?? []),
      ),
      alerts: alerts.map(alertJson),
    });
  } catch (error) {
    handleError(res, error);
  }
});

export default router;
