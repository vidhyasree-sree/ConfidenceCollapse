import type { ColumnSummary, DataRow } from "./csv";
import {
  evaluateLabels,
  isUsableLabel,
  predictRow,
  type EvaluationMetrics,
  type SerializedModel,
} from "./ml";

export interface DistributionBin {
  label: string;
  count: number;
}

export interface FeatureDriftResult {
  feature: string;
  type: "numeric" | "categorical";
  score: number;
  pValue: number | null;
  method: "kolmogorov_smirnov" | "chi_square" | "unavailable";
  status: "low" | "warning" | "high" | "unavailable";
  referenceDistribution: DistributionBin[];
  currentDistribution: DistributionBin[];
}

export interface MonitoringAlertDraft {
  severity: "low" | "medium" | "high";
  title: string;
  explanation: string;
  recommendedAction: string;
}

export interface MonitoringAnalysis {
  rowCount: number;
  overallDriftScore: number;
  featureDrift: FeatureDriftResult[];
  predictionDistribution: Record<string, number>;
  referencePredictionDistribution: Record<string, number>;
  predictionDriftScore: number;
  performance: EvaluationMetrics | null;
  performanceStatus: "not_measured" | "stable" | "degraded";
  predictions: {
    rowIndex: number;
    predictedLabel: string;
    actualLabel: string | null;
    confidence: number;
    probabilities: Record<string, number>;
  }[];
  alerts: MonitoringAlertDraft[];
}

export interface MonitoringThresholds {
  driftThreshold: number;
  predictionDriftThreshold: number;
  performanceDropThreshold: number;
}

function distributions(values: string[]): Map<string, number> {
  const result = new Map<string, number>();
  for (const value of values) result.set(value, (result.get(value) ?? 0) + 1);
  return result;
}

function totalVariation(
  left: Record<string, number>,
  right: Record<string, number>,
  classes: string[],
): number {
  const leftTotal = Object.values(left).reduce((sum, value) => sum + value, 0);
  const rightTotal = Object.values(right).reduce((sum, value) => sum + value, 0);
  if (!leftTotal || !rightTotal) return 0;
  return (
    classes.reduce(
      (sum, category) =>
        sum + Math.abs((left[category] ?? 0) / leftTotal - (right[category] ?? 0) / rightTotal),
      0,
    ) / 2
  );
}

function ksStatistic(reference: number[], current: number[]): number {
  const values = [...new Set([...reference, ...current])].sort((a, b) => a - b);
  let maxDistance = 0;
  for (const value of values) {
    const leftCdf = reference.filter((item) => item <= value).length / reference.length;
    const rightCdf = current.filter((item) => item <= value).length / current.length;
    maxDistance = Math.max(maxDistance, Math.abs(leftCdf - rightCdf));
  }
  return maxDistance;
}

function approximateKsPValue(score: number, leftSize: number, rightSize: number): number {
  const effectiveSize = Math.sqrt((leftSize * rightSize) / (leftSize + rightSize));
  const lambda = (effectiveSize + 0.12 + 0.11 / effectiveSize) * score;
  if (lambda <= 0) return 1;
  let sum = 0;
  for (let index = 1; index <= 100; index += 1) {
    const term = Math.exp(-2 * index * index * lambda * lambda);
    sum += index % 2 === 1 ? term : -term;
    if (term < 1e-12) break;
  }
  return Math.max(0, Math.min(1, 2 * sum));
}

function logGamma(value: number): number {
  const coefficients = [
    676.5203681218851,
    -1259.1392167224028,
    771.3234287776531,
    -176.6150291621406,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
  ];
  if (value < 0.5) {
    return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * value)) - logGamma(1 - value);
  }
  const shifted = value - 1;
  let series = 0.99999999999980993;
  coefficients.forEach((coefficient, index) => {
    series += coefficient / (shifted + index + 1);
  });
  const t = shifted + coefficients.length - 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (shifted + 0.5) * Math.log(t) - t + Math.log(series);
}

function gammaQ(shape: number, x: number): number {
  if (x <= 0) return 1;
  if (shape <= 0) return 0;
  const logFactor = shape * Math.log(x) - x - logGamma(shape);
  if (x < shape + 1) {
    let term = 1 / shape;
    let sum = term;
    let denominator = shape;
    for (let index = 1; index < 200; index += 1) {
      denominator += 1;
      term *= x / denominator;
      sum += term;
      if (Math.abs(term) < Math.abs(sum) * 1e-14) break;
    }
    const lower = sum * Math.exp(logFactor);
    return Math.max(0, Math.min(1, 1 - lower));
  }

  let b = x + 1 - shape;
  let c = 1 / 1e-30;
  let d = 1 / b;
  let result = d;
  for (let index = 1; index < 200; index += 1) {
    const coefficient = -index * (index - shape);
    b += 2;
    d = coefficient * d + b;
    if (Math.abs(d) < 1e-30) d = 1e-30;
    c = b + coefficient / c;
    if (Math.abs(c) < 1e-30) c = 1e-30;
    d = 1 / d;
    const delta = d * c;
    result *= delta;
    if (Math.abs(delta - 1) < 1e-14) break;
  }
  return Math.max(0, Math.min(1, Math.exp(logFactor) * result));
}

function chiSquarePValue(statistic: number, degreesOfFreedom: number): number {
  return gammaQ(degreesOfFreedom / 2, statistic / 2);
}

function numericBins(
  reference: number[],
  current: number[],
  referenceMissing: number,
  currentMissing: number,
): [DistributionBin[], DistributionBin[]] {
  const sorted = [...reference].sort((a, b) => a - b);
  const cutPoints = [0.2, 0.4, 0.6, 0.8]
    .map((quantile) => sorted[Math.min(sorted.length - 1, Math.floor(quantile * sorted.length))])
    .filter((value, index, all) => index === 0 || value > all[index - 1]);
  const labels = [
    `≤ ${cutPoints[0]?.toPrecision(4) ?? "—"}`,
    ...cutPoints.slice(1).map((value, index) => `${cutPoints[index].toPrecision(4)} – ${value.toPrecision(4)}`),
    `> ${cutPoints.at(-1)?.toPrecision(4) ?? "—"}`,
  ];
  const countValues = (values: number[], missing: number) => {
    const counts = Array.from({ length: labels.length }, () => 0);
    for (const value of values) {
      let index = 0;
      while (index < cutPoints.length && value > cutPoints[index]) index += 1;
      counts[index] += 1;
    }
    return [
      ...labels.map((label, index) => ({ label, count: counts[index] })),
      { label: "(missing)", count: missing },
    ];
  };
  return [countValues(reference, referenceMissing), countValues(current, currentMissing)];
}

function categoricalBins(
  reference: string[],
  current: string[],
): [DistributionBin[], DistributionBin[], Record<string, number>, Record<string, number>, string[]] {
  const referenceCounts = distributions(reference);
  const currentCounts = distributions(current);
  const candidates = [...new Set([...referenceCounts.keys(), ...currentCounts.keys()])]
    .sort(
      (a, b) =>
        (referenceCounts.get(b) ?? 0) + (currentCounts.get(b) ?? 0) -
        ((referenceCounts.get(a) ?? 0) + (currentCounts.get(a) ?? 0)),
    );
  const kept = candidates.length > 8 ? candidates.slice(0, 7) : candidates;
  const categories = candidates.length > kept.length ? [...kept, "Other"] : kept;
  const fold = (counts: Map<string, number>, total: number) => {
    const result: Record<string, number> = {};
    for (const category of kept) result[category] = counts.get(category) ?? 0;
    if (categories.includes("Other")) {
      result.Other = total - kept.reduce((sum, category) => sum + (counts.get(category) ?? 0), 0);
    }
    return result;
  };
  const foldedReference = fold(referenceCounts, reference.length);
  const foldedCurrent = fold(currentCounts, current.length);
  return [
    categories.map((label) => ({ label, count: foldedReference[label] ?? 0 })),
    categories.map((label) => ({ label, count: foldedCurrent[label] ?? 0 })),
    foldedReference,
    foldedCurrent,
    categories,
  ];
}

function featureComparison(
  feature: ColumnSummary,
  referenceRows: DataRow[],
  currentRows: DataRow[],
  threshold: number,
): FeatureDriftResult {
  const numeric =
    feature.type === "numeric"
      ? ([referenceRows, currentRows].map((rows) =>
          rows
            .map((row) => row[feature.name])
            .filter((value): value is number => typeof value === "number" && Number.isFinite(value)),
        ) as [number[], number[]])
      : null;

  let score = 0;
  let pValue: number | null = null;
  let method: FeatureDriftResult["method"] = "unavailable";
  let referenceDistribution: DistributionBin[] = [];
  let currentDistribution: DistributionBin[] = [];

  if (numeric) {
    const [referenceValues, currentValues] = numeric;
    const referenceMissing = referenceRows.length - referenceValues.length;
    const currentMissing = currentRows.length - currentValues.length;
    [referenceDistribution, currentDistribution] = numericBins(
      referenceValues,
      currentValues,
      referenceMissing,
      currentMissing,
    );
    const missingRateShift = Math.abs(
      referenceMissing / referenceRows.length - currentMissing / currentRows.length,
    );
    score = missingRateShift;
    if (referenceValues.length >= 8 && currentValues.length >= 8) {
      const distributionShift = ksStatistic(referenceValues, currentValues);
      score = Math.max(score, distributionShift);
      pValue = approximateKsPValue(distributionShift, referenceValues.length, currentValues.length);
      method = "kolmogorov_smirnov";
    }
  } else {
    const referenceValues = referenceRows.map((row) =>
      row[feature.name] === null || row[feature.name] === undefined ? "(missing)" : String(row[feature.name]),
    );
    const currentValues = currentRows.map((row) =>
      row[feature.name] === null || row[feature.name] === undefined ? "(missing)" : String(row[feature.name]),
    );
    const [
      referenceBins,
      currentBins,
      referenceCounts,
      currentCounts,
      categories,
    ] = categoricalBins(referenceValues, currentValues);
    referenceDistribution = referenceBins;
    currentDistribution = currentBins;
    score = totalVariation(referenceCounts, currentCounts, categories);

    if (referenceValues.length >= 8 && currentValues.length >= 8 && categories.length >= 2) {
      const chiSquare = categories.reduce((sum, category) => {
        const combined = (referenceCounts[category] ?? 0) + (currentCounts[category] ?? 0);
        const expected = combined / 2;
        if (expected < 5) return Number.NaN;
        const leftDifference = (referenceCounts[category] ?? 0) - expected;
        const rightDifference = (currentCounts[category] ?? 0) - expected;
        return sum + (leftDifference * leftDifference) / expected + (rightDifference * rightDifference) / expected;
      }, 0);
      if (Number.isFinite(chiSquare)) {
        pValue = chiSquarePValue(chiSquare, categories.length - 1);
        method = "chi_square";
      }
    }
  }

  const status =
    method === "unavailable"
      ? "unavailable"
      : score >= Math.min(1, threshold * 2)
        ? "high"
        : score >= threshold
          ? "warning"
          : "low";

  return {
    feature: feature.name,
    type: feature.type,
    score,
    pValue,
    method,
    status,
    referenceDistribution,
    currentDistribution,
  };
}

function countDistribution(values: string[], classes: string[]): Record<string, number> {
  return Object.fromEntries(
    classes.map((className) => [className, values.filter((value) => value === className).length]),
  );
}

export function analyzeMonitoring(args: {
  model: SerializedModel;
  referenceRows: DataRow[];
  incomingRows: DataRow[];
  columns: ColumnSummary[];
  targetColumn: string;
  baselineMetrics: EvaluationMetrics;
  thresholds: MonitoringThresholds;
}): MonitoringAnalysis {
  const {
    model,
    referenceRows,
    incomingRows,
    columns,
    targetColumn,
    baselineMetrics,
    thresholds,
  } = args;
  const featureDrift = model.features.map((featureSpec) => {
    const feature = columns.find((column) => column.name === featureSpec.name);
    if (!feature) {
      return {
        feature: featureSpec.name,
        type: featureSpec.type,
        score: 0,
        pValue: null,
        method: "unavailable" as const,
        status: "unavailable" as const,
        referenceDistribution: [],
        currentDistribution: [],
      };
    }
    return featureComparison(feature, referenceRows, incomingRows, thresholds.driftThreshold);
  });
  const comparable = featureDrift.filter((feature) => feature.method !== "unavailable");
  const overallDriftScore = comparable.length
    ? comparable.reduce((sum, feature) => sum + feature.score, 0) / comparable.length
    : 0;

  const predictions = incomingRows.map((row, rowIndex) => {
    const prediction = predictRow(model, row);
    const actual = isUsableLabel(row[targetColumn]) ? String(row[targetColumn]) : null;
    return { rowIndex, ...prediction, actualLabel: actual };
  });
  const referencePredictions = referenceRows.map((row) => predictRow(model, row).predictedLabel);
  const predictionLabels = predictions.map((prediction) => prediction.predictedLabel);
  const predictionDistribution = countDistribution(predictionLabels, model.classes);
  const referencePredictionDistribution = countDistribution(referencePredictions, model.classes);
  const predictionDriftScore = totalVariation(
    predictionDistribution,
    referencePredictionDistribution,
    model.classes,
  );

  const labeled = predictions.filter(
    (prediction) =>
      prediction.actualLabel !== null && model.classes.includes(prediction.actualLabel),
  );
  const performance =
    labeled.length > 0
      ? evaluateLabels(
          labeled.map((prediction) => prediction.actualLabel as string),
          labeled.map((prediction) => prediction.predictedLabel),
          model.classes,
        )
      : null;
  const performanceStatus = !performance
    ? "not_measured"
    : performance.accuracy <= baselineMetrics.accuracy - thresholds.performanceDropThreshold
      ? "degraded"
      : "stable";

  const alerts: MonitoringAlertDraft[] = [];
  const changedFeatures = featureDrift.filter(
    (feature) => feature.method !== "unavailable" && feature.score >= thresholds.driftThreshold,
  );
  if (changedFeatures.length > 0) {
    const highCount = changedFeatures.filter((feature) => feature.status === "high").length;
    alerts.push({
      severity: highCount > 0 ? "high" : "medium",
      title: `${changedFeatures.length} feature${changedFeatures.length === 1 ? "" : "s"} show drift`,
      explanation: `${changedFeatures
        .map(
          (feature) =>
            `${feature.feature} (score ${feature.score.toFixed(3)}${feature.pValue === null ? "" : `, p≈${feature.pValue.toFixed(3)}`})`,
        )
        .join("; ")} crossed the configured drift threshold of ${thresholds.driftThreshold.toFixed(2)}. This indicates input distribution change, not proof of reduced model accuracy.`,
      recommendedAction:
        "Check incoming schema and data quality, investigate the affected feature distributions, and compare model performance once labels are available.",
    });
  }
  const nearThresholdFeatures = featureDrift.filter(
    (feature) =>
      feature.method !== "unavailable" &&
      feature.pValue !== null &&
      feature.pValue < 0.05 &&
      feature.score >= thresholds.driftThreshold * 0.75 &&
      feature.score < thresholds.driftThreshold,
  );
  if (nearThresholdFeatures.length > 0) {
    alerts.push({
      severity: "low",
      title: `${nearThresholdFeatures.length} feature${nearThresholdFeatures.length === 1 ? "" : "s"} show early drift evidence`,
      explanation: `${nearThresholdFeatures
        .map((feature) => `${feature.feature} (score ${feature.score.toFixed(3)}, p≈${feature.pValue?.toFixed(3)})`)
        .join("; ")} show statistically detectable distribution changes near the configured threshold of ${thresholds.driftThreshold.toFixed(2)}, but have not crossed it.`,
      recommendedAction:
        "Review the affected feature's data quality and continue monitoring; confirm the change across additional incoming batches before taking action.",
    });
  }
  if (predictionDriftScore >= thresholds.predictionDriftThreshold) {
    alerts.push({
      severity:
        predictionDriftScore >= Math.min(1, thresholds.predictionDriftThreshold * 2)
          ? "high"
          : "medium",
      title: "Predicted class distribution shifted",
      explanation: `Prediction distribution total variation is ${predictionDriftScore.toFixed(3)}, above the configured threshold of ${thresholds.predictionDriftThreshold.toFixed(2)}. This is a change in outputs, not a direct measurement of accuracy.`,
      recommendedAction:
        "Review class proportions and representative incoming records, then validate results against ground-truth labels when available.",
    });
  }
  if (performanceStatus === "degraded" && performance) {
    alerts.push({
      severity: "high",
      title: "Measured performance has degraded",
      explanation: `Accuracy on ${labeled.length} labeled incoming rows is ${(performance.accuracy * 100).toFixed(1)}%, compared with ${(baselineMetrics.accuracy * 100).toFixed(1)}% on the held-out reference set. The configured drop threshold is ${(thresholds.performanceDropThreshold * 100).toFixed(1)} percentage points.`,
      recommendedAction:
        "Verify label quality and incoming data joins, inspect per-class metrics, and consider retraining only after confirming the degradation.",
    });
  }

  return {
    rowCount: incomingRows.length,
    overallDriftScore,
    featureDrift,
    predictionDistribution,
    referencePredictionDistribution,
    predictionDriftScore,
    performance,
    performanceStatus,
    predictions,
    alerts,
  };
}

