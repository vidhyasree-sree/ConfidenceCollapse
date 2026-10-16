import type { ColumnSummary, DataRow } from "./csv";
import { InputError } from "./errors";

type FeatureValue = number | string;
type Algorithm = "decision_tree" | "random_forest";

interface FeatureSpec {
  name: string;
  type: "numeric" | "categorical";
  imputation: FeatureValue;
}

export type TreeNode =
  | { kind: "leaf"; probabilities: number[] }
  | {
      kind: "split";
      featureIndex: number;
      mode: "numeric" | "categorical";
      threshold?: number;
      category?: string;
      left: TreeNode;
      right: TreeNode;
    };

export interface SerializedModel {
  algorithm: Algorithm;
  classes: string[];
  features: FeatureSpec[];
  trees: TreeNode[];
}

export interface EvaluationMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  averaging: "binary" | "macro";
  confusionMatrix: number[][];
  perClass: {
    label: string;
    precision: number;
    recall: number;
    f1: number;
    support: number;
  }[];
}

export interface FeatureImportance {
  feature: string;
  importance: number;
}

export interface ScoredRow {
  predictedLabel: string;
  actualLabel: string | null;
  confidence: number;
  probabilities: Record<string, number>;
}

export interface TrainedModelData {
  serialized: SerializedModel;
  classes: string[];
  metrics: EvaluationMetrics;
  featureImportances: FeatureImportance[];
  evaluation: (ScoredRow & { rowIndex: number })[];
  rowCount: number;
  featureCount: number;
}

function randomGenerator(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffled<T>(values: T[], random: () => number): T[] {
  const copy = [...values];
  for (let index = copy.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(random() * (index + 1));
    [copy[index], copy[swapIndex]] = [copy[swapIndex], copy[index]];
  }
  return copy;
}

function labelOf(value: unknown): string | null {
  if (value === null || value === undefined || String(value).trim() === "") return null;
  return String(value);
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle];
}

function gini(counts: number[]): number {
  const total = counts.reduce((sum, count) => sum + count, 0);
  if (total === 0) return 0;
  return 1 - counts.reduce((sum, count) => sum + (count / total) ** 2, 0);
}

function toFeatures(row: DataRow, features: FeatureSpec[]): FeatureValue[] {
  return features.map((feature) => {
    const raw = row[feature.name];
    if (raw === null || raw === undefined || String(raw).trim() === "") {
      return feature.imputation;
    }
    if (feature.type === "numeric") {
      const numeric = typeof raw === "number" ? raw : Number(raw);
      return Number.isFinite(numeric) ? numeric : feature.imputation;
    }
    return String(raw);
  });
}

function classCounts(labels: number[], classCount: number): number[] {
  const counts = Array.from({ length: classCount }, () => 0);
  for (const label of labels) counts[label] += 1;
  return counts;
}

function findBestSplit(
  rows: number[],
  values: FeatureValue[][],
  labels: number[],
  featureIndices: number[],
  features: FeatureSpec[],
  classCount: number,
): {
  featureIndex: number;
  mode: "numeric" | "categorical";
  threshold?: number;
  category?: string;
  gain: number;
} | null {
  const parentCounts = classCounts(rows.map((index) => labels[index]), classCount);
  const parentImpurity = gini(parentCounts);
  let best: ReturnType<typeof findBestSplit> = null;

  for (const featureIndex of featureIndices) {
    const feature = features[featureIndex];
    if (feature.type === "numeric") {
      const sorted = rows
        .map((rowIndex) => ({
          value: values[rowIndex][featureIndex] as number,
          label: labels[rowIndex],
        }))
        .sort((a, b) => a.value - b.value);
      const boundaries: number[] = [];
      for (let index = 1; index < sorted.length; index += 1) {
        if (sorted[index - 1].value !== sorted[index].value) boundaries.push(index);
      }
      if (boundaries.length === 0) continue;
      const stride = Math.max(1, Math.ceil(boundaries.length / 24));
      const leftCounts = Array.from({ length: classCount }, () => 0);
      let boundaryIndex = 0;
      for (let index = 0; index < sorted.length - 1; index += 1) {
        leftCounts[sorted[index].label] += 1;
        const position = index + 1;
        if (position !== boundaries[boundaryIndex]) continue;
        const selected = boundaryIndex % stride === 0 || boundaryIndex === boundaries.length - 1;
        boundaryIndex += 1;
        if (!selected || position < 2 || sorted.length - position < 2) continue;
        const rightCounts = parentCounts.map((count, label) => count - leftCounts[label]);
        const weightedImpurity =
          (position / sorted.length) * gini(leftCounts) +
          ((sorted.length - position) / sorted.length) * gini(rightCounts);
        const gain = parentImpurity - weightedImpurity;
        if (!best || gain > best.gain) {
          best = {
            featureIndex,
            mode: "numeric",
            threshold: (sorted[position - 1].value + sorted[position].value) / 2,
            gain,
          };
        }
      }
    } else {
      const countsByCategory = new Map<string, number[]>();
      for (const rowIndex of rows) {
        const value = String(values[rowIndex][featureIndex]);
        const counts = countsByCategory.get(value) ?? Array.from({ length: classCount }, () => 0);
        counts[labels[rowIndex]] += 1;
        countsByCategory.set(value, counts);
      }
      const candidates = [...countsByCategory.entries()]
        .sort(
          (a, b) =>
            b[1].reduce((sum, count) => sum + count, 0) -
            a[1].reduce((sum, count) => sum + count, 0),
        )
        .slice(0, 32);
      for (const [category, leftCounts] of candidates) {
        const leftSize = leftCounts.reduce((sum, count) => sum + count, 0);
        const rightSize = rows.length - leftSize;
        if (leftSize < 2 || rightSize < 2) continue;
        const rightCounts = parentCounts.map((count, label) => count - leftCounts[label]);
        const weightedImpurity =
          (leftSize / rows.length) * gini(leftCounts) +
          (rightSize / rows.length) * gini(rightCounts);
        const gain = parentImpurity - weightedImpurity;
        if (!best || gain > best.gain) {
          best = { featureIndex, mode: "categorical", category, gain };
        }
      }
    }
  }

  return best && best.gain > 1e-10 ? best : null;
}

function fitTree(
  trainingRows: number[],
  values: FeatureValue[][],
  labels: number[],
  features: FeatureSpec[],
  classes: string[],
  maxDepth: number,
  random: () => number,
  randomFeatureCount: number,
): { tree: TreeNode; importance: number[] } {
  const importance = Array.from({ length: features.length }, () => 0);
  const build = (rows: number[], depth: number): TreeNode => {
    const counts = classCounts(rows.map((index) => labels[index]), classes.length);
    const total = counts.reduce((sum, count) => sum + count, 0);
    const probabilities = counts.map((count) => count / total);
    if (
      depth >= maxDepth ||
      rows.length < 4 ||
      counts.filter((count) => count > 0).length < 2 ||
      features.length === 0
    ) {
      return { kind: "leaf", probabilities };
    }

    const allFeatures = Array.from({ length: features.length }, (_, index) => index);
    const chosenFeatures =
      randomFeatureCount >= allFeatures.length
        ? allFeatures
        : shuffled(allFeatures, random).slice(0, randomFeatureCount);
    const split = findBestSplit(rows, values, labels, chosenFeatures, features, classes.length);
    if (!split) return { kind: "leaf", probabilities };

    const leftRows: number[] = [];
    const rightRows: number[] = [];
    for (const rowIndex of rows) {
      const value = values[rowIndex][split.featureIndex];
      const goesLeft =
        split.mode === "numeric"
          ? (value as number) <= (split.threshold as number)
          : String(value) === split.category;
      (goesLeft ? leftRows : rightRows).push(rowIndex);
    }
    if (leftRows.length < 2 || rightRows.length < 2) {
      return { kind: "leaf", probabilities };
    }
    importance[split.featureIndex] += split.gain * rows.length;
    return {
      kind: "split",
      featureIndex: split.featureIndex,
      mode: split.mode,
      ...(split.threshold === undefined ? {} : { threshold: split.threshold }),
      ...(split.category === undefined ? {} : { category: split.category }),
      left: build(leftRows, depth + 1),
      right: build(rightRows, depth + 1),
    };
  };

  return { tree: build(trainingRows, 0), importance };
}

function predictTree(tree: TreeNode, row: FeatureValue[]): number[] {
  let node = tree;
  while (node.kind === "split") {
    const value = row[node.featureIndex];
    const left =
      node.mode === "numeric"
        ? (value as number) <= (node.threshold as number)
        : String(value) === node.category;
    node = left ? node.left : node.right;
  }
  return node.probabilities;
}

function scoreProbabilities(
  probabilities: number[],
  classes: string[],
): Omit<ScoredRow, "actualLabel"> {
  let bestIndex = 0;
  for (let index = 1; index < probabilities.length; index += 1) {
    if (probabilities[index] > probabilities[bestIndex]) bestIndex = index;
  }
  return {
    predictedLabel: classes[bestIndex],
    confidence: probabilities[bestIndex] ?? 0,
    probabilities: Object.fromEntries(
      classes.map((className, index) => [className, probabilities[index] ?? 0]),
    ),
  };
}

export function predictRow(model: SerializedModel, row: DataRow): Omit<ScoredRow, "actualLabel"> {
  const values = toFeatures(row, model.features);
  const probabilities = Array.from({ length: model.classes.length }, () => 0);
  for (const tree of model.trees) {
    const contribution = predictTree(tree, values);
    contribution.forEach((value, index) => {
      probabilities[index] += value / model.trees.length;
    });
  }
  return scoreProbabilities(probabilities, model.classes);
}

export function evaluateLabels(
  actualLabels: string[],
  predictedLabels: string[],
  classes: string[],
): EvaluationMetrics {
  const classIndex = new Map(classes.map((label, index) => [label, index]));
  const matrix = Array.from({ length: classes.length }, () =>
    Array.from({ length: classes.length }, () => 0),
  );
  actualLabels.forEach((actual, index) => {
    const actualIndex = classIndex.get(actual);
    const predictedIndex = classIndex.get(predictedLabels[index]);
    if (actualIndex !== undefined && predictedIndex !== undefined) {
      matrix[actualIndex][predictedIndex] += 1;
    }
  });
  const perClass = classes.map((label, index) => {
    const truePositive = matrix[index][index];
    const support = matrix[index].reduce((sum, value) => sum + value, 0);
    const predicted = matrix.reduce((sum, row) => sum + row[index], 0);
    const precision = predicted > 0 ? truePositive / predicted : 0;
    const recall = support > 0 ? truePositive / support : 0;
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
    return { label, precision, recall, f1, support };
  });
  const total = actualLabels.length;
  const correct = matrix.reduce((sum, row, index) => sum + row[index], 0);
  const averaging = classes.length === 2 ? "binary" : "macro";
  const includedMetrics = averaging === "binary" ? [perClass[1]] : perClass;
  const average = (key: "precision" | "recall" | "f1") =>
    includedMetrics.length
      ? includedMetrics.reduce((sum, metric) => sum + metric[key], 0) / includedMetrics.length
      : 0;
  return {
    accuracy: total > 0 ? correct / total : 0,
    precision: average("precision"),
    recall: average("recall"),
    f1: average("f1"),
    averaging,
    confusionMatrix: matrix,
    perClass,
  };
}

export function trainClassifier(args: {
  rows: DataRow[];
  columns: ColumnSummary[];
  targetColumn: string;
  algorithm: Algorithm;
  testSize: number;
  randomSeed: number;
  maxDepth: number;
  estimators: number;
}): TrainedModelData {
  const { rows, columns, targetColumn, algorithm, testSize, randomSeed, maxDepth, estimators } =
    args;
  if (!columns.some((column) => column.name === targetColumn)) {
    throw new InputError("Choose a target column that exists in the dataset.");
  }
  const featureColumns = columns.filter((column) => column.name !== targetColumn);
  if (featureColumns.length === 0) throw new InputError("At least one feature column is required.");

  const labeledRows = rows
    .map((row) => ({ row, label: labelOf(row[targetColumn]) }))
    .filter((item): item is { row: DataRow; label: string } => item.label !== null);
  if (labeledRows.length < 10) {
    throw new InputError("At least 10 rows with a non-empty target value are required.");
  }
  const groups = new Map<string, typeof labeledRows>();
  for (const item of labeledRows) {
    groups.set(item.label, [...(groups.get(item.label) ?? []), item]);
  }
  if (groups.size < 2) throw new InputError("Classification needs at least two target classes.");
  if ([...groups.values()].some((group) => group.length < 2)) {
    throw new InputError("Each target class needs at least two rows for a train/test split.");
  }

  const random = randomGenerator(randomSeed);
  const training: typeof labeledRows = [];
  const testing: typeof labeledRows = [];
  for (const group of groups.values()) {
    const shuffledGroup = shuffled(group, random);
    const testCount = Math.min(
      shuffledGroup.length - 1,
      Math.max(1, Math.round(shuffledGroup.length * testSize)),
    );
    testing.push(...shuffledGroup.slice(0, testCount));
    training.push(...shuffledGroup.slice(testCount));
  }

  const classes = [...groups.keys()].sort((a, b) => a.localeCompare(b));
  const classIndices = new Map(classes.map((label, index) => [label, index]));
  const labels = labeledRows.map((item) => classIndices.get(item.label) as number);
  const rowIndexes = new Map(labeledRows.map((item, index) => [item.row, index]));
  const originalRowIndexes = new Map(rows.map((row, index) => [row, index]));
  const trainingIndexes = training.map((item) => rowIndexes.get(item.row) as number);
  const testingIndexes = testing.map((item) => rowIndexes.get(item.row) as number);

  const featureSpecs: FeatureSpec[] = featureColumns.map((column) => {
    const values = training
      .map(({ row }) => row[column.name])
      .filter((value) => value !== null && value !== undefined);
    if (column.type === "numeric") {
      const numericValues = values
        .map((value) => (typeof value === "number" ? value : Number(value)))
        .filter(Number.isFinite);
      return { name: column.name, type: "numeric", imputation: median(numericValues) };
    }
    const frequencies = new Map<string, number>();
    values.forEach((value) => frequencies.set(String(value), (frequencies.get(String(value)) ?? 0) + 1));
    const mostCommon = [...frequencies.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ?? "__MISSING__";
    return { name: column.name, type: "categorical", imputation: mostCommon };
  });

  const values = labeledRows.map(({ row }) => toFeatures(row, featureSpecs));
  const treeCount = algorithm === "random_forest" ? Math.min(estimators, 100) : 1;
  const randomFeatureCount =
    algorithm === "random_forest" ? Math.max(1, Math.floor(Math.sqrt(featureSpecs.length))) : featureSpecs.length;
  const allImportances = Array.from({ length: featureSpecs.length }, () => 0);
  const trees: TreeNode[] = [];

  for (let treeIndex = 0; treeIndex < treeCount; treeIndex += 1) {
    const bootstrap =
      algorithm === "random_forest"
        ? Array.from(
            { length: trainingIndexes.length },
            () => trainingIndexes[Math.floor(random() * trainingIndexes.length)],
          )
        : trainingIndexes;
    const fitted = fitTree(
      bootstrap,
      values,
      labels,
      featureSpecs,
      classes,
      maxDepth,
      random,
      randomFeatureCount,
    );
    trees.push(fitted.tree);
    fitted.importance.forEach((value, index) => {
      allImportances[index] += value;
    });
  }

  const model: SerializedModel = {
    algorithm,
    classes,
    features: featureSpecs,
    trees,
  };
  const evaluation = testingIndexes.map((index) => {
    const predictions = trees.map((tree) => predictTree(tree, values[index]));
    const averaged = classes.map(
      (_, classIndex) =>
        predictions.reduce((sum, prediction) => sum + prediction[classIndex], 0) / trees.length,
    );
    return {
      ...scoreProbabilities(averaged, classes),
      actualLabel: classes[labels[index]],
      rowIndex: originalRowIndexes.get(labeledRows[index].row) as number,
    };
  });
  const metrics = evaluateLabels(
    evaluation.map((item) => item.actualLabel as string),
    evaluation.map((item) => item.predictedLabel),
    classes,
  );
  const importanceTotal = allImportances.reduce((sum, value) => sum + value, 0);
  const featureImportances = featureSpecs
    .map((feature, index) => ({
      feature: feature.name,
      importance: importanceTotal > 0 ? allImportances[index] / importanceTotal : 0,
    }))
    .sort((a, b) => b.importance - a.importance);

  return {
    serialized: model,
    classes,
    metrics,
    featureImportances,
    evaluation,
    rowCount: labeledRows.length,
    featureCount: featureSpecs.length,
  };
}

export function isUsableLabel(value: unknown): value is string {
  return labelOf(value) !== null;
}

