import { InputError } from "./errors";

export type CellValue = string | number | boolean | null;
export type DataRow = Record<string, CellValue>;

export interface ColumnSummary {
  name: string;
  type: "numeric" | "categorical";
  missingValues: number;
  uniqueValues: number;
  mean: number | null;
}

export interface ParsedDataset {
  columns: ColumnSummary[];
  rows: DataRow[];
  rowCount: number;
  columnCount: number;
  missingCellCount: number;
}

const MAX_FILE_BYTES = 10 * 1024 * 1024;
const MAX_ROWS = 20_000;
const MAX_COLUMNS = 100;

function decodeCsv(source: string): string[][] {
  const records: string[][] = [];
  let record: string[] = [];
  let field = "";
  let quoted = false;

  for (let index = 0; index < source.length; index += 1) {
    const char = source[index];
    if (quoted) {
      if (char === '"') {
        if (source[index + 1] === '"') {
          field += '"';
          index += 1;
        } else {
          quoted = false;
        }
      } else {
        field += char;
      }
      continue;
    }

    if (char === '"' && field.length === 0) {
      quoted = true;
    } else if (char === ",") {
      record.push(field);
      field = "";
    } else if (char === "\n" || char === "\r") {
      record.push(field);
      field = "";
      if (record.some((cell) => cell.trim() !== "")) records.push(record);
      record = [];
      if (char === "\r" && source[index + 1] === "\n") index += 1;
    } else {
      field += char;
    }
  }

  if (quoted) throw new InputError("CSV contains an unclosed quoted field.");
  record.push(field);
  if (record.some((cell) => cell.trim() !== "")) records.push(record);
  return records;
}

export function parseCsv(source: string): ParsedDataset {
  if (Buffer.byteLength(source, "utf8") > MAX_FILE_BYTES) {
    throw new InputError("CSV is larger than the 10 MB upload limit.");
  }

  const records = decodeCsv(source.replace(/^\uFEFF/, ""));
  if (records.length < 2) {
    throw new InputError("CSV needs a header row and at least one data row.");
  }

  const headers = records[0].map((cell) => cell.trim());
  if (headers.length > MAX_COLUMNS) {
    throw new InputError(`CSV exceeds the ${MAX_COLUMNS}-column limit.`);
  }
  if (headers.some((header) => !header)) {
    throw new InputError("Every CSV column needs a non-empty header.");
  }
  if (new Set(headers).size !== headers.length) {
    throw new InputError("CSV column names must be unique.");
  }
  if (records.length - 1 > MAX_ROWS) {
    throw new InputError(`CSV exceeds the ${MAX_ROWS.toLocaleString()}-row limit.`);
  }

  const rawRows = records.slice(1).map((record, rowIndex) => {
    if (record.length > headers.length) {
      throw new InputError(`Row ${rowIndex + 2} has more cells than the header.`);
    }
    return headers.map((_, columnIndex) => {
      const raw = record[columnIndex]?.trim() ?? "";
      return raw === "" ? null : raw;
    });
  });

  const columns: ColumnSummary[] = headers.map((name, index) => {
    const values = rawRows.map((row) => row[index]);
    const nonMissing = values.filter((value) => value !== null);
    const numeric = nonMissing.length > 0 && nonMissing.every((value) => Number.isFinite(Number(value)));
    const numbers = numeric ? nonMissing.map(Number) : [];
    return {
      name,
      type: numeric ? "numeric" : "categorical",
      missingValues: values.length - nonMissing.length,
      uniqueValues: new Set(nonMissing.map((value) => String(value))).size,
      mean: numeric
        ? numbers.reduce((sum, value) => sum + value, 0) / numbers.length
        : null,
    };
  });

  const numericColumns = new Set(
    columns.filter((column) => column.type === "numeric").map((column) => column.name),
  );
  const rows = rawRows.map((values) =>
    Object.fromEntries(
      headers.map((header, index) => {
        const value = values[index];
        return [header, value === null ? null : numericColumns.has(header) ? Number(value) : value];
      }),
    ),
  );
  const missingCellCount = columns.reduce((sum, column) => sum + column.missingValues, 0);

  return {
    rows,
    columns,
    rowCount: rows.length,
    columnCount: columns.length,
    missingCellCount,
  };
}

export function toDatasetSummary<T extends {
  id: string;
  name: string;
  purpose: string;
  targetColumn: string | null;
  columns: unknown;
  rowCount: number;
  columnCount: number;
  missingCellCount: number;
  createdAt: Date;
}>(dataset: T) {
  return {
    id: dataset.id,
    name: dataset.name,
    purpose: dataset.purpose as "reference" | "incoming",
    targetColumn: dataset.targetColumn,
    columns: dataset.columns as ColumnSummary[],
    rowCount: dataset.rowCount,
    columnCount: dataset.columnCount,
    missingCellCount: dataset.missingCellCount,
    createdAt: dataset.createdAt.toISOString(),
  };
}
