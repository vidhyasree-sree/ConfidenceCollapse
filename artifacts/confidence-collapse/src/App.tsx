import { type ReactNode, useEffect, useMemo, useState } from 'react';
import { QueryClient, QueryClientProvider, useQueryClient } from '@tanstack/react-query';
import { ErrorBoundary } from '@/components/error-boundary';
import { Toaster } from '@/components/ui/toaster';
import { TooltipProvider } from '@/components/ui/tooltip';
import NotFound from '@/pages/not-found';
import { Link, Route, Switch, useLocation, useParams, Router as WouterRouter } from 'wouter';
import {
  Activity, ArrowDownToLine, ArrowRight, BarChart3, Bell, Check, ChevronRight,
  CircleHelp, Database, FileText, Gauge, Layers3, LineChart, RefreshCw,
  Moon, Settings2, ShieldCheck, SlidersHorizontal, Sparkles, Sun, Trash2, Upload, X,
} from 'lucide-react';
import {
  Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart as RechartsLineChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import {
  getGetDashboardSummaryQueryKey, getGetDatasetQueryKey, getGetModelQueryKey, getGetModelReportQueryKey,
  getGetSettingsQueryKey, getListAlertsQueryKey, getListDatasetsQueryKey, getListModelsQueryKey,
  getListMonitoringRunsQueryKey, getListPredictionsQueryKey, useDeleteDataset, useGetDashboardSummary,
  useGetDataset, useGetModel, useGetModelReport, useGetSettings, useHealthCheck, useListAlerts,
  useListDatasets, useListModels, useListMonitoringRuns, useListPredictions, usePredictWithModel, useRunMonitoring,
  useTrainModel, useUpdateAlert, useUpdateSettings, useUploadDataset,
} from '@workspace/api-client-react';

const queryClient = new QueryClient();
type PageProps = { children: ReactNode };

const pages = [
  { href: '/', label: 'Overview', icon: Gauge },
  { href: '/datasets', label: 'Datasets', icon: Database },
  { href: '/models', label: 'Models', icon: Layers3 },
  { href: '/monitoring', label: 'Monitoring', icon: Activity },
  { href: '/alerts', label: 'Alerts', icon: Bell },
  { href: '/reports', label: 'Reports', icon: FileText },
];
const titles: Record<string, string> = {
  '/': 'Overview', '/datasets': 'Datasets', '/models': 'Models', '/monitoring': 'Monitoring',
  '/alerts': 'Alerts', '/reports': 'Reports', '/settings': 'Settings',
};
const fmt = (value: number | null | undefined, digits = 1) => value == null ? '—' : `${(value * 100).toFixed(digits)}%`;
const date = (value?: string | null) => value ? new Date(value).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' }) : '—';
const shortDate = (value?: string | null) => value ? new Date(value).toLocaleDateString(undefined, { dateStyle: 'medium' }) : '—';
const titleCase = (str?: string) => str ? str.replaceAll('_', ' ').replace(/\b\w/g, (c) => c.toUpperCase()) : '—';
const errText = (e: unknown) => e instanceof Error ? e.message : 'Something went wrong. Try again.';
function Button({ children, className = '', ...props }: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return <button className={`btn ${className}`} {...props}>{children}</button>;
}
function PageHead({ eyebrow, title, description, action }: { eyebrow: string; title: string; description?: string; action?: ReactNode }) {
  return <div className="page-head"><div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1>{description && <p className="subhead">{description}</p>}</div>{action}</div>;
}
function Panel({ title, subtitle, action, children, className = '' }: { title: string; subtitle?: string; action?: ReactNode; children: ReactNode; className?: string }) {
  return <section className={`panel ${className}`}><header className="panel-head"><div><div className="panel-title">{title}</div>{subtitle && <div className="panel-note">{subtitle}</div>}</div>{action}</header>{children}</section>;
}
function Empty({ icon: Icon = Database, title, copy, action }: { icon?: typeof Database; title: string; copy: string; action?: ReactNode }) {
  return <div className="empty"><div className="empty-icon"><Icon size={20}/></div><h3>{title}</h3><p>{copy}</p>{action && <div style={{ marginTop: 16 }}>{action}</div>}</div>;
}
function LoadingRows({ count = 3 }: { count?: number }) { return <div className="pad stack" aria-label="Loading">{Array.from({ length: count }, (_, i) => <div className="skeleton" key={i}/>)}</div>; }
function ErrorState({ error, retry }: { error: unknown; retry: () => void }) {
  return <div className="error-box" role="alert"><span>{errText(error)}</span><Button onClick={retry} data-testid="button-retry"><RefreshCw/>Retry</Button></div>;
}
function Pill({ children, tone = '' }: { children: ReactNode; tone?: string }) { return <span className={`pill ${tone}`}>{children}</span>; }
function AppShell({ children }: PageProps) {
  const [location] = useLocation();
  const [isDark, setIsDark] = useState(() => localStorage.getItem('confidence-collapse-theme') === 'dark');
  const health = useHealthCheck();
  useEffect(() => {
    document.documentElement.classList.toggle('dark', isDark);
    localStorage.setItem('confidence-collapse-theme', isDark ? 'dark' : 'light');
  }, [isDark]);
  const baseLocation = location.startsWith('/models/') ? '/models' : location;
  return <div className="shell">
    <aside className="sidebar">
      <Link href="/" className="brand" data-testid="link-brand"><span className="brand-mark"><Activity size={18}/></span><span>Confidence<span style={{ fontWeight: 400, opacity: .74 }}> Collapse</span></span></Link>
      <div className="nav-caption">Workspace</div><nav className="nav-items" aria-label="Primary navigation">
        {pages.map(({ href, label, icon: Icon }) => <Link key={href} href={href} className={`nav-link ${baseLocation === href ? 'active' : ''}`} data-testid={`link-nav-${label.toLowerCase()}`}><Icon/><span>{label}</span></Link>)}
      </nav>
      <div className="nav-caption" style={{ marginTop: 27 }}>Manage</div>
      <Link href="/settings" className={`nav-link ${baseLocation === '/settings' ? 'active' : ''}`} data-testid="link-nav-settings"><Settings2/><span>Settings</span></Link>
      <div className="side-bottom">Model quality, with evidence.<br/><span style={{ opacity: .65 }}>Monitoring workspace</span></div>
    </aside>
    <div className="main">
      <div className="topbar"><div className="crumb">{titles[baseLocation] || 'Model evaluation'}</div><div className="workspace"><span className="live-dot"/>{health.isLoading ? 'Connecting' : health.isError ? 'Service unavailable' : 'API connected'}<span style={{ margin: '0 4px', opacity: .4 }}>|</span><Button className="btn-quiet" title={isDark ? 'Switch to light mode' : 'Switch to dark mode'} aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'} onClick={() => setIsDark(value => !value)} data-testid="button-toggle-theme">{isDark ? <Sun/> : <Moon/>}</Button><CircleHelp size={14}/><span>Help</span></div></div>
      {children}
    </div>
  </div>;
}
function Dashboard() {
  const q = useGetDashboardSummary();
  const summary = q.data;
  return <div className="page">
    <PageHead eyebrow="Model operations / overview" title="Model quality, in view." description="A clear read on your datasets, evaluated models, and the evidence behind current performance." action={<Link className="btn btn-primary" href="/monitoring" data-testid="link-run-monitoring"><Activity size={15}/>Run monitoring<ArrowRight size={14}/></Link>}/>
    {q.isLoading && <div className="stats">{[1, 2, 3, 4].map(i => <div className="panel stat skeleton" key={i}/>)}</div>}
    {q.isError && <ErrorState error={q.error} retry={() => q.refetch()}/>}
    {summary && <>
      <div className="stats">
        <div className="panel stat" data-testid="metric-dataset-count"><div className="stat-label"><Database size={14}/>Datasets</div><div className="stat-value">{summary.datasetCount}</div><div className="stat-foot">Reference and incoming data</div></div>
        <div className="panel stat" data-testid="metric-model-count"><div className="stat-label"><Layers3 size={14}/>Evaluated models</div><div className="stat-value">{summary.modelCount}</div><div className="stat-foot">Trained against persisted data</div></div>
        <div className="panel stat" data-testid="metric-open-alerts"><div className="stat-label"><Bell size={14}/>Open alerts</div><div className="stat-value">{summary.openAlertCount}</div><div className="stat-foot">Evidence-backed signals</div></div>
        <div className="panel stat" data-testid="metric-latest-accuracy"><div className="stat-label"><BarChart3 size={14}/>Latest accuracy</div><div className="stat-value">{fmt(summary.latestAccuracy, 1)}</div><div className="stat-foot">Measured against labeled outcomes</div></div>
      </div>
      <div className="grid-2">
        <Panel title="Monitoring snapshot" subtitle="Most recent completed comparison">
          {summary.latestRunAt ? <div className="pad" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 18 }}>
            <div><div className="stat-label">Latest run</div><strong style={{ display: 'block', marginTop: 8, fontSize: 14 }}>{date(summary.latestRunAt)}</strong><div className="stat-foot" style={{ marginTop: 6 }}>Drift scores describe distribution change, not model accuracy.</div></div>
            <div style={{ textAlign: 'right' }}><div className="stat-label">Overall data drift</div><div className="stat-value" data-testid="metric-latest-drift">{fmt(summary.latestDriftScore, 1)}</div></div>
          </div> : <Empty icon={LineChart} title="No monitoring evidence yet" copy="Run a comparison between a model and an incoming dataset to establish your first drift result." action={<Link href="/monitoring" className="btn btn-primary" data-testid="link-first-monitoring">Set up monitoring</Link>}/>}
        </Panel>
        <Panel title="Recent activity" subtitle="Recorded platform events" action={<Link href="/datasets" className="btn btn-quiet" data-testid="link-all-datasets">Explore data<ChevronRight size={14}/></Link>}>
          {summary.activity.length ? <div>{summary.activity.map(item => <div key={item.id} data-testid={`activity-item-${item.id}`} style={{ display: 'grid', gridTemplateColumns: '30px 1fr auto', alignItems: 'center', gap: 10, padding: '13px 17px', borderTop: '1px solid hsl(var(--border))' }}>
            <div className="empty-icon" style={{ width: 29, height: 29, margin: 0, borderRadius: 8 }}>{item.kind === 'alert' ? <Bell size={14}/> : item.kind === 'monitoring' ? <Activity size={14}/> : item.kind === 'model' ? <Layers3 size={14}/> : <Database size={14}/>}</div>
            <div><div style={{ fontSize: 12, fontWeight: 600 }}>{item.title}</div><div className="panel-note">{item.detail}</div></div><span className="mono" style={{ color: 'hsl(var(--muted-foreground))', whiteSpace: 'nowrap' }}>{shortDate(item.createdAt)}</span>
          </div>)}</div> : <Empty icon={Activity} title="Activity will appear here" copy="Upload data or train a model to begin building an evidence trail."/>}
        </Panel>
      </div>
      <div className="notice" style={{ marginTop: 16 }}><strong>Keep the signals distinct.</strong> Predicted class is a label; confidence is a probability estimate. Data drift compares input distributions, prediction drift compares output distributions, and performance degradation is measured only when actual labels are available.</div>
    </>}
  </div>;
}
function Datasets() {
  const q = useListDatasets();
  const [purpose, setPurpose] = useState<'reference' | 'incoming'>('reference');
  const [name, setName] = useState('');
  const [targetColumn, setTargetColumn] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [selected, setSelected] = useState('');
  const [feedback, setFeedback] = useState('');
  const queryClient = useQueryClient();
  const upload = useUploadDataset({ mutation: { onSuccess: () => {
    queryClient.invalidateQueries({ queryKey: getListDatasetsQueryKey() });
    queryClient.invalidateQueries({ queryKey: getGetDashboardSummaryQueryKey() });
    setName(''); setTargetColumn(''); setFile(null); setFeedback('Dataset uploaded and validated.');
  } } });
  const remove = useDeleteDataset({ mutation: { onSuccess: () => {
    queryClient.invalidateQueries({ queryKey: getListDatasetsQueryKey() });
    queryClient.invalidateQueries({ queryKey: getGetDashboardSummaryQueryKey() });
    setSelected(''); setFeedback('Dataset removed.');
  } } });
  const detail = useGetDataset(selected, { query: { enabled: !!selected, queryKey: getGetDatasetQueryKey(selected) } });
  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!name.trim() || !file) return;
    const csvContent = await file.text();
    upload.mutate({ data: { name: name.trim(), purpose, csvContent, targetColumn: targetColumn.trim() || null } });
  };
  const dataset = (q.data || []).find(d => d.id === selected);
  return <div className="page">
    <PageHead eyebrow="Data / inventory" title="Datasets" description="Upload CSV data, inspect schema and missingness, and keep reference data separate from incoming batches."/>
    <div className="grid-2-equal" style={{ alignItems: 'start' }}>
      <Panel title="Add a dataset" subtitle="CSV parsing and validation run on upload">
        <form className="pad" onSubmit={submit}>
          <div className="field"><label htmlFor="dataset-name">Dataset name</label><input className="input" id="dataset-name" value={name} onChange={e => setName(e.target.value)} placeholder="Give this dataset a clear name" required data-testid="input-dataset-name"/></div>
          <div className="form-row">
            <div className="field"><label htmlFor="dataset-purpose">Purpose</label><select className="input" id="dataset-purpose" value={purpose} onChange={e => setPurpose(e.target.value as 'reference' | 'incoming')} data-testid="select-dataset-purpose"><option value="reference">Reference baseline</option><option value="incoming">Incoming batch</option></select></div>
          <div className="field"><label htmlFor="dataset-target">Target column <span style={{ color: 'hsl(var(--muted-foreground))', fontWeight: 400 }}>optional</span></label><input className="input" id="dataset-target" value={targetColumn} onChange={e => setTargetColumn(e.target.value)} placeholder="Target column name" data-testid="input-target-column"/></div>
          </div>
          <div className="field"><label htmlFor="dataset-file">CSV file</label><input className="input" id="dataset-file" type="file" accept=".csv,text/csv" onChange={e => setFile(e.target.files?.[0] || null)} required data-testid="input-dataset-file"/></div>
          {upload.isError && <div className="error-box" role="alert">{errText(upload.error)}</div>}
          {feedback && <div className="notice" role="status" data-testid="status-dataset-upload">{feedback}</div>}
          <Button type="submit" className="btn-primary" disabled={!name.trim() || !file || upload.isPending} data-testid="button-upload-dataset"><Upload/>{upload.isPending ? 'Validating dataset…' : 'Upload and validate'}</Button>
        </form>
      </Panel>
      <Panel title="Dataset inventory" subtitle={q.data ? `${q.data.length} persisted dataset${q.data.length === 1 ? '' : 's'}` : 'Persisted reference and incoming data'}>
        {q.isLoading ? <LoadingRows/> : q.isError ? <div className="pad"><ErrorState error={q.error} retry={() => q.refetch()}/></div> : !q.data?.length ? <Empty title="No datasets yet" copy="Upload a CSV to inspect columns, row counts, and data quality before training or monitoring."/> :
          <div className="table-wrap"><table><thead><tr><th>Dataset</th><th>Purpose</th><th>Rows</th><th>Columns</th><th></th></tr></thead><tbody>{q.data.map(d => <tr key={d.id} data-testid={`row-dataset-${d.id}`}><td><button className="btn btn-quiet" onClick={() => setSelected(d.id)} data-testid={`button-preview-dataset-${d.id}`} style={{ padding: 0, color: 'hsl(var(--foreground))' }}><span style={{ textAlign: 'left' }}><strong>{d.name}</strong><span className="panel-note" style={{ display: 'block' }}>{d.targetColumn ? `Target: ${d.targetColumn}` : 'No target selected'}</span></span></button></td><td><Pill tone={d.purpose === 'reference' ? 'blue' : ''}>{d.purpose}</Pill></td><td>{d.rowCount.toLocaleString()}</td><td>{d.columnCount}</td><td><Button className="btn-quiet btn-danger" onClick={() => { if (confirm(`Remove ${d.name}?`)) remove.mutate({ datasetId: d.id }); }} disabled={remove.isPending} aria-label={`Delete ${d.name}`} data-testid={`button-delete-dataset-${d.id}`}><Trash2 size={14}/></Button></td></tr>)}</tbody></table></div>}
      </Panel>
    </div>
    {selected && <div style={{ marginTop: 16 }}>
      <Panel title={dataset ? `Preview · ${dataset.name}` : 'Dataset preview'} subtitle={dataset ? `${dataset.rowCount.toLocaleString()} rows · ${dataset.columnCount} columns · ${dataset.missingCellCount.toLocaleString()} missing cells` : 'Schema validation and sample rows'}>
        <div style={{ display: 'flex', justifyContent: 'flex-end', padding: '8px 14px 0' }}><Button className="btn-quiet" onClick={() => setSelected('')} data-testid="button-close-preview"><X size={14}/>Close preview</Button></div>
        {detail.isLoading ? <LoadingRows/> : detail.isError ? <div className="pad"><ErrorState error={detail.error} retry={() => detail.refetch()}/></div> : detail.data && <div className="pad">
          <div className="grid-2-equal">
            <div><div className="panel-title" style={{ marginBottom: 9 }}>Column validation</div>{detail.data.columns.map(col => <div key={col.name} style={{ padding: '9px 0', borderTop: '1px solid hsl(var(--border))', display: 'grid', gridTemplateColumns: '1fr auto auto', gap: 12, fontSize: 11 }} data-testid={`column-summary-${col.name}`}><strong>{col.name}</strong><span>{col.type}</span><span style={{ color: col.missingValues ? '#a05d0b' : 'hsl(var(--muted-foreground))' }}>{col.missingValues} missing</span></div>)}</div>
            <div><div className="panel-title" style={{ marginBottom: 9 }}>Sample rows</div>{detail.data.preview.length ? <div className="table-wrap"><table><thead><tr>{detail.data.columns.map(c => <th key={c.name}>{c.name}</th>)}</tr></thead><tbody>{detail.data.preview.slice(0, 8).map((row, idx) => <tr key={idx} data-testid={`preview-row-${idx}`}>{detail.data.columns.map(c => <td key={c.name}>{row[c.name] == null ? <span style={{ color: 'hsl(var(--muted-foreground))' }}>null</span> : String(row[c.name])}</td>)}</tr>)}</tbody></table></div> : <div className="notice">The API returned no preview rows for this dataset.</div>}</div>
          </div>
        </div>}
      </Panel>
    </div>}
  </div>;
}
function Models() {
  const queryClient = useQueryClient();
  const models = useListModels();
  const datasets = useListDatasets();
  const train = useTrainModel({ mutation: { onSuccess: () => {
    queryClient.invalidateQueries({ queryKey: getListModelsQueryKey() });
    queryClient.invalidateQueries({ queryKey: getGetDashboardSummaryQueryKey() });
  } } });
  const referenceData = (datasets.data || []).filter(d => d.purpose === 'reference');
  const [datasetId, setDatasetId] = useState('');
  const chosen = referenceData.find(d => d.id === datasetId);
  const [target, setTarget] = useState('');
  const [algorithm, setAlgorithm] = useState<'decision_tree' | 'random_forest'>('random_forest');
  const [testSize, setTestSize] = useState('0.2');
  const [randomSeed, setRandomSeed] = useState('42');
  const [maxDepth, setMaxDepth] = useState('5');
  const [estimators, setEstimators] = useState('100');
  const comparisonData = useMemo(() => (models.data || []).map(model => ({
    name: model.name,
    accuracy: model.metrics.accuracy,
    precision: model.metrics.precision,
    recall: model.metrics.recall,
    f1: model.metrics.f1,
  })), [models.data]);
  useEffect(() => { if (chosen?.targetColumn) setTarget(chosen.targetColumn); }, [chosen?.id, chosen?.targetColumn]);
  const onTrain = (e: React.FormEvent) => { e.preventDefault(); if (!datasetId || !target) return; train.mutate({ data: { datasetId, targetColumn: target, algorithm, testSize: Number(testSize), randomSeed: Number(randomSeed), maxDepth: Number(maxDepth), estimators: Number(estimators) } }); };
  return <div className="page">
    <PageHead eyebrow="Models / training" title="Models" description="Train classifiers from reference datasets, then evaluate their predictions against held-out labels."/>
    <div className="grid-2-equal" style={{ alignItems: 'start' }}>
      <Panel title="Train a classifier" subtitle="Training creates persisted evaluation evidence">
        <form className="pad" onSubmit={onTrain}>
          <div className="field"><label htmlFor="train-dataset">Reference dataset</label><select id="train-dataset" className="input" value={datasetId} onChange={e => setDatasetId(e.target.value)} data-testid="select-train-dataset"><option value="">Select a reference dataset</option>{referenceData.map(d => <option key={d.id} value={d.id}>{d.name} · {d.rowCount} rows</option>)}</select></div>
          <div className="field"><label htmlFor="train-target">Target column</label><select id="train-target" className="input" value={target} onChange={e => setTarget(e.target.value)} disabled={!chosen} data-testid="select-train-target"><option value="">Select a target column</option>{chosen?.columns.map(c => <option key={c.name} value={c.name}>{c.name}</option>)}</select></div>
          <div className="form-row">
            <div className="field"><label htmlFor="train-algorithm">Algorithm</label><select id="train-algorithm" className="input" value={algorithm} onChange={e => setAlgorithm(e.target.value as 'decision_tree' | 'random_forest')} data-testid="select-train-algorithm"><option value="random_forest">Random forest</option><option value="decision_tree">Decision tree</option></select></div>
            <div className="field"><label htmlFor="train-test-size">Test split</label><input id="train-test-size" className="input" type="number" min=".1" max=".5" step=".05" value={testSize} onChange={e => setTestSize(e.target.value)} data-testid="input-test-size"/></div>
          </div>
          <div className="form-row">
            <div className="field"><label htmlFor="train-seed">Random seed</label><input id="train-seed" className="input" type="number" value={randomSeed} onChange={e => setRandomSeed(e.target.value)} data-testid="input-random-seed"/></div>
            {algorithm === 'decision_tree' ? <div className="field"><label htmlFor="train-depth">Maximum depth</label><input id="train-depth" className="input" type="number" min="1" max="12" value={maxDepth} onChange={e => setMaxDepth(e.target.value)} data-testid="input-max-depth"/></div> : <div className="field"><label htmlFor="train-estimators">Estimators</label><input id="train-estimators" className="input" type="number" min="1" max="100" value={estimators} onChange={e => setEstimators(e.target.value)} data-testid="input-estimators"/></div>}
          </div>
          {train.isError && <div className="error-box" role="alert">{errText(train.error)}</div>}
          {train.data && <div className="notice" role="status" data-testid="status-training">Training complete: {train.data.name}</div>}
          <Button type="submit" className="btn-primary" disabled={!datasetId || !target || train.isPending} data-testid="button-train-model"><Sparkles/>{train.isPending ? 'Training model…' : 'Train and evaluate'}</Button>
          {!referenceData.length && !datasets.isLoading && <div className="panel-note" style={{ marginTop: 12 }}>Upload a reference dataset first. Incoming datasets cannot be used for model training.</div>}
        </form>
      </Panel>
      <Panel title="Trained models" subtitle="Current models and measured evaluation metrics">
        {models.isLoading ? <LoadingRows/> : models.isError ? <div className="pad"><ErrorState error={models.error} retry={() => models.refetch()}/></div> : !models.data?.length ? <Empty icon={Layers3} title="No models trained" copy="Choose a labeled reference dataset to fit and evaluate your first classifier."/> :
          <div className="table-wrap"><table><thead><tr><th>Model</th><th>Algorithm</th><th>Accuracy</th><th>Rows</th></tr></thead><tbody>{models.data.map(model => <tr key={model.id} data-testid={`row-model-${model.id}`}><td><Link href={`/models/${model.id}`} className="btn btn-quiet" style={{ padding: 0, color: 'hsl(var(--foreground))' }} data-testid={`link-model-${model.id}`}><span style={{ textAlign: 'left' }}><strong>{model.name}</strong><span className="panel-note" style={{ display: 'block' }}>{model.datasetName} · {shortDate(model.trainedAt)}</span></span><ChevronRight size={14}/></Link></td><td>{titleCase(model.algorithm)}</td><td><strong>{fmt(model.metrics.accuracy)}</strong></td><td>{model.rowCount.toLocaleString()}</td></tr>)}</tbody></table></div>}
      </Panel>
    </div>
    {comparisonData.length > 0 && <div style={{ marginTop: 16 }}>
      <Panel title="Model performance comparison" subtitle="Metrics from persisted held-out evaluations">
        <div className="chart-wrap chart-tall" data-testid="chart-model-comparison">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={comparisonData} margin={{ top: 14, right: 18, left: 4, bottom: 40 }}>
              <CartesianGrid stroke="#e5eaf1" vertical={false}/>
              <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-14} textAnchor="end" interval={0} height={52}/>
              <YAxis domain={[0, 1]} tickFormatter={value => `${Math.round(value * 100)}%`} tick={{ fontSize: 10 }} width={38}/>
              <Tooltip formatter={value => fmt(Number(value))}/>
              <Legend/>
              <Bar dataKey="accuracy" name="Accuracy" fill="#2f68c8" radius={[3, 3, 0, 0]}/>
              <Bar dataKey="precision" name="Precision" fill="#8978dc" radius={[3, 3, 0, 0]}/>
              <Bar dataKey="recall" name="Recall" fill="#4f9d83" radius={[3, 3, 0, 0]}/>
              <Bar dataKey="f1" name="F1" fill="#d39b43" radius={[3, 3, 0, 0]}/>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Panel>
    </div>}
  </div>;
}
function ModelDetail() {
  const { modelId = '' } = useParams<{ modelId: string }>();
  const q = useGetModel(modelId, { query: { enabled: !!modelId, queryKey: getGetModelQueryKey(modelId) } });
  const pred = useListPredictions(modelId, { query: { enabled: !!modelId, queryKey: getListPredictionsQueryKey(modelId) } });
  const queryClient = useQueryClient();
  const model = q.data;
  const [rowsJson, setRowsJson] = useState('');
  const [rowsError, setRowsError] = useState('');
  const [selectedPredictionId, setSelectedPredictionId] = useState('');
  const [predictionClassFilter, setPredictionClassFilter] = useState('all');
  const [confidenceMinimum, setConfidenceMinimum] = useState('0');
  const [confidenceMaximum, setConfidenceMaximum] = useState('1');
  const [predictionOutcomeFilter, setPredictionOutcomeFilter] = useState('all');
  const predict = usePredictWithModel({ mutation: { onSuccess: () => {
    queryClient.invalidateQueries({ queryKey: getListPredictionsQueryKey(modelId) });
    queryClient.invalidateQueries({ queryKey: getGetModelReportQueryKey(modelId) });
  } } });
  const visiblePredictions = useMemo(() => (pred.data || []).filter(item => {
    const confidencePasses = item.confidence >= Number(confidenceMinimum || 0)
      && item.confidence <= Number(confidenceMaximum || 1);
    const classPasses = predictionClassFilter === 'all' || item.predictedLabel === predictionClassFilter;
    const outcomePasses = predictionOutcomeFilter === 'all'
      || (predictionOutcomeFilter === 'unlabeled' && item.actualLabel == null)
      || (predictionOutcomeFilter === 'correct' && item.actualLabel != null && item.actualLabel === item.predictedLabel)
      || (predictionOutcomeFilter === 'incorrect' && item.actualLabel != null && item.actualLabel !== item.predictedLabel);
    return confidencePasses && classPasses && outcomePasses;
  }), [pred.data, confidenceMinimum, confidenceMaximum, predictionClassFilter, predictionOutcomeFilter]);
  const selectedPrediction = visiblePredictions.find(item => item.id === selectedPredictionId) || visiblePredictions[0] || null;
  const predictionDistribution = (model?.classes || []).map(label => ({
    label,
    count: visiblePredictions.filter(item => item.predictedLabel === label).length,
  }));
  const probabilityData = selectedPrediction
    ? Object.entries(selectedPrediction.probabilities).map(([label, probability]) => ({
        label,
        probability,
        selected: label === selectedPrediction.predictedLabel,
      }))
    : [];
  const confusionMax = Math.max(1, ...(model?.metrics.confusionMatrix.flat() || []));
  const submitPredictions = (event: React.FormEvent) => {
    event.preventDefault();
    try {
      const rows = JSON.parse(rowsJson);
      if (!Array.isArray(rows) || rows.length === 0 || !rows.every(row => row && typeof row === 'object' && !Array.isArray(row))) {
        setRowsError('Provide a non-empty JSON array of row objects.');
        return;
      }
      setRowsError('');
      predict.mutate({ modelId, data: { rows } });
    } catch {
      setRowsError('Enter valid JSON: an array of objects using the model feature names.');
    }
  };
  return <div className="page">
    <PageHead eyebrow="Models / evaluation evidence" title={model?.name || 'Model evaluation'} description={model ? `${titleCase(model.algorithm)} · ${model.datasetName} · trained ${date(model.trainedAt)}` : 'Evaluation results, class-level outcomes, and prediction confidence.'} action={<Link href="/models" className="btn" data-testid="link-back-models">Back to models</Link>}/>
    {q.isLoading && <div className="stats">{[1,2,3,4].map(i => <div key={i} className="panel stat skeleton"/> )}</div>}
    {q.isError && <ErrorState error={q.error} retry={() => q.refetch()}/>}
    {model && <>
      <div className="stats">
        {[
          ['Accuracy', fmt(model.metrics.accuracy), 'Overall correct predictions'],
          ['Precision', fmt(model.metrics.precision), `${titleCase(model.metrics.averaging)} averaging`],
          ['Recall', fmt(model.metrics.recall), model.metrics.averaging === 'binary' ? `Positive class: ${model.classes[1]}` : 'Macro mean across classes'],
          ['F1 score', fmt(model.metrics.f1), 'Precision and recall harmonic mean'],
        ].map(([label, value, foot]) => <div className="panel stat" key={label} data-testid={`metric-model-${label.toLowerCase().replace(' ', '-')}`}><div className="stat-label">{label}</div><div className="stat-value">{value}</div><div className="stat-foot">{foot}</div></div>)}
      </div>
      <div className="grid-2-equal">
        <Panel title="Confusion matrix" subtitle="Held-out actual labels by predicted class">
          <div className="pad">
            {model.metrics.confusionMatrix.length ? <div className="table-wrap"><table className="matrix"><thead><tr><th>Actual ↓ / predicted →</th>{model.classes.map(c => <th key={c}>{c}</th>)}</tr></thead><tbody>{model.metrics.confusionMatrix.map((row, i) => <tr key={i}><th>{model.classes[i] ?? `Class ${i+1}`}</th>{row.map((count, j) => { const intensity = count / confusionMax; return <td key={j} style={{ backgroundColor: `rgba(47, 104, 200, ${0.08 + intensity * 0.42})`, color: intensity > 0.65 ? '#fff' : 'hsl(var(--foreground))' }} data-testid={`matrix-cell-${i}-${j}`}>{count.toLocaleString()}</td>; })}</tr>)}</tbody></table></div> : <Empty title="No confusion matrix" copy="The model evaluation returned no class-level matrix."/>}
            <div className="panel-note" style={{ marginTop: 13 }}>Rows show actual classes. Columns show predicted classes. Diagonal cells are correct classifications.</div>
          </div>
        </Panel>
        <Panel title="Feature importance" subtitle="Relative contribution returned by the fitted model">
          {model.featureImportances.length ? <div className="pad">{model.featureImportances.map(item => <div className="bar-row" key={item.feature} data-testid={`importance-${item.feature}`}><span>{item.feature}</span><div className="bar-track"><div className="bar-fill" style={{ width: `${Math.max(0, Math.min(100, item.importance * 100))}%` }}/></div><span className="mono">{fmt(item.importance, 1)}</span></div>)}</div> : <Empty title="Feature importance unavailable" copy="This model did not return feature importance values."/>}
        </Panel>
      </div>
      <div style={{ marginTop: 16 }} className="grid-2-equal">
        <Panel title="Per-class performance" subtitle="Metrics on the held-out evaluation split">
          {model.metrics.perClass.length ? <><div className="table-wrap"><table><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead><tbody>{model.metrics.perClass.map(c => <tr key={c.label} data-testid={`row-class-${c.label}`}><td><strong>{c.label}</strong></td><td>{fmt(c.precision)}</td><td>{fmt(c.recall)}</td><td>{fmt(c.f1)}</td><td>{c.support}</td></tr>)}</tbody></table></div><div className="panel-note" style={{ padding: '10px 15px' }}>{model.metrics.averaging === 'binary' ? `Binary precision, recall, and F1 use “${model.classes[1]}” as the positive class.` : 'Macro averaging gives each class equal weight, regardless of support.'}</div></> : <Empty title="No class metrics" copy="Class-level metrics were not returned for this evaluation."/>}
        </Panel>
        <Panel title="Prediction analysis" subtitle="Predicted label and confidence are different signals">
          <div className="notice" style={{ margin: 15 }}>Predicted class is the model's chosen label. Confidence is the probability assigned to that chosen label, not a measurement of correctness.</div>
          <form className="pad" onSubmit={submitPredictions} style={{ paddingTop: 0 }}>
            <div className="field"><label htmlFor="prediction-rows">Predict new rows</label><textarea id="prediction-rows" className="input" rows={4} value={rowsJson} onChange={e => setRowsJson(e.target.value)} placeholder="Paste a JSON array of row objects" aria-describedby="prediction-row-help" data-testid="input-prediction-rows" style={{ resize: 'vertical', fontFamily: 'var(--app-font-mono)', fontSize: 11 }}/><div id="prediction-row-help" className="panel-note">Submit up to 5,000 JSON row objects using this model's feature columns. Inputs are sent to the prediction API.</div></div>
            {rowsError && <div className="error-box" role="alert" data-testid="error-prediction-input">{rowsError}</div>}
            {predict.isError && <div className="error-box" role="alert">{errText(predict.error)}</div>}
            {predict.data && <div className="notice" role="status" data-testid="status-prediction-result">Returned {predict.data.predictions.length} predictions. Output class distribution is based on this submitted batch.</div>}
            <Button className="btn-primary" type="submit" disabled={!rowsJson.trim() || predict.isPending} data-testid="button-predict-rows"><Sparkles/>{predict.isPending ? 'Generating predictions…' : 'Generate predictions'}</Button>
          </form>
          {pred.isLoading ? <LoadingRows count={2}/> : pred.isError ? <div className="pad"><ErrorState error={pred.error} retry={() => pred.refetch()}/></div> : !pred.data?.length ? <Empty icon={LineChart} title="No predictions available" copy="No persisted evaluation or generated predictions were returned."/> : <>
            <div className="prediction-filters" aria-label="Prediction filters">
              <label className="field"><span>Predicted class</span><select className="input" value={predictionClassFilter} onChange={e => setPredictionClassFilter(e.target.value)} data-testid="select-prediction-class-filter"><option value="all">All classes</option>{model?.classes.map(label => <option key={label} value={label}>{label}</option>)}</select></label>
              <label className="field"><span>Minimum confidence</span><input className="input" type="number" min="0" max="1" step=".01" value={confidenceMinimum} onChange={e => setConfidenceMinimum(e.target.value)} data-testid="input-confidence-filter"/></label>
              <label className="field"><span>Maximum confidence</span><input className="input" type="number" min="0" max="1" step=".01" value={confidenceMaximum} onChange={e => setConfidenceMaximum(e.target.value)} data-testid="input-confidence-maximum-filter"/></label>
              <label className="field"><span>Outcome</span><select className="input" value={predictionOutcomeFilter} onChange={e => setPredictionOutcomeFilter(e.target.value)} data-testid="select-prediction-outcome-filter"><option value="all">All outcomes</option><option value="correct">Correct</option><option value="incorrect">Incorrect</option><option value="unlabeled">Not labeled</option></select></label>
            </div>
            {visiblePredictions.length === 0 ? <Empty icon={SlidersHorizontal} title="No predictions match these filters" copy="Adjust the class, confidence, or outcome filters to view recorded results."/> : <>
              <div className="chart-grid">
                <section className="chart-card">
                  <div className="chart-heading"><div className="panel-title">Predicted class counts</div><div className="panel-note">{visiblePredictions.length.toLocaleString()} matching records</div></div>
                  <div className="chart-wrap" data-testid="chart-prediction-distribution">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={predictionDistribution} margin={{ top: 8, right: 12, left: -15, bottom: 8 }}>
                        <CartesianGrid stroke="#e5eaf1" vertical={false}/>
                        <XAxis dataKey="label" tick={{ fontSize: 10 }}/>
                        <YAxis allowDecimals={false} tick={{ fontSize: 10 }}/>
                        <Tooltip/>
                        <Bar dataKey="count" name="Predictions" fill="#2f68c8" radius={[4, 4, 0, 0]}/>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </section>
                <section className="chart-card">
                  <div className="chart-heading">
                    <div className="panel-title">Class probabilities</div>
                    <div className="panel-note">{selectedPrediction ? `Row ${selectedPrediction.rowIndex} · predicted ${selectedPrediction.predictedLabel} · confidence ${fmt(selectedPrediction.confidence)}` : 'Select a prediction below'}</div>
                  </div>
                  {probabilityData.length ? <div className="chart-wrap" data-testid={`chart-confidence-${selectedPrediction?.id}`}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={probabilityData} margin={{ top: 8, right: 12, left: -15, bottom: 8 }}>
                        <CartesianGrid stroke="#e5eaf1" vertical={false}/>
                        <XAxis dataKey="label" tick={{ fontSize: 10 }}/>
                        <YAxis domain={[0, 1]} tickFormatter={value => `${Math.round(value * 100)}%`} tick={{ fontSize: 10 }}/>
                        <Tooltip formatter={value => fmt(Number(value))}/>
                        <Bar dataKey="probability" name="Class probability" radius={[4, 4, 0, 0]}>
                          {probabilityData.map(item => <Cell key={item.label} fill={item.selected ? '#2f68c8' : '#9a8de4'}/>)}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div> : <Empty title="No class probabilities returned" copy="The backend did not provide probabilities for this prediction."/ >}
                </section>
              </div>
              <div className="table-wrap"><table><thead><tr><th>Row</th><th>Predicted class</th><th>Actual class</th><th>Outcome</th><th>Confidence</th></tr></thead><tbody>{visiblePredictions.slice(0, 30).map(p => <tr key={p.id} tabIndex={0} aria-selected={selectedPrediction?.id === p.id} onClick={() => setSelectedPredictionId(p.id)} onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setSelectedPredictionId(p.id); } }} data-testid={`row-prediction-${p.id}`} className={selectedPrediction?.id === p.id ? 'row-selected' : ''}><td>{p.rowIndex}</td><td><strong>{p.predictedLabel}</strong></td><td>{p.actualLabel ?? 'Not labeled'}</td><td>{p.actualLabel == null ? 'Not measured' : p.actualLabel === p.predictedLabel ? 'Correct' : 'Incorrect'}</td><td>{fmt(p.confidence)}</td></tr>)}</tbody></table></div>
            </>}
          </>}
        </Panel>
      </div>
    </>}
  </div>;
}
function Monitoring() {
  const queryClient = useQueryClient();
  const models = useListModels();
  const datasets = useListDatasets();
  const runs = useListMonitoringRuns();
  const [modelId, setModelId] = useState('');
  const [datasetId, setDatasetId] = useState('');
  const incoming = (datasets.data || []).filter(d => d.purpose === 'incoming');
  const run = useRunMonitoring({ mutation: { onSuccess: result => {
    queryClient.invalidateQueries({ queryKey: getListMonitoringRunsQueryKey() });
    queryClient.invalidateQueries({ queryKey: getListAlertsQueryKey() });
    queryClient.invalidateQueries({ queryKey: getGetDashboardSummaryQueryKey() });
    queryClient.invalidateQueries({ queryKey: getGetModelReportQueryKey(result.modelId) });
  } } });
  const submit = (e: React.FormEvent) => { e.preventDefault(); if (modelId && datasetId) run.mutate({ data: { modelId, datasetId } }); };
  const performanceHistory = (runs.data || []).flatMap(item => item.performance ? [{
    date: shortDate(item.createdAt),
    dataset: item.datasetName,
    accuracy: item.performance.accuracy,
  }] : []).reverse();
  return <div className="page">
    <PageHead eyebrow="Monitoring / comparisons" title="Monitor incoming data" description="Compare an incoming dataset to the model's reference baseline. Distribution shift is not the same as measured performance loss."/>
    <Panel title="Run a comparison" subtitle="Select the model and incoming dataset to evaluate">
      <form className="pad" onSubmit={submit}>
        <div className="form-row">
          <div className="field"><label htmlFor="monitor-model">Model</label><select className="input" id="monitor-model" value={modelId} onChange={e => setModelId(e.target.value)} data-testid="select-monitor-model"><option value="">Select model</option>{(models.data || []).map(m => <option key={m.id} value={m.id}>{m.name} · {m.datasetName}</option>)}</select></div>
          <div className="field"><label htmlFor="monitor-dataset">Incoming dataset</label><select className="input" id="monitor-dataset" value={datasetId} onChange={e => setDatasetId(e.target.value)} data-testid="select-monitor-dataset"><option value="">Select incoming data</option>{incoming.map(d => <option key={d.id} value={d.id}>{d.name} · {d.rowCount} rows</option>)}</select></div>
        </div>
        {run.isError && <div className="error-box" role="alert">{errText(run.error)}</div>}
        {run.data && <div className="notice" role="status" data-testid={`status-monitoring-run-${run.data.id}`}>Comparison completed. Overall data drift {fmt(run.data.overallDriftScore)}; prediction drift {fmt(run.data.predictionDriftScore)}; performance {run.data.performanceStatus === 'not_measured' ? 'not measured' : titleCase(run.data.performanceStatus)}.</div>}
        <Button className="btn-primary" type="submit" disabled={!modelId || !datasetId || run.isPending} data-testid="button-run-monitoring"><Activity/>{run.isPending ? 'Comparing distributions…' : 'Run comparison'}</Button>
        {!incoming.length && !datasets.isLoading && <div className="panel-note" style={{ marginTop: 10 }}>Upload a dataset with purpose “Incoming batch” to enable comparisons.</div>}
      </form>
    </Panel>
    {run.data && <div style={{ marginTop: 16 }}><RunEvidence run={run.data}/></div>}
    <div style={{ marginTop: 16 }}><Panel title="Monitoring history" subtitle={runs.data ? `${runs.data.length} persisted run${runs.data.length === 1 ? '' : 's'}` : 'Completed incoming-data comparisons'}>
      {runs.isLoading ? <LoadingRows/> : runs.isError ? <div className="pad"><ErrorState error={runs.error} retry={() => runs.refetch()}/></div> : !runs.data?.length ? <Empty icon={Activity} title="No monitoring runs" copy="Completed comparisons will be listed here with their data drift, prediction drift, and performance evidence."/> :
        <div className="table-wrap"><table><thead><tr><th>Run</th><th>Model</th><th>Incoming data</th><th>Rows</th><th>Data drift</th><th>Prediction drift</th><th>Performance</th></tr></thead><tbody>{runs.data.map(r => <tr key={r.id} data-testid={`row-monitoring-run-${r.id}`}><td>{date(r.createdAt)}</td><td>{r.modelName}</td><td>{r.datasetName}</td><td>{r.rowCount.toLocaleString()}</td><td><strong>{fmt(r.overallDriftScore)}</strong></td><td>{fmt(r.predictionDriftScore)}</td><td><Pill tone={r.performanceStatus === 'degraded' ? 'bad' : r.performanceStatus === 'stable' ? 'good' : ''}>{titleCase(r.performanceStatus)}</Pill></td></tr>)}</tbody></table></div>}
    </Panel></div>
    {performanceHistory.length > 0 && <div style={{ marginTop: 16 }}>
      <Panel title="Measured performance history" subtitle="Only monitoring runs with actual ground-truth labels are included">
        <div className="chart-wrap chart-tall" data-testid="chart-performance-history">
          <ResponsiveContainer width="100%" height="100%">
            <RechartsLineChart data={performanceHistory} margin={{ top: 12, right: 20, left: 4, bottom: 8 }}>
              <CartesianGrid stroke="#e5eaf1" vertical={false}/>
              <XAxis dataKey="date" tick={{ fontSize: 10 }}/>
              <YAxis domain={[0, 1]} tickFormatter={value => `${Math.round(value * 100)}%`} tick={{ fontSize: 10 }} width={38}/>
              <Tooltip formatter={value => fmt(Number(value))} labelFormatter={(label, payload) => payload?.[0]?.payload?.dataset ? `${label} · ${payload[0].payload.dataset}` : label}/>
              <Line type="monotone" dataKey="accuracy" name="Accuracy" stroke="#2f68c8" strokeWidth={2} dot={{ r: 3 }} activeDot={{ r: 5 }}/>
            </RechartsLineChart>
          </ResponsiveContainer>
        </div>
      </Panel>
    </div>}
  </div>;
}
function RunEvidence({ run }: { run: import('@workspace/api-client-react').MonitoringRun }) {
  const [selectedFeature, setSelectedFeature] = useState(run.featureDrift[0]?.feature || '');
  useEffect(() => { setSelectedFeature(run.featureDrift[0]?.feature || ''); }, [run.id]);
  const feature = run.featureDrift.find(item => item.feature === selectedFeature);
  const incomingBins = new Map((feature?.currentDistribution || []).map(bin => [bin.label, bin.count]));
  const featureDistribution = (feature?.referenceDistribution || []).map(bin => ({
    label: bin.label,
    reference: bin.count,
    incoming: incomingBins.get(bin.label) || 0,
  }));
  const driftScores = run.featureDrift
    .filter(item => item.method !== 'unavailable')
    .map(item => ({ feature: item.feature, score: item.score }));
  const predictionLabels = Array.from(new Set([
    ...Object.keys(run.referencePredictionDistribution),
    ...Object.keys(run.predictionDistribution),
  ]));
  const predictionBars = predictionLabels.map(label => ({
    label,
    reference: run.referencePredictionDistribution[label] || 0,
    incoming: run.predictionDistribution[label] || 0,
  }));
  return <Panel title={`Evidence · ${run.datasetName}`} subtitle={`${run.modelName} · ${date(run.createdAt)} · ${run.rowCount.toLocaleString()} rows`}>
    <div className="stats" style={{ padding: '0 16px', margin: '15px 0' }}>
      <div className="panel stat"><div className="stat-label">Data drift</div><div className="stat-value">{fmt(run.overallDriftScore)}</div><div className="stat-foot">Input feature distributions changed</div></div>
      <div className="panel stat"><div className="stat-label">Prediction drift</div><div className="stat-value">{fmt(run.predictionDriftScore)}</div><div className="stat-foot">Predicted-label distribution changed</div></div>
      <div className="panel stat"><div className="stat-label">Performance</div><div className="stat-value" style={{ fontSize: 20 }}>{titleCase(run.performanceStatus)}</div><div className="stat-foot">{run.performance ? `Accuracy ${fmt(run.performance.accuracy)}` : 'Actual labels unavailable; not measured'}</div></div>
      <div className="panel stat"><div className="stat-label">Generated alerts</div><div className="stat-value">{run.alertIds.length}</div><div className="stat-foot">Linked to recorded evidence</div></div>
    </div>
    <div className="panel-note" style={{ padding: '0 20px 16px' }}>Feature scores below describe input data drift; they do not represent prediction confidence or model correctness.</div>
    {run.featureDrift.length ? <div className="table-wrap"><table><thead><tr><th>Feature</th><th>Type</th><th>Method</th><th>Score</th><th>p-value</th><th>Status</th></tr></thead><tbody>{run.featureDrift.map(f => <tr key={f.feature} data-testid={`row-feature-drift-${f.feature}`}><td><strong>{f.feature}</strong></td><td>{f.type}</td><td>{titleCase(f.method)}</td><td>{fmt(f.score)}</td><td>{f.pValue == null ? '—' : f.pValue.toPrecision(3)}</td><td><Pill tone={f.status === 'high' ? 'bad' : f.status === 'warning' ? 'warn' : f.status === 'low' ? 'good' : ''}>{f.status}</Pill></td></tr>)}</tbody></table></div> : <Empty title="No feature-level drift results" copy="This run did not return feature drift details."/>}
    <div className="panel-note" style={{ padding: '10px 16px' }}>Numeric drift scores combine the KS statistic with absolute missing-rate shift; KS p-values are asymptotic approximations for the distribution-shape test. Categorical scores include missing values, and chi-square results are withheld when sample-size or expected-count assumptions are not met.</div>
    <div className="chart-grid">
      <section className="chart-card">
        <div className="chart-heading"><div className="panel-title">Feature distributions</div><div className="panel-note">Counts from the stored reference and incoming datasets</div></div>
        {run.featureDrift.length > 0 && <label className="field chart-select"><span>Feature</span><select className="input" value={selectedFeature} onChange={e => setSelectedFeature(e.target.value)} data-testid={`select-drift-feature-${run.id}`}>{run.featureDrift.map(item => <option key={item.feature} value={item.feature}>{item.feature}</option>)}</select></label>}
        {featureDistribution.length ? <div className="chart-wrap" data-testid={`chart-feature-distribution-${run.id}`}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={featureDistribution} margin={{ top: 8, right: 12, left: -15, bottom: 8 }}>
              <CartesianGrid stroke="#e5eaf1" vertical={false}/>
              <XAxis dataKey="label" tick={{ fontSize: 9 }}/>
              <YAxis allowDecimals={false} tick={{ fontSize: 10 }}/>
              <Tooltip/>
              <Legend/>
              <Bar dataKey="reference" name="Reference" fill="#9a8de4" radius={[3, 3, 0, 0]}/>
              <Bar dataKey="incoming" name="Incoming" fill="#2f68c8" radius={[3, 3, 0, 0]}/>
            </BarChart>
          </ResponsiveContainer>
        </div> : <Empty title="No comparable distribution" copy="This feature has too few comparable values to chart."/>}
      </section>
      <section className="chart-card">
        <div className="chart-heading"><div className="panel-title">Feature drift scores</div><div className="panel-note">Only features with a suitable statistical test are plotted</div></div>
        {driftScores.length ? <div className="chart-wrap" data-testid={`chart-feature-drift-${run.id}`}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={driftScores} margin={{ top: 8, right: 12, left: -15, bottom: 8 }}>
              <CartesianGrid stroke="#e5eaf1" vertical={false}/>
              <XAxis dataKey="feature" tick={{ fontSize: 9 }}/>
              <YAxis domain={[0, 1]} tick={{ fontSize: 10 }}/>
              <Tooltip/>
              <Bar dataKey="score" name="Drift score" fill="#8978dc" radius={[4, 4, 0, 0]}/>
            </BarChart>
          </ResponsiveContainer>
        </div> : <Empty title="No feature scores to chart" copy="Sample sizes or test assumptions were insufficient for statistical interpretation."/>}
      </section>
    </div>
    {predictionLabels.length > 0 && <div style={{ borderTop: '1px solid hsl(var(--border))', marginTop: 4 }}>
      <div className="panel-head"><div><div className="panel-title">Predicted class distribution</div><div className="panel-note">Observed prediction counts compared with the reference distribution</div></div></div>
      <div className="chart-wrap chart-tall" data-testid={`chart-monitoring-predictions-${run.id}`}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={predictionBars} margin={{ top: 12, right: 18, left: 4, bottom: 8 }}>
            <CartesianGrid stroke="#e5eaf1" vertical={false}/>
            <XAxis dataKey="label" tick={{ fontSize: 10 }}/>
            <YAxis allowDecimals={false} tick={{ fontSize: 10 }}/>
            <Tooltip/>
            <Legend/>
            <Bar dataKey="reference" name="Reference predictions" fill="#9a8de4" radius={[3, 3, 0, 0]}/>
            <Bar dataKey="incoming" name="Incoming predictions" fill="#2f68c8" radius={[3, 3, 0, 0]}/>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="table-wrap"><table><thead><tr><th>Predicted class</th><th>Reference predictions</th><th>Incoming predictions</th></tr></thead><tbody>{predictionLabels.map(label => <tr key={label} data-testid={`row-prediction-distribution-${label}`}><td><strong>{label}</strong></td><td>{run.referencePredictionDistribution[label] ?? 0}</td><td>{run.predictionDistribution[label] ?? 0}</td></tr>)}</tbody></table></div>
    </div>}
  </Panel>;
}
function Alerts() {
  const q = useListAlerts();
  const queryClient = useQueryClient();
  const mutation = useUpdateAlert({ mutation: { onSuccess: alert => {
    queryClient.invalidateQueries({ queryKey: getListAlertsQueryKey() });
    queryClient.invalidateQueries({ queryKey: getGetDashboardSummaryQueryKey() });
    queryClient.invalidateQueries({ queryKey: getGetModelReportQueryKey(alert.modelId) });
  } } });
  const [filter, setFilter] = useState('all');
  const alerts = useMemo(() => (q.data || []).filter(a => filter === 'all' || a.status === filter), [q.data, filter]);
  const alertFrequency = useMemo(() => {
    const days = new Map<string, { day: string; low: number; medium: number; high: number }>();
    (q.data || []).forEach(alert => {
      const day = new Date(alert.createdAt).toISOString().slice(0, 10);
      const entry = days.get(day) || { day, low: 0, medium: 0, high: 0 };
      entry[alert.severity] += 1;
      days.set(day, entry);
    });
    return [...days.values()].sort((a, b) => a.day.localeCompare(b.day)).slice(-14);
  }, [q.data]);
  return <div className="page">
    <PageHead eyebrow="Monitoring / alert evidence" title="Alerts" description="Each alert includes the change detected and a recommended action. Resolve when reviewed; reopen if it needs attention again."/>
    {alertFrequency.length > 0 && <div style={{ marginBottom: 16 }}>
      <Panel title="Alert frequency" subtitle="Recorded alerts by creation date and severity">
        <div className="chart-wrap chart-tall" data-testid="chart-alert-frequency">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={alertFrequency} margin={{ top: 12, right: 18, left: 4, bottom: 8 }}>
              <CartesianGrid stroke="#e5eaf1" vertical={false}/>
              <XAxis dataKey="day" tickFormatter={value => shortDate(value)} tick={{ fontSize: 10 }}/>
              <YAxis allowDecimals={false} tick={{ fontSize: 10 }}/>
              <Tooltip labelFormatter={value => shortDate(String(value))}/>
              <Legend/>
              <Bar dataKey="low" name="Low" stackId="severity" fill="#7cae98"/>
              <Bar dataKey="medium" name="Medium" stackId="severity" fill="#d39b43"/>
              <Bar dataKey="high" name="High" stackId="severity" fill="#c9675e" radius={[3, 3, 0, 0]}/>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Panel>
    </div>}
    <Panel title="Alert register" subtitle={q.data ? `${q.data.filter(a => a.status === 'open').length} open · ${q.data.filter(a => a.status === 'resolved').length} resolved` : 'Evidence-backed monitoring signals'} action={<label style={{ display: 'flex', alignItems: 'center', gap: 7, fontSize: 11 }}><SlidersHorizontal size={14}/><select className="input" style={{ width: 118, padding: '7px 8px' }} value={filter} onChange={e => setFilter(e.target.value)} aria-label="Filter alerts" data-testid="select-alert-filter"><option value="all">All alerts</option><option value="open">Open</option><option value="resolved">Resolved</option></select></label>}>
      {q.isLoading ? <LoadingRows/> : q.isError ? <div className="pad"><ErrorState error={q.error} retry={() => q.refetch()}/></div> : !alerts.length ? <Empty icon={ShieldCheck} title={filter === 'all' ? 'No alerts recorded' : `No ${filter} alerts`} copy="Alerts are created when a monitoring comparison finds evidence beyond your configured thresholds."/> :
        <div>{alerts.map(alert => <article key={alert.id} data-testid={`alert-card-${alert.id}`} style={{ padding: '18px 20px', borderTop: '1px solid hsl(var(--border))' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 15 }}>
            <div style={{ flex: 1 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}><Pill tone={alert.severity === 'high' ? 'bad' : alert.severity === 'medium' ? 'warn' : 'blue'}>{alert.severity} severity</Pill><Pill tone={alert.status === 'open' ? 'warn' : 'good'}>{alert.status}</Pill><span className="panel-note">{date(alert.createdAt)}</span></div>
              <h3 style={{ fontSize: 14, margin: '11px 0 6px' }}>{alert.title}</h3><p style={{ fontSize: 12, lineHeight: 1.6, margin: '0 0 10px', color: 'hsl(var(--muted-foreground))' }}>{alert.explanation}</p>
              <div className="notice"><strong>Recommended action</strong><br/>{alert.recommendedAction}</div>
              <div style={{ marginTop: 10 }}><Link href="/monitoring" className="btn btn-quiet" style={{ padding: '4px 0' }} data-testid={`link-alert-run-${alert.id}`}>Monitoring run <span className="mono">{alert.monitoringRunId}</span><ArrowRight size={13}/></Link></div>
            </div>
            <Button className={alert.status === 'open' ? 'btn-primary' : ''} onClick={() => mutation.mutate({ alertId: alert.id, data: { status: alert.status === 'open' ? 'resolved' : 'open' } })} disabled={mutation.isPending} data-testid={`button-${alert.status === 'open' ? 'resolve' : 'reopen'}-alert-${alert.id}`}>{alert.status === 'open' ? <Check/> : <RefreshCw/>}{alert.status === 'open' ? 'Resolve' : 'Reopen'}</Button>
          </div>
        </article>)}
        {mutation.isError && <div className="pad"><div className="error-box" role="alert">{errText(mutation.error)}</div></div>}
        </div>}
    </Panel>
  </div>;
}
function Reports() {
  const models = useListModels();
  const [modelId, setModelId] = useState('');
  const report = useGetModelReport(modelId, { query: { enabled: !!modelId, queryKey: getGetModelReportQueryKey(modelId) } });
  const model = models.data?.find(m => m.id === modelId);
  const download = (format: 'json' | 'csv') => {
    if (!report.data) return;
    let content = '';
    let type = '';
    let ext = format;
    if (format === 'json') { content = JSON.stringify(report.data, null, 2); type = 'application/json'; }
    else {
      const rows = [['section', 'id', 'model', 'recorded_at', 'detail']];
      const r = report.data;
      rows.push(['model', r.model.id, r.model.name, r.model.trainedAt, `accuracy=${r.model.metrics.accuracy}; precision=${r.model.metrics.precision}; recall=${r.model.metrics.recall}; f1=${r.model.metrics.f1}`]);
      r.predictions.forEach(p => rows.push(['prediction', p.id, p.predictedLabel, p.createdAt, `actual=${p.actualLabel ?? ''}; confidence=${p.confidence}; source=${p.source}`]));
      r.monitoringRuns.forEach(run => rows.push(['monitoring', run.id, run.modelName, run.createdAt, `data_drift=${run.overallDriftScore}; prediction_drift=${run.predictionDriftScore}; performance=${run.performanceStatus}`]));
      r.alerts.forEach(a => rows.push(['alert', a.id, a.title, a.createdAt, `severity=${a.severity}; status=${a.status}; explanation=${a.explanation}`]));
      content = rows.map(row => row.map(cell => `"${String(cell).replaceAll('"', '""')}"`).join(',')).join('\r\n');
      type = 'text/csv'; ext = 'csv';
    }
    const url = URL.createObjectURL(new Blob([content], { type }));
    const a = document.createElement('a'); a.href = url; a.download = `${(report.data.model.name || 'model-report').replace(/[^a-z0-9-_]+/gi, '-')}.${ext}`; a.click(); URL.revokeObjectURL(url);
  };
  return <div className="page">
    <PageHead eyebrow="Evidence / export" title="Reports" description="Select a trained model to assemble its real evaluation, predictions, monitoring history, and linked alerts into an exportable report."/>
    <Panel title="Build a model report" subtitle="Report data is fetched from the API; no synthetic values are added">
      <div className="pad">
        <div className="field" style={{ maxWidth: 520 }}><label htmlFor="report-model">Model</label><select className="input" id="report-model" value={modelId} onChange={e => setModelId(e.target.value)} data-testid="select-report-model"><option value="">Select model</option>{(models.data || []).map(m => <option value={m.id} key={m.id}>{m.name} · {m.datasetName}</option>)}</select></div>
        {models.isError && <ErrorState error={models.error} retry={() => models.refetch()}/>}
        {models.isLoading && <div className="skeleton"/>}
        {!models.isLoading && models.data?.length === 0 && <Empty icon={FileText} title="No model reports available" copy="Train a model before generating its report."/>}
        {modelId && report.isLoading && <LoadingRows count={2}/>}
        {modelId && report.isError && <ErrorState error={report.error} retry={() => report.refetch()}/>}
        {report.data && <div style={{ marginTop: 17 }}>
          <div className="grid-2-equal">
            <div className="notice"><strong>{report.data.model.name}</strong><br/>Trained {date(report.data.model.trainedAt)} on {report.data.model.datasetName}. Evaluation accuracy {fmt(report.data.model.metrics.accuracy)}.</div>
            <div className="notice"><strong>Included evidence</strong><br/>{report.data.predictions.length} predictions · {report.data.monitoringRuns.length} monitoring runs · {report.data.alerts.length} alerts</div>
          </div>
          <div style={{ display: 'flex', gap: 9, marginTop: 15, flexWrap: 'wrap' }}>
            <Button className="btn-primary" onClick={() => download('json')} data-testid="button-export-json"><ArrowDownToLine/>Export complete report JSON</Button>
            <Button onClick={() => download('csv')} data-testid="button-export-csv"><ArrowDownToLine/>Export summary CSV</Button>
          </div>
          <div className="panel-note" style={{ marginTop: 12 }}>CSV contains the selected model metrics plus its returned prediction, monitoring, and alert records. JSON preserves the complete API report payload.</div>
        </div>}
      </div>
    </Panel>
  </div>;
}
function Settings() {
  const q = useGetSettings();
  const queryClient = useQueryClient();
  const update = useUpdateSettings({ mutation: { onSuccess: settings => {
    queryClient.setQueryData(getGetSettingsQueryKey(), settings);
    queryClient.invalidateQueries({ queryKey: getGetDashboardSummaryQueryKey() });
  } } });
  const [values, setValues] = useState({ driftThreshold: '', predictionDriftThreshold: '', performanceDropThreshold: '' });
  useEffect(() => { if (q.data) setValues({ driftThreshold: String(q.data.driftThreshold), predictionDriftThreshold: String(q.data.predictionDriftThreshold), performanceDropThreshold: String(q.data.performanceDropThreshold) }); }, [q.data]);
  const submit = (e: React.FormEvent) => { e.preventDefault(); update.mutate({ data: { driftThreshold: Number(values.driftThreshold), predictionDriftThreshold: Number(values.predictionDriftThreshold), performanceDropThreshold: Number(values.performanceDropThreshold) } }); };
  const fields = [
    { key: 'driftThreshold' as const, label: 'Data drift threshold', help: 'Maximum acceptable input feature distribution shift before a drift alert is considered.' },
    { key: 'predictionDriftThreshold' as const, label: 'Prediction drift threshold', help: 'Maximum acceptable change in the distribution of predicted classes.' },
    { key: 'performanceDropThreshold' as const, label: 'Performance drop threshold', help: 'Maximum measured performance decrease versus the reference. Requires actual labels.' },
  ];
  return <div className="page">
    <PageHead eyebrow="Workspace / configuration" title="Monitoring thresholds" description="Set the boundaries used to interpret future comparisons. Thresholds are proportions from 0 to 1."/>
    <div className="grid-2-equal" style={{ alignItems: 'start' }}>
      <Panel title="Alert thresholds" subtitle="Saved values are applied to monitoring comparisons">
        {q.isLoading ? <LoadingRows count={3}/> : q.isError ? <div className="pad"><ErrorState error={q.error} retry={() => q.refetch()}/></div> : <form className="pad" onSubmit={submit}>
          {fields.map(f => <div className="field" key={f.key}><label htmlFor={f.key}>{f.label}</label><div style={{ display: 'flex', alignItems: 'center', gap: 10 }}><input className="input" type="number" id={f.key} min="0" max="1" step=".01" required value={values[f.key]} onChange={e => setValues(v => ({ ...v, [f.key]: e.target.value }))} data-testid={`input-${f.key}`}/><span className="mono" style={{ color: 'hsl(var(--muted-foreground))' }}>{fmt(Number(values[f.key]))}</span></div><div className="panel-note">{f.help}</div></div>)}
          {update.isError && <div className="error-box" role="alert">{errText(update.error)}</div>}
          {update.isSuccess && <div className="notice" role="status" data-testid="status-settings-saved">Thresholds saved.</div>}
          <Button className="btn-primary" type="submit" disabled={update.isPending} data-testid="button-save-settings"><Check/>{update.isPending ? 'Saving thresholds…' : 'Save thresholds'}</Button>
        </form>}
      </Panel>
      <Panel title="How to read the signals" subtitle="Different metrics answer different questions">
        <div className="pad stack">
          <div className="notice"><strong>Data drift</strong><br/>Did feature distributions change between reference and incoming rows?</div>
          <div className="notice"><strong>Prediction drift</strong><br/>Did the model produce a different mix of predicted labels?</div>
          <div className="notice"><strong>Performance degradation</strong><br/>Did correctness metrics decrease when actual labels are available?</div>
          <div className="notice"><strong>Prediction confidence</strong><br/>How much probability did the model assign to its selected class? This is not an accuracy score.</div>
        </div>
      </Panel>
    </div>
  </div>;
}
function Router() {
  return <AppShell><RoutedErrorBoundary><Switch>
    <Route path="/" component={Dashboard}/>
    <Route path="/datasets" component={Datasets}/>
    <Route path="/models" component={Models}/>
    <Route path="/models/:modelId" component={ModelDetail}/>
    <Route path="/monitoring" component={Monitoring}/>
    <Route path="/alerts" component={Alerts}/>
    <Route path="/reports" component={Reports}/>
    <Route path="/settings" component={Settings}/>
    <Route component={NotFound}/>
  </Switch></RoutedErrorBoundary></AppShell>;
}
function RoutedErrorBoundary({ children }: PageProps) {
  const [location] = useLocation();
  return <ErrorBoundary resetKey={location}>{children}</ErrorBoundary>;
}
function App() {
  return <QueryClientProvider client={queryClient}><TooltipProvider><WouterRouter base={import.meta.env.BASE_URL.replace(/\/$/, '')}><Router/></WouterRouter><Toaster/></TooltipProvider></QueryClientProvider>;
}
export default App;
