import { useEffect, useState, useCallback } from 'react';
import {
  Activity, GitBranch, AlertTriangle, RefreshCw,
  CheckCircle, XCircle, TrendingUp, Database,
  Clock, Cpu, BarChart3, Shield, ChevronRight, Loader
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

async function mlopsFetch(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function mlopsPost(path) {
  const res = await fetch(`${API_BASE}${path}`, { method: 'POST' });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

/* ─── tiny helpers ───────────────────────────────────────────────── */

function StatusDot({ ok }) {
  return (
    <span style={{
      display: 'inline-block', width: 8, height: 8, borderRadius: '50%',
      background: ok ? '#10B981' : '#EF4444',
      boxShadow: ok ? '0 0 6px rgba(16,185,129,0.5)' : '0 0 6px rgba(239,68,68,0.5)',
      marginRight: 8,
    }} />
  );
}

function Metric({ label, value, sub, color }) {
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 700, color: color || 'var(--text-primary)' }}>
        {value ?? '—'}
      </div>
      {sub && <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

/* ─── main component ─────────────────────────────────────────────── */

export default function MLOps() {
  const [status, setStatus]           = useState(null);
  const [drift, setDrift]             = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [models, setModels]           = useState(null);
  const [loading, setLoading]         = useState(true);
  const [reloading, setReloading]     = useState(false);
  const [error, setError]             = useState(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [s, d, p, m] = await Promise.allSettled([
        mlopsFetch('/mlops/status'),
        mlopsFetch('/mlops/drift'),
        mlopsFetch('/mlops/predictions/summary'),
        mlopsFetch('/mlops/models'),
      ]);
      setStatus(s.status === 'fulfilled' ? s.value : null);
      setDrift(d.status === 'fulfilled' ? d.value : null);
      setPredictions(p.status === 'fulfilled' ? p.value : null);
      setModels(m.status === 'fulfilled' ? m.value : null);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  async function handleReload() {
    setReloading(true);
    try {
      await mlopsPost('/mlops/reload');
      await refresh();
    } catch (e) {
      setError('Reload failed: ' + e.message);
    } finally {
      setReloading(false);
    }
  }

  const tm = status?.training_metrics || {};
  const driftOk = drift && drift.status !== 'error' && !drift.drift_detected;
  const driftDetected = drift?.drift_detected;

  return (
    <div className="main-content">
      {/* Header */}
      <div className="page-header">
        <div>
          <h1 className="page-title">MLOps Dashboard</h1>
          <p className="page-subtitle">Model monitoring, drift detection & registry</p>
        </div>
        <div className="flex gap-8 items-center">
          <button
            className="btn btn-outline btn-sm"
            onClick={refresh}
            disabled={loading}
            style={{ display: 'flex', alignItems: 'center', gap: 6 }}
          >
            <RefreshCw size={13} className={loading ? 'spin' : ''} />
            Refresh
          </button>
          <button
            className="btn btn-primary btn-sm"
            onClick={handleReload}
            disabled={reloading}
            style={{ display: 'flex', alignItems: 'center', gap: 6 }}
          >
            {reloading ? <Loader size={13} className="spin" /> : <Cpu size={13} />}
            Reload Model
          </button>
        </div>
      </div>

      {error && (
        <div style={{
          background: 'var(--risk-critical-bg)', border: '1px solid var(--risk-critical-border)',
          borderRadius: 8, padding: '10px 16px', margin: '0 24px', color: 'var(--risk-critical)',
          fontSize: 13, display: 'flex', alignItems: 'center', gap: 8,
        }}>
          <AlertTriangle size={14} /> {error}
        </div>
      )}

      <div className="page-body">
        {/* ─── Row 1: Model Status + Training Metrics ─── */}
        <div className="bento-grid bento-4col">
          {/* Model Info Card */}
          <div className="card col-span-2">
            <div className="card-header">
              <span className="card-title"><Shield size={15} /> Active Model</span>
              {status?.mlflow_version && (
                <span className="badge badge-low" style={{ fontSize: 10 }}>
                  v{status.mlflow_version}
                </span>
              )}
            </div>
            <div className="card-body">
              {status ? (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                  <div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 2 }}>Type</div>
                    <div style={{ fontWeight: 600 }}>{status.model_type || 'XGBoost'}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 2 }}>Features</div>
                    <div style={{ fontWeight: 600 }}>{status.feature_count || 49}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 2 }}>Threshold</div>
                    <div style={{ fontWeight: 600 }}>{status.decision_threshold}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 2 }}>Git SHA</div>
                    <div style={{ fontFamily: 'monospace', fontSize: 12 }}>
                      {(status.git_sha || 'unknown').slice(0, 8)}
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 2 }}>Training Rows</div>
                    <div style={{ fontWeight: 600 }}>
                      {status.training_rows ? Number(status.training_rows).toLocaleString() : '—'}
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 2 }}>Fraud Rate</div>
                    <div style={{ fontWeight: 600 }}>
                      {status.fraud_rate ? `${(status.fraud_rate * 100).toFixed(2)}%` : '—'}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="empty-state"><Database /><p>Loading model info…</p></div>
              )}
            </div>
          </div>

          {/* Training Metrics */}
          <div className="stat-card">
            <div className="stat-label"><TrendingUp size={13} /> AUC-PR</div>
            <div className="stat-value" style={{ color: '#6366F1' }}>
              {tm.auc_pr != null ? tm.auc_pr.toFixed(4) : '—'}
            </div>
            <div className="stat-sub">Primary metric</div>
          </div>
          <div className="stat-card">
            <div className="stat-label"><BarChart3 size={13} /> AUC-ROC</div>
            <div className="stat-value" style={{ color: '#10B981' }}>
              {tm.auc_roc != null ? tm.auc_roc.toFixed(4) : '—'}
            </div>
            <div className="stat-sub">Classification power</div>
          </div>
        </div>

        {/* ─── Row 2: Drift Status + Predictions Summary ─── */}
        <div className="bento-grid bento-3col mt-16">
          {/* Drift Card */}
          <div className="card col-span-2">
            <div className="card-header">
              <span className="card-title"><Activity size={15} /> Drift Detection</span>
              {drift && (
                <span className={`badge ${driftDetected ? 'badge-critical' : 'badge-low'}`}
                      style={{ fontSize: 10 }}>
                  {driftDetected ? '⚠ DRIFT DETECTED' : '● STABLE'}
                </span>
              )}
            </div>
            <div className="card-body">
              {drift ? (
                drift.status === 'error' ? (
                  <div className="empty-state">
                    <AlertTriangle />
                    <p>{drift.message || 'Could not compute drift. Train a model with MLflow first.'}</p>
                  </div>
                ) : (
                  <>
                    <div style={{
                      display: 'flex', justifyContent: 'space-around',
                      padding: '12px 0', borderBottom: '1px solid var(--border)', marginBottom: 16,
                    }}>
                      <Metric
                        label="Predictions Analysed"
                        value={drift.predictions_analysed ?? drift.predictions_logged ?? drift.n_predictions ?? drift.total_predictions ?? 0}
                      />
                      <Metric
                        label="Drifted Features"
                        value={drift.n_drifted_features ?? 0}
                        color={drift.n_drifted_features > 0 ? '#DC2626' : '#10B981'}
                      />
                      <Metric
                        label="Prediction Drift"
                        value={drift.prediction_drift_detected ? 'YES' : 'NO'}
                        color={drift.prediction_drift_detected ? '#DC2626' : '#10B981'}
                      />
                    </div>

                    {/* Top drifted features */}
                    {drift.drifted_features && drift.drifted_features.length > 0 && (
                      <div>
                        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8, color: 'var(--text-secondary)' }}>
                          Top Drifted Features (PSI &gt; 0.25)
                        </div>
                        <table className="data-table">
                          <thead>
                            <tr>
                              <th>Feature</th>
                              <th>PSI</th>
                              <th>Status</th>
                            </tr>
                          </thead>
                          <tbody>
                            {drift.drifted_features.slice(0, 8).map((f, i) => (
                              <tr key={i}>
                                <td className="font-mono" style={{ fontSize: 12 }}>{f.feature}</td>
                                <td style={{ fontWeight: 600, color: f.psi > 0.25 ? '#DC2626' : '#10B981' }}>
                                  {f.psi.toFixed(4)}
                                </td>
                                <td>
                                  {f.psi > 0.25
                                    ? <span className="badge badge-critical" style={{ fontSize: 10 }}>DRIFTED</span>
                                    : <span className="badge badge-low" style={{ fontSize: 10 }}>OK</span>
                                  }
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}

                    {drift.status === 'insufficient_data' && (
                      <div style={{ textAlign: 'center', padding: '16px 8px' }}>
                        <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: 4 }}>
                          Accumulating Production Baseline
                        </div>
                        <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                          {drift.predictions_logged ?? 0} of {drift.minimum_required ?? 100} predictions logged.
                          Full PSI distribution check activates at {drift.minimum_required ?? 100} samples.
                        </div>
                      </div>
                    )}

                    {drift.drifted_features && drift.drifted_features.length === 0 && (
                      <div style={{ textAlign: 'center', color: '#10B981', padding: 16 }}>
                        <CheckCircle size={32} style={{ marginBottom: 8 }} />
                        <div style={{ fontWeight: 600 }}>All features stable</div>
                        <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                          No PSI &gt; 0.25 detected
                        </div>
                      </div>
                    )}
                  </>
                )
              ) : (
                <div className="empty-state">
                  <Activity />
                  <p>Loading drift status…</p>
                </div>
              )}
            </div>
          </div>

          {/* Predictions Summary */}
          <div className="card">
            <div className="card-header">
              <span className="card-title"><BarChart3 size={15} /> Predictions</span>
            </div>
            <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              {predictions ? (
                predictions.total_predictions > 0 ? (
                  <>
                    <Metric
                      label="Total Predictions"
                      value={predictions.total_predictions.toLocaleString()}
                    />
                    <div style={{ borderTop: '1px solid var(--border)', paddingTop: 12 }}>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>
                        Risk Tier Distribution
                      </div>
                      {predictions.risk_tier_distribution &&
                        Object.entries(predictions.risk_tier_distribution).map(([tier, count]) => {
                          const colors = {
                            CRITICAL: '#DC2626', HIGH: '#F59E0B',
                            MEDIUM: '#3B82F6', LOW: '#10B981',
                          };
                          const pct = ((count / predictions.total_predictions) * 100).toFixed(1);
                          return (
                            <div key={tier} style={{
                              display: 'flex', alignItems: 'center', gap: 8,
                              marginBottom: 6, fontSize: 12,
                            }}>
                              <span style={{
                                width: 8, height: 8, borderRadius: '50%',
                                background: colors[tier] || '#64748B', flexShrink: 0,
                              }} />
                              <span style={{ flex: 1, color: 'var(--text-secondary)' }}>{tier}</span>
                              <span style={{ fontWeight: 600 }}>{count}</span>
                              <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>{pct}%</span>
                            </div>
                          );
                        })
                      }
                    </div>
                    {predictions.avg_fraud_probability != null && (
                      <div style={{ borderTop: '1px solid var(--border)', paddingTop: 12 }}>
                        <Metric
                          label="Avg Fraud Probability"
                          value={predictions.avg_fraud_probability.toFixed(4)}
                          color="#F59E0B"
                        />
                      </div>
                    )}
                    {predictions.time_range?.earliest && (
                      <div style={{
                        fontSize: 11, color: 'var(--text-muted)',
                        borderTop: '1px solid var(--border)', paddingTop: 8,
                      }}>
                        <Clock size={11} style={{ marginRight: 4 }} />
                        {new Date(predictions.time_range.earliest).toLocaleDateString()} — {new Date(predictions.time_range.latest).toLocaleDateString()}
                      </div>
                    )}
                  </>
                ) : (
                  <div className="empty-state">
                    <BarChart3 />
                    <p>No predictions logged yet</p>
                    <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                      Run an investigation to start logging
                    </span>
                  </div>
                )
              ) : (
                <div className="empty-state"><Loader className="spin" /><p>Loading…</p></div>
              )}
            </div>
          </div>
        </div>

        {/* ─── Row 3: Model Registry ─── */}
        <div className="card mt-16">
          <div className="card-header">
            <span className="card-title"><GitBranch size={15} /> Model Registry (MLflow)</span>
            <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
              {models?.total != null ? `${models.total} version${models.total !== 1 ? 's' : ''}` : ''}
            </span>
          </div>
          <div className="card-body" style={{ padding: 0 }}>
            {models && models.models && models.models.length > 0 ? (
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Version</th>
                    <th>Alias</th>
                    <th>AUC-PR (drifted)</th>
                    <th>Recall (drifted)</th>
                    <th>AUC-PR (canonical)</th>
                    <th>Created</th>
                    <th>Run ID</th>
                  </tr>
                </thead>
                <tbody>
                  {models.models.map((m, i) => {
                    const isChampion = m.aliases?.includes('champion');
                    return (
                      <tr key={i} style={isChampion ? { background: 'var(--risk-low-bg)' } : {}}>
                        <td style={{ fontWeight: 700 }}>v{m.version}</td>
                        <td>
                          {m.aliases?.length > 0
                            ? m.aliases.map(a => (
                                <span key={a} className={`badge ${a === 'champion' ? 'badge-low' : 'badge-medium'}`}
                                      style={{ fontSize: 10, marginRight: 4 }}>
                                  {a}
                                </span>
                              ))
                            : <span style={{ color: 'var(--text-muted)' }}>—</span>
                          }
                        </td>
                        <td style={{ fontWeight: 600 }}>
                          {m.metrics?.drifted_auc_pr != null ? m.metrics.drifted_auc_pr.toFixed(4) : '—'}
                        </td>
                        <td>
                          {m.metrics?.drifted_recall != null ? m.metrics.drifted_recall.toFixed(4) : '—'}
                        </td>
                        <td>
                          {m.metrics?.canonical_auc_pr != null ? m.metrics.canonical_auc_pr.toFixed(4) : '—'}
                        </td>
                        <td style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                          {m.creation_timestamp
                            ? new Date(m.creation_timestamp).toLocaleDateString()
                            : '—'}
                        </td>
                        <td className="font-mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                          {m.run_id ? m.run_id.slice(0, 8) : '—'}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            ) : (
              <div className="empty-state" style={{ padding: 32 }}>
                <GitBranch />
                <p>{models?.message || 'No models registered in MLflow yet.'}</p>
                <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                  Run training to register the first model version
                </span>
              </div>
            )}
          </div>
        </div>

        {/* ─── Row 4: System Info ─── */}
        <div className="bento-grid bento-3col mt-16" style={{ marginBottom: 32 }}>
          <div className="stat-card">
            <div className="stat-label"><Clock size={13} /> Last Reloaded</div>
            <div style={{ fontSize: 13, fontWeight: 600, marginTop: 8 }}>
              {status?.last_reloaded_at
                ? new Date(status.last_reloaded_at).toLocaleString()
                : 'At startup'}
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-label"><Database size={13} /> Model Modified</div>
            <div style={{ fontSize: 13, fontWeight: 600, marginTop: 8 }}>
              {status?.model_file_modified_at
                ? new Date(status.model_file_modified_at).toLocaleString()
                : '—'}
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-label"><Activity size={13} /> Drift Status</div>
            <div style={{ fontSize: 14, fontWeight: 700, marginTop: 8, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <StatusDot ok={driftOk} />
              {driftDetected ? 'Drift Detected' : drift?.status === 'error' ? 'Unavailable' : 'Stable'}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

