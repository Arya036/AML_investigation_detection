"""
api/main.py — Sentinel AI FastAPI Backend  
=====================================================
Exposes the LangGraph agent and tool suite as a REST API.
The React frontend consumes these endpoints.

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /health                   — platform stats
    POST /investigate              — start investigation job (async)
    GET  /investigate/{job_id}     — poll job status / get result
    POST /tools/graph              — build transaction network graph
    POST /tools/rag                — search regulatory corpus
    GET  /alerts                   — list database alerts
    GET  /transactions/{account_id}— raw transaction history
"""

import os
import sys
import json
import sqlite3
import threading
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Path setup ───────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sentinel.api")

DB_PATH     = os.environ.get("PS6_DB_PATH",    str(ROOT / "fundflow.db"))
CHROMA_PATH = os.environ.get("CHROMA_DB_PATH", str(ROOT / "rag" / "vector_store"))

DISCLAIMER = (
    "AI-generated DRAFT for internal compliance use only. NOT court-admissible. "
    "Requires compliance officer review."
)

# ── In-memory job store ───────────────────────────────────────
# Format: { job_id: { status, account_id, payload, error, started_at } }
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

# ── App setup ───────────────────────────────────────────
app = FastAPI(
    title="Sentinel AI — AML Investigation API",
    description="Autonomous AML investigation platform for PS6 hackathon.",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:3000",
        "http://localhost:4173",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:4173",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)


# ── MLOps dashboard routes ────────────────────────────────
try:
    from mlops.dashboard import router as mlops_router
    app.include_router(mlops_router)
except ImportError:
    pass  # mlops module not available


# ── DB helper ────────────────────────────────────────────
def _get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _db_stats() -> dict:
    try:
        conn = _get_db()
        txns   = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        fraud  = conn.execute("SELECT COUNT(*) FROM transactions WHERE is_fraud=1").fetchone()[0]
        alerts = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
        conn.close()
        fraud_rate = round(100 * fraud / txns, 2) if txns else 0
        return {"transactions": txns, "fraud_labeled": fraud, "alerts": alerts,
                "fraud_rate_pct": fraud_rate,
                "data_note": "Synthetic PaySim dataset. Amounts are synthetic units, not INR."}
    except Exception as e:
        logger.error("DB stats error: %s", e)
        return {"transactions": 0, "fraud_labeled": 0, "alerts": 0, "error": str(e)}


def _chroma_ready() -> bool:
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        col = client.get_collection("sentinel_regulations")
        return col.count() > 0
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════
# DETERMINISTIC SECTION BUILDER  (no LLM required)
# ═══════════════════════════════════════════════════════════════

def _load_tool_json(tool_func, *args, **kwargs):
    """Call a LangChain tool's underlying .func and parse its JSON string result.
    Returns {} on any failure so one broken tool never blanks the whole card set."""
    try:
        raw = tool_func(*args, **kwargs)
        if isinstance(raw, (dict, list)):
            return raw
        return json.loads(raw)
    except Exception as e:
        logger.error("Tool call %s failed: %s", getattr(tool_func, "__name__", tool_func), e)
        return {}


def _pick_number(feature: dict):
    """Return shap_value if present (even when it is 0.0), else contribution, else 0.0.
    NOTE: uses explicit None checks so a legitimate 0.0 SHAP value is NOT dropped."""
    sv = feature.get("shap_value")
    if sv is None:
        sv = feature.get("contribution")
    if sv is None:
        sv = 0.0
    return sv


def _build_sections(account_id: str, result: Optional[dict]) -> dict:
    """Build all structured cards from tool/DB/model outputs.

    Prefers structured data already attached to `result` (from a successful
    agent run). For anything missing — including when the agent failed and
    `result` is empty — it recomputes directly via the tools' .func (pure
    DB / engine / embedding calls, NO LLM). Each section is independently
    fault-tolerant.
    """
    result = result or {}

    from agent.tools import (
        get_transaction_history,
        get_transaction_graph,
        score_risk,
        detect_typology,
        search_regulations,
    )

    # 1. Account Summary ────────────────────────────────────────
    summary = (result.get("history_data") or {}).get("summary") or {}
    if not summary:
        summary = _load_tool_json(get_transaction_history.func, account_id).get("summary") or {}

    account_summary = {
        "total_transactions":    summary.get("total_transactions"),
        "transactions_sent":     summary.get("transactions_sent"),
        "transactions_received": summary.get("transactions_received"),
        "total_amount_sent":     summary.get("total_amount_sent"),
        "total_amount_received": summary.get("total_amount_received"),
        "avg_amount":            summary.get("avg_amount"),
        "max_amount":            summary.get("max_amount"),
        "fraud_flagged_count":   summary.get("fraud_flagged_count"),
        "high_risk_count":       summary.get("high_risk_count"),
        "date_range":            summary.get("date_range"),
    }

    # 2. Graph Intelligence ─────────────────────────────────────
    gdata = result.get("graph_data") or {}
    if not gdata:
        gdata = _load_tool_json(get_transaction_graph.func, account_id)
    profile = gdata.get("graph_profile") or {}
    graph_intelligence = {
        "mule_score":        gdata.get("mule_score"),
        "is_suspected_mule": gdata.get("is_suspected_mule"),
        "in_ring":           gdata.get("in_ring"),
        "ring_count":        gdata.get("ring_count"),
        "ring_ids":          gdata.get("ring_ids") or [],
        "graph_profile": {
            "in_degree":     profile.get("in_degree"),
            "out_degree":    profile.get("out_degree"),
            "net_flow":      profile.get("net_flow"),
            "max_fraud_prob": profile.get("max_fraud_prob"),
        },
        "connected_nodes":   gdata.get("connected_nodes") or [],
    }

    # 3. ML Risk + SHAP ────────────────────────────────────────
    rdata = result.get("risk_data") or {}
    if not rdata:
        # NOTE: keep the same call signature the agent uses. If score_risk needs
        # real transaction features rather than {}, fix it inside agent.tools so
        # both the agent and this fallback stay consistent.
        rdata = _load_tool_json(score_risk.func, account_id, {})

    top_feats = []
    for f in rdata.get("top_features") or []:
        val = _pick_number(f)
        top_feats.append({
            "feature":      f.get("feature"),
            "shap_value":   val,
            "contribution": val,
            "direction":    f.get("direction", ""),
        })
    ml_risk = {
        "top_shap_features": top_feats,
        "decision_threshold": rdata.get("decision_threshold", 0.7),
    }
    risk_tier         = rdata.get("risk_tier", "UNKNOWN")
    fraud_probability = rdata.get("fraud_probability")

    # 4. Typologies ───────────────────────────────────────────
    typs = result.get("typologies") or []
    if not typs:
        typs = _load_tool_json(detect_typology.func, account_id).get("typologies") or []
    typologies = [
        {"type": t.get("type"), "description": t.get("description"), "risk": t.get("risk")}
        for t in typs
    ]

    # 5. Regulatory Citations (finding-driven) ──────────────────────────
    regs = result.get("regulations") or []
    if not regs:
        typology_types = [t.get("type") for t in typologies]
        if "MULE_ACCOUNT" in typology_types:
            rag_query = "money mule layering intermediary shell accounts"
        elif "ROUND_TRIPPING" in typology_types:
            rag_query = "circular trading round tripping fund flow loops"
        elif "LARGE_VALUE_CLUSTERING" in typology_types:
            rag_query = "structuring suspicious transaction threshold avoidance reporting"
        elif "SMURFING_FAN_IN" in typology_types:
            rag_query = "smurfing structuring fan-in multiple senders"
        else:
            rag_query = "general compliance transaction monitoring reporting guidelines"
        regs = _load_tool_json(search_regulations.func, rag_query, top_k=3).get("results") or []

    regulatory_citations = [
        {"rank": i + 1, "source": r.get("source"), "page": r.get("page"), "text": r.get("text")}
        for i, r in enumerate(regs)
    ]

    recommendation = (
        "ESCALATE"
        if risk_tier in ("CRITICAL", "HIGH") or graph_intelligence.get("is_suspected_mule")
        else "REVIEW"
    )

    return {
        "account_summary":      account_summary,
        "graph_intelligence":   graph_intelligence,
        "ml_risk":              ml_risk,
        "risk_tier":            risk_tier,
        "fraud_probability":    fraud_probability,
        "regulatory_citations": regulatory_citations,
        "typologies":           typologies,
        "recommendation":       recommendation,
        "disclaimer":           DISCLAIMER,
    }


# ═══════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════

# ── Health ──────────────────────────────────────────────────
@app.get("/health", tags=["Platform"])
def health():
    """Returns platform stats for the landing page stats strip."""
    db = _db_stats()
    return {
        "status": "ok",
        "database": {
            "total_transactions": db.get("transactions", 0),
            "total_alerts":       db.get("alerts", 0),
            "fraud_labeled":      db.get("fraud_labeled", 0),
            "fraud_rate":         db.get("fraud_rate_pct", 0),
            "data_note":          db.get("data_note", ""),
        },
        "model":        "XGBoost (PaySim-trained, FundFlow engine)",
        "rag_ingested": _chroma_ready(),
        "agent":        "LangGraph ReAct (GPT-4o-mini)",
        "version":      "1.1.0",
    }


# ── Investigate ────────────────────────────────────────────
class InvestigateRequest(BaseModel):
    account_id: str


def _run_investigation(job_id: str, account_id: str):
    """Background thread.

    Order matters:
      1) Try the LLM agent for the STR prose + guardrails (best effort).
      2) ALWAYS build the deterministic cards afterwards, preferring any
         structured data the agent produced, otherwise recomputing via tools.

    A rate-limited / failed agent no longer blanks the investigation — it only
    drops the narrative prose, and llm_status records why.
    """
    result: dict = {}
    llm_status = "ok"
    llm_error: Optional[str] = None

    # 1) Best-effort LLM agent run.
    try:
        from agent.orchestrator import run_investigation
        result = run_investigation(account_id) or {}
    except Exception as e:
        llm_status = "unavailable"
        llm_error = str(e)
        logger.warning(
            "Agent LLM run failed for %s (%s) — serving deterministic cards only",
            account_id, e,
        )

    # 2) Deterministic cards ALWAYS. This is the resilience guarantee.
    try:
        sections = _build_sections(account_id, result)
    except Exception as e:
        # Only a hard DB/engine failure lands here — genuinely unrecoverable.
        logger.error("Deterministic section build failed for %s: %s", account_id, e)
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(e)
        return

    raw_trace = result.get("tool_trace", [])
    g = result.get("guardrails", {}) or {}

    llm_note = None
    if llm_status != "ok":
        llm_note = (
            "AI narrative unavailable (LLM rate-limited or failed). "
            "All metrics below are computed directly from the database and models."
        )

    payload = {
        "job_id":            job_id,
        "status":           "done",
        "account_id":       account_id,
        "tool_trace":       [{"tool": t, "status": "done"} for t in raw_trace],
        "risk_tier":        sections.get("risk_tier"),
        "fraud_probability": sections.get("fraud_probability"),
        "guardrails": {
            "passed":     g.get("passed", False),
            "violations": g.get("violations", []),
            "warnings":   g.get("warnings", []),
        },
        "str_sections":   sections,
        "str_draft_text": result.get("str_draft", ""),
        "llm_status":     llm_status,   # "ok" | "unavailable"
        "llm_error":      llm_error,
        "llm_note":       llm_note,
    }

    with _jobs_lock:
        _jobs[job_id]["status"] = "done"
        _jobs[job_id]["payload"] = payload


@app.post("/investigate", status_code=202, tags=["Investigation"])
def start_investigation(req: InvestigateRequest):
    """
    Start an asynchronous investigation. Returns job_id to poll.
    Poll GET /investigate/{job_id} every 2 seconds until status == 'done'.
    """
    if not req.account_id or not req.account_id.strip():
        raise HTTPException(status_code=400, detail="account_id is required")

    account_id = req.account_id.strip()
    job_id = f"inv_{account_id}_{int(datetime.now(timezone.utc).timestamp())}"

    with _jobs_lock:
        _jobs[job_id] = {
            "job_id":     job_id,
            "account_id": account_id,
            "status":     "running",
            "payload":    None,
            "error":      None,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

    t = threading.Thread(target=_run_investigation, args=(job_id, account_id), daemon=True)
    t.start()

    return {
        "job_id":            job_id,
        "account_id":        account_id,
        "status":            "running",
        "estimated_seconds": 60,
        "poll_url":          f"/investigate/{job_id}",
    }


@app.get("/investigate/{job_id}", tags=["Investigation"])
def get_investigation(job_id: str):
    """Poll investigation status. Returns full result when status == 'done'."""
    with _jobs_lock:
        job = _jobs.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    if job["status"] == "error":
        # Only reached on a hard DB/engine failure, NOT on an LLM rate limit.
        return {"job_id": job_id, "status": "error", "error": job.get("error", "Unknown error")}

    if job["status"] == "running":
        return {"job_id": job_id, "account_id": job["account_id"], "status": "running", "tool_trace": []}

    # Done — payload was fully assembled in the worker thread.
    return job.get("payload") or {"job_id": job_id, "status": "done"}


# ── Graph tool ─────────────────────────────────────────────
class GraphRequest(BaseModel):
    account_id: str
    max_hops: int = 2


def _graph_edges_from_sql(account_id: str):
    """Shared SQL edge/node builder used by the primary path and the fallback."""
    conn = _get_db()
    rows = conn.execute("""
        SELECT sender_account, receiver_account, amount,
               fraud_probability, txn_type
        FROM transactions
        WHERE sender_account = ? OR receiver_account = ?
        LIMIT 200
    """, (account_id, account_id)).fetchall()
    conn.close()

    nodes_set = set()
    edges = []
    for r in rows:
        nodes_set.add(r["sender_account"])
        nodes_set.add(r["receiver_account"])
        edges.append({
            "from":       r["sender_account"],
            "to":         r["receiver_account"],
            "amount":     round(r["amount"] or 0, 2),
            "fraud_prob": round(r["fraud_probability"] or 0, 4),
            "txn_type":   r["txn_type"] or "UNKNOWN",
        })
    nodes = [{"id": n, "is_subject": n == account_id} for n in nodes_set]
    return nodes, edges


@app.post("/tools/graph", tags=["Tools"])
def get_graph(req: GraphRequest):
    """Build the multi-hop transaction network for an account. Used by Graph View page.

    Tier-2 engine: expands the account's neighbourhood up to max_hops from SQLite,
    builds the real FundFlowGraph, then runs causal fund-flow tracing, mule scoring,
    and ring (cycle) detection. Any failure degrades gracefully to a flat SQL view.
    """
    try:
        import pandas as pd
        from graph.fund_flow import FundFlowGraph
        from graph.mule_detector import compute_mule_scores
        from graph.ring_detector import find_cycles, get_ring_summary

        max_hops     = max(1, min(int(req.max_hops or 2), 4))   # clamp 1..4
        MAX_TXNS     = 4000     # hard cap on edges pulled (memory/latency guard)
        MAX_FRONTIER = 300      # cap accounts expanded per hop (SQL IN-size guard)

        # --- 1. BFS-expand the neighbourhood from SQLite -------------------
        conn = _get_db()
        seen_accounts = {req.account_id}
        frontier      = [req.account_id]
        rows_by_txn   = {}
        for _hop in range(max_hops):
            if not frontier or len(rows_by_txn) >= MAX_TXNS:
                break
            batch_accounts = frontier[:MAX_FRONTIER]
            ph = ",".join("?" * len(batch_accounts))
            q = (
                "SELECT txn_id, timestamp, sender_account, receiver_account, "
                "amount, txn_type, fraud_probability, is_fraud "
                "FROM transactions "
                f"WHERE sender_account IN ({ph}) OR receiver_account IN ({ph}) "
                "LIMIT ?"
            )
            batch = conn.execute(
                q, batch_accounts + batch_accounts + [MAX_TXNS]
            ).fetchall()
            next_frontier = []
            for r in batch:
                if len(rows_by_txn) >= MAX_TXNS:
                    break
                if r["txn_id"] in rows_by_txn:
                    continue
                rows_by_txn[r["txn_id"]] = dict(r)
                for acct in (r["sender_account"], r["receiver_account"]):
                    if acct not in seen_accounts:
                        seen_accounts.add(acct)
                        next_frontier.append(acct)
            frontier = next_frontier
        conn.close()

        if not rows_by_txn:
            return {
                "account_id": req.account_id,
                "nodes": [], "edges": [],
                "stats": {"total_nodes": 0, "total_edges": 0, "high_risk_edges": 0},
                "graph_analysis": {}, "engine": "graph",
            }

        df = pd.DataFrame(list(rows_by_txn.values()))

        # --- 2. Build the real fund-flow graph ----------------------------
        fg = FundFlowGraph()
        fg.build_from_df(df)
        Gnx = fg.G

        # --- 3. Mule scoring (needs a 'step' column; synthesize if absent) -
        if "step" not in df.columns:
            _ts = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.assign(
                step=((_ts - _ts.min()).dt.total_seconds() // 3600)
                .fillna(0).astype(int)
            )
        try:
            mule_df = compute_mule_scores(Gnx, df)
        except Exception as _me:
            logger.warning("mule scoring failed: %s", _me)
            mule_df = pd.DataFrame(
                columns=["account", "mule_score", "is_suspected_mule",
                         "passthrough_ratio", "unique_senders",
                         "unique_receivers", "avg_fwd_delay_min"]
            )
        mule_lookup = (dict(zip(mule_df["account"], mule_df["mule_score"]))
                       if len(mule_df) else {})

        subj = mule_df[mule_df["account"] == req.account_id] if len(mule_df) else mule_df
        if len(subj):
            m = subj.iloc[0]
            mule_score = float(m["mule_score"])
            is_mule    = bool(int(m["is_suspected_mule"]))
        else:
            mule_score, is_mule = 0.0, False

        # Graph-metrics profile for the subject, read straight from the graph so
        # it always populates. Frontend (GraphView.jsx) expects exactly:
        # in_degree, out_degree, total_received, total_sent,
        # passthrough_ratio, fan_out_ratio.
        acct = req.account_id
        if acct in Gnx:
            _node = Gnx.nodes[acct]
            _in_deg  = int(Gnx.in_degree(acct))      # incoming transfer edges
            _out_deg = int(Gnx.out_degree(acct))     # outgoing transfer edges
            _dist_senders   = len(set(Gnx.predecessors(acct)))
            _dist_receivers = len(set(Gnx.successors(acct)))
            _recv = float(_node.get("total_received", 0) or 0)
            _sent = float(_node.get("total_sent", 0) or 0)
            _passthrough = min(_sent / _recv, 1.0) if _recv > 0 else 0.0
            _fan_out = (_dist_receivers / _dist_senders) if _dist_senders > 0 else float(_dist_receivers)
            graph_profile = {
                "in_degree":         _in_deg,
                "out_degree":        _out_deg,
                "total_received":    round(_recv, 2),
                "total_sent":        round(_sent, 2),
                "passthrough_ratio": round(_passthrough, 4),
                "fan_out_ratio":     round(_fan_out, 4),
                "unique_senders":    _dist_senders,
                "unique_receivers":  _dist_receivers,
            }
        else:
            graph_profile = {}

        # --- 4. Causal multi-hop fund-flow trace from the subject ---------
        try:
            trace = fg.trace_fund_flow(req.account_id, max_hops=max_hops)
        except Exception as _te:
            logger.warning("fund-flow trace failed: %s", _te)
            trace = {"paths": [], "nodes": [], "edges": [], "summary": {}}

        # --- 5. Ring / cycle detection (guarded: simple_cycles can explode)-
        subj_rings, ring_summary = [], {}
        if Gnx.number_of_nodes() <= 200 and Gnx.number_of_edges() <= 600:
            try:
                rings        = find_cycles(Gnx, max_length=6)
                subj_rings   = [r for r in rings if req.account_id in r["accounts"]]
                ring_summary = get_ring_summary(rings)
            except Exception as _re:
                logger.warning("ring detection failed: %s", _re)
        else:
            ring_summary = {"note": "ring detection skipped (neighbourhood too large)"}

        # --- 6. Build nodes + edges for the frontend from the real graph --
        nodes = [
            {
                "id":         n,
                "is_subject": n == req.account_id,
                "mule_score": round(float(mule_lookup.get(n, 0.0)), 4),
            }
            for n in Gnx.nodes()
        ]
        edges = []
        for u, v, data in Gnx.edges(data=True):
            edges.append({
                "from":       u,
                "to":         v,
                "amount":     round(float(data.get("amount", 0) or 0), 2),
                "fraud_prob": round(float(data.get("fraud_prob", 0) or 0), 4),
                "txn_type":   data.get("txn_type") or "UNKNOWN",
            })
        high_risk = sum(1 for e in edges if e["fraud_prob"] > 0.7)

        return {
            "account_id": req.account_id,
            "nodes":      nodes,
            "edges":      edges,
            "stats": {
                "total_nodes":     len(nodes),
                "total_edges":     len(edges),
                "high_risk_edges": high_risk,
                "hops_expanded":   max_hops,
            },
            "graph_analysis": {
                "mule_score":        mule_score,
                "is_suspected_mule": is_mule,
                "in_ring":           len(subj_rings) > 0,
                "ring_count":        len(subj_rings),
                "rings":             subj_rings[:5],
                "ring_summary":      ring_summary,
                "graph_profile":     graph_profile,
                "fund_flow":         trace.get("summary", {}),
                "fund_flow_paths":   trace.get("edges", [])[:50],
            },
            "engine": "graph",
        }

    except Exception as e:
        logger.warning("Graph engine failed (%s), using SQL fallback", e)
        try:
            nodes, edges = _graph_edges_from_sql(req.account_id)
            return {
                "account_id": req.account_id,
                "nodes":      nodes,
                "edges":      edges,
                "stats":      {"total_nodes": len(nodes), "total_edges": len(edges)},
                "graph_analysis": {},
                "engine": "sql_fallback",
            }
        except Exception as e2:
            raise HTTPException(status_code=500, detail=str(e2))


# ── RAG tool ──────────────────────────────────────────────
class RagRequest(BaseModel):
    query: str
    top_k: int = 3


@app.post("/tools/rag", tags=["Tools"])
def rag_search(req: RagRequest):
    """Semantic search over the regulatory corpus. Used by RAG Lookup page."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")
    try:
        from rag.retriever import retrieve_regulations
        return retrieve_regulations(req.query, top_k=min(req.top_k, 10))
    except Exception as e:
        logger.error("RAG error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ── Alerts ───────────────────────────────────────────────
def _normalize_accounts_involved(val):
    """Coerce accounts_involved into a clean list[str] regardless of stored form."""
    if not val:
        return []
    if isinstance(val, list):
        return [str(x) for x in val]
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
            return [str(parsed)]
        except json.JSONDecodeError:
            if "," in val:
                return [x.strip() for x in val.split(",") if x.strip()]
            return [val.strip()]
    return [str(val)]


@app.get("/alerts", tags=["Alerts"])
def list_alerts(
    severity: Optional[str] = Query(None, description="Filter: CRITICAL, HIGH, MEDIUM, LOW"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """List database alerts. Supports severity filter and pagination."""
    try:
        conn = _get_db()
        if severity and severity.upper() != "ALL":
            total = conn.execute(
                "SELECT COUNT(*) FROM alerts WHERE severity=?", (severity.upper(),)
            ).fetchone()[0]
            rows = conn.execute(
                "SELECT * FROM alerts WHERE severity=? ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                (severity.upper(), limit, offset),
            ).fetchall()
        else:
            total = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
            rows = conn.execute(
                "SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        conn.close()

        alerts = []
        for row in rows:
            a = dict(row)
            a["accounts_involved"] = _normalize_accounts_involved(a.get("accounts_involved"))
            alerts.append(a)

        return {"total": total, "limit": limit, "offset": offset, "alerts": alerts}
    except Exception as e:
        logger.error("Alerts error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ── Transactions ───────────────────────────────────────────
@app.get("/transactions/{account_id}", tags=["Data"])
def get_transactions(
    account_id: str,
    limit: int = Query(50, ge=1, le=200),
):
    """Raw transaction history for an account."""
    try:
        conn = _get_db()
        rows = conn.execute("""
            SELECT txn_id, timestamp, sender_account, receiver_account,
                   amount, txn_type, channel, risk_tier,
                   fraud_probability, is_fraud
            FROM transactions
            WHERE sender_account = ? OR receiver_account = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (account_id, account_id, limit)).fetchall()
        conn.close()
        return {
            "account_id":   account_id,
            "count":        len(rows),
            "transactions": [dict(r) for r in rows],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Root ──────────────────────────────────���─────────────
@app.get("/", tags=["Platform"])
def root():
    return {
        "name":    "Sentinel AI API",
        "version": "1.1.0",
        "docs":    "/docs",
        "health":  "/health",
    }
