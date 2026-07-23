import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Layout from '../components/Layout';
import { graphAPI } from '../services/api';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faBuilding, faNewspaper, faTags, faProjectDiagram,
  faComments, faCheckCircle, faTimesCircle, faArrowRight, faMinusCircle
} from '@fortawesome/free-solid-svg-icons';

const StatCard = ({ icon, label, value, sub, color = 'var(--accent)' }) => (
  <div className="stat-card">
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
      <div style={{
        width: 36, height: 36, borderRadius: 10,
        background: `${color}20`, display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}>
        <FontAwesomeIcon icon={icon} style={{ color, fontSize: 15 }} />
      </div>
      <span className="stat-label">{label}</span>
    </div>
    <div className="stat-value">{value}</div>
    {sub && <div className="stat-sub">{sub}</div>}
  </div>
);

const Dashboard = () => {
  const navigate = useNavigate();
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    graphAPI.getStats()
      .then(r => setStats(r.data))
      .catch(() => setStats({ neo4j_connected: false }))
      .finally(() => setLoading(false));
  }, []);

  const fmt = (n) => n?.toLocaleString() ?? '—';

  return (
    <Layout>
      {/* Hero banner */}
      <div style={{
        background: 'linear-gradient(135deg, rgba(59,130,246,0.1), rgba(6,182,212,0.06))',
        border: '1px solid rgba(59,130,246,0.2)',
        borderRadius: 16, padding: '28px 32px', marginBottom: 24,
        display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between',
        flexWrap: 'wrap', gap: 16,
      }}>
        <div>
          <div style={{ fontSize: 11, color: 'var(--cyan)', fontWeight: 700, letterSpacing: '1px', textTransform: 'uppercase', marginBottom: 8 }}>
            Temporal Knowledge Graph RAG
          </div>
          <h1 style={{ fontSize: 26, fontWeight: 700, margin: '0 0 8px', lineHeight: 1.3 }}>
            Welcome to TemporalRAG
          </h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: 14, maxWidth: 480, lineHeight: 1.7 }}>
            Explore AI/IT company knowledge graphs and generate temporally-bounded
            explanations for B-MTGNN stock predictions using Neo4j + DeepSeek-R1.
          </p>
        </div>
        <div style={{ display: 'flex', gap: 10 }}>
          <button className="btn btn-primary" onClick={() => navigate('/chat')}>
            <FontAwesomeIcon icon={faComments} /> Start Chat
          </button>
          <button className="btn btn-ghost" onClick={() => navigate('/graph')}>
            <FontAwesomeIcon icon={faProjectDiagram} /> View Graph
          </button>
        </div>
      </div>

      {/* System Status */}
      <div style={{ marginBottom: 16, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <h2 style={{ fontSize: 15, fontWeight: 600 }}>System Status</h2>
        <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
          {loading ? 'Loading...' : 'Live'}
        </span>
      </div>

      {/* Connection Status Row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12, marginBottom: 24 }}>
        {[
          { label: 'Neo4j Database', ok: stats?.neo4j_connected, detail: 'bolt://localhost:7687' },
          { label: 'vLLM Embedding', ok: stats?.vllm_embedding_connected, detail: 'localhost:8000' },
          { label: 'vLLM LLM', ok: stats?.vllm_llm_connected, detail: 'localhost:8001' },
        ].map(s => (
          <div key={s.label} className="card" style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <FontAwesomeIcon
              icon={s.ok === true ? faCheckCircle : s.ok === false ? faTimesCircle : faMinusCircle}
              style={{ color: s.ok ? 'var(--green)' : s.ok === null ? 'var(--yellow)' : 'var(--red)', fontSize: 18 }}
            />
            <div>
              <div style={{ fontWeight: 600, fontSize: 13 }}>{s.label}</div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'JetBrains Mono, monospace' }}>
                {s.detail}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Neo4j Stats */}
      <div style={{ marginBottom: 16 }}>
        <h2 style={{ fontSize: 15, fontWeight: 600 }}>Knowledge Graph Statistics</h2>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 16, marginBottom: 32 }}>
        <StatCard icon={faBuilding} label="Companies" value={loading ? '—' : fmt(stats?.company_count)} sub="Target AI/IT firms" color="var(--accent)" />
        <StatCard icon={faNewspaper} label="Articles" value={loading ? '—' : fmt(stats?.article_count)} sub="GDELT news articles" color="var(--green)" />
        <StatCard icon={faTags} label="Themes" value={loading ? '—' : fmt(stats?.theme_count)} sub="Topic themes" color="var(--purple)" />
        <StatCard icon={faProjectDiagram} label="Relations" value={loading ? '—' : fmt(stats?.relation_count)} sub="CO_OCCURRED_WITH edges" color="var(--cyan)" />
      </div>

      {/* Quick Actions */}
      <h2 style={{ fontSize: 15, fontWeight: 600, marginBottom: 16 }}>Quick Start</h2>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 16 }}>
        {[
          {
            title: 'Ask TemporalRAG',
            desc: 'Chat with the AI using temporal Knowledge Graph context and DeepSeek-R1 reasoning.',
            color: 'var(--accent)',
            action: () => navigate('/chat'),
            cta: 'Open Chat',
          },
          {
            title: 'Explore Knowledge Graph',
            desc: 'Visualize 30 AI/IT company relations with interactive D3 graph and Legend filtering.',
            color: 'var(--green)',
            action: () => navigate('/graph'),
            cta: 'View Graph',
          },
        ].map(qa => (
          <div key={qa.title} className="card" style={{ borderColor: `${qa.color}30` }}>
            <div style={{ color: qa.color, fontWeight: 700, fontSize: 14, marginBottom: 8 }}>{qa.title}</div>
            <p style={{ color: 'var(--text-secondary)', fontSize: 13, lineHeight: 1.6, marginBottom: 16 }}>{qa.desc}</p>
            <button className="btn btn-ghost" onClick={qa.action} style={{ fontSize: 12 }}>
              {qa.cta} <FontAwesomeIcon icon={faArrowRight} size="xs" />
            </button>
          </div>
        ))}
      </div>
    </Layout>
  );
};

export default Dashboard;
