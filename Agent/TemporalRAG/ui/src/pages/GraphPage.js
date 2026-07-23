import React, { useState, useEffect, useCallback, useRef } from 'react';
import Layout from '../components/Layout';
import GraphViewer from '../components/graph/GraphViewer';
import { graphAPI } from '../services/api';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faRefresh, faFilter, faSpinner, faInfoCircle,
  faBuilding, faNewspaper, faTags, faTimes,
  faCompress, faExpand
} from '@fortawesome/free-solid-svg-icons';

const LEGEND_CONFIG = [
  { type: 'Company', label: 'Company', color: '#3b82f6', desc: 'AI/IT target firms' },
  { type: 'Article', label: 'Article', color: '#10b981', desc: 'News articles (GDELT)' },
  { type: 'Theme',   label: 'Theme',   color: '#8b5cf6', desc: 'Topic themes' },
];

const NODE_COLORS = { Company: '#3b82f6', Article: '#10b981', Theme: '#8b5cf6' };
const NODE_ICONS  = { Company: faBuilding, Article: faNewspaper, Theme: faTags };

const generateMonthOptions = () => {
  const opts = [];
  const MAX_YEAR = 2026;
  const MAX_MONTH = 6; // 2026-06 까지만
  for (let y = MAX_YEAR; y >= 2014; y--) {
    const mMax = y === MAX_YEAR ? MAX_MONTH : 12;
    for (let m = mMax; m >= 1; m--)
      opts.push(`${y}-${String(m).padStart(2, '0')}`);
  }
  return opts;
};
const MONTH_OPTIONS = generateMonthOptions();

const GraphPage = () => {
  const [graphData, setGraphData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTypes, setActiveTypes] = useState({ Company: true, Article: false, Theme: false });
  const [cutoffDate, setCutoffDate] = useState('');
  const [minWeight, setMinWeight] = useState(1);
  const [graphHeight, setGraphHeight] = useState(600);
  const containerRef = useRef(null);

  // Expand state
  const [expandedTicker, setExpandedTicker] = useState(null);
  const [expandLoading, setExpandLoading] = useState(false);
  const [baseData, setBaseData] = useState(null);

  // Selected node detail panel
  const [detailNode, setDetailNode] = useState(null);
  const [detailEdges, setDetailEdges] = useState([]);

  // Fetch base graph
  const fetchGraph = useCallback(async () => {
    setLoading(true);
    setError(null);
    setExpandedTicker(null);
    setDetailNode(null);
    try {
      const r = await graphAPI.getGraph({
        cutoff_date: cutoffDate || undefined,
        min_weight: minWeight,
        include_articles: activeTypes.Article,
        include_themes: activeTypes.Theme,
      });
      setBaseData(r.data);
      setGraphData(r.data);
    } catch (e) {
      setError(e.response?.data?.detail || e.message || 'Failed to load graph');
    } finally {
      setLoading(false);
    }
  }, [cutoffDate, minWeight, activeTypes.Article, activeTypes.Theme]);

  useEffect(() => { fetchGraph(); }, []); // eslint-disable-line

  // Height tracking
  useEffect(() => {
    const update = () => {
      if (containerRef.current) setGraphHeight(containerRef.current.clientHeight);
    };
    update();
    window.addEventListener('resize', update);
    return () => window.removeEventListener('resize', update);
  }, []);

  // Toggle legend (Company always on)
  const toggleType = (type) => {
    if (type === 'Company') return;
    setActiveTypes(prev => ({ ...prev, [type]: !prev[type] }));
  };

  // Node click → expand + show detail panel
  const handleNodeClick = async (node) => {
    if (node.type !== 'Company') return;

    // Build detail edges from current graph
    if (graphData) {
      const connected = graphData.edges.filter(e => {
        const src = e.source?.id ?? e.source;
        const tgt = e.target?.id ?? e.target;
        return src === node.id || tgt === node.id;
      });
      setDetailEdges(connected);
    }
    setDetailNode(node);

    // Collapse if already expanded
    if (expandedTicker === node.id) {
      setExpandedTicker(null);
      setGraphData(baseData);
      // Restore activeTypes
      setActiveTypes(prev => ({ ...prev, Article: false, Theme: false }));
      return;
    }

    // Expand
    setExpandLoading(true);
    setExpandedTicker(node.id);
    // Auto-enable Article & Theme in legend when expanding
    setActiveTypes(prev => ({ ...prev, Article: true, Theme: true }));

    try {
      const r = await graphAPI.expandCompany(node.id, cutoffDate || undefined);
      const expandData = r.data;

      if (baseData) {
        const nodeMap = new Map();
        [...(baseData.nodes || []), ...(expandData.nodes || [])].forEach(n => nodeMap.set(n.id, n));
        const edgeMap = new Map();
        [...(baseData.edges || []), ...(expandData.edges || [])].forEach(e => edgeMap.set(e.id, e));
        const merged = { nodes: [...nodeMap.values()], edges: [...edgeMap.values()] };
        setGraphData(merged);

        // Update detail edges
        const connected = [...edgeMap.values()].filter(e => {
          const src = e.source?.id ?? e.source;
          const tgt = e.target?.id ?? e.target;
          return src === node.id || tgt === node.id;
        });
        setDetailEdges(connected);
      } else {
        setGraphData(expandData);
      }
    } catch (e) {
      console.error('Expand failed:', e);
    } finally {
      setExpandLoading(false);
    }
  };

  // Close detail panel and collapse
  const handleCollapseAll = () => {
    setExpandedTicker(null);
    setDetailNode(null);
    setDetailEdges([]);
    setActiveTypes(prev => ({ ...prev, Article: false, Theme: false }));
    setGraphData(baseData);
  };

  return (
    <Layout title="Knowledge Graph">
      {/* ── Filter Bar ── */}
      <div className="graph-filters" style={{ marginBottom: 0, borderRadius: '12px 12px 0 0' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <FontAwesomeIcon icon={faFilter} style={{ color: 'var(--text-muted)', fontSize: 12 }} />
          <span style={{ fontSize: 12, color: 'var(--text-muted)', fontWeight: 600 }}>Filters:</span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>Cutoff:</span>
          <select
            className="input"
            style={{ width: 120, padding: '4px 8px', fontSize: 12 }}
            value={cutoffDate}
            onChange={e => setCutoffDate(e.target.value)}
          >
            <option value="">All time</option>
            {MONTH_OPTIONS.map(v => <option key={v} value={v}>{v}</option>)}
          </select>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>Min weight:</span>
          <input
            type="range" min={1} max={20} value={minWeight}
            onChange={e => setMinWeight(Number(e.target.value))}
            style={{ width: 80 }}
          />
          <span style={{ fontSize: 12, color: 'var(--text-primary)', fontFamily: 'JetBrains Mono, monospace', minWidth: 20 }}>
            {minWeight}
          </span>
        </div>

        <button className="btn btn-primary" style={{ padding: '5px 12px', fontSize: 12 }} onClick={fetchGraph}>
          <FontAwesomeIcon icon={faRefresh} size="xs" /> Apply
        </button>

        {expandedTicker && (
          <button className="btn btn-ghost" style={{ padding: '5px 12px', fontSize: 12 }} onClick={handleCollapseAll}>
            <FontAwesomeIcon icon={faCompress} size="xs" /> Collapse
          </button>
        )}

        {graphData && (
          <span style={{ fontSize: 11, color: 'var(--text-muted)', marginLeft: 'auto' }}>
            {graphData.nodes?.length ?? 0} nodes · {graphData.edges?.length ?? 0} edges
            {expandedTicker && (
              <> · Expanded: <strong style={{ color: 'var(--cyan)' }}>{expandedTicker}</strong></>
            )}
          </span>
        )}
      </div>

      {/* ── Main Area ── */}
      <div style={{ display: 'flex', height: 'calc(100vh - 56px - 48px - 56px)', minHeight: 400, borderRadius: '0 0 12px 12px', overflow: 'hidden' }}>

        {/* ── Graph Canvas ── */}
        <div
          ref={containerRef}
          className="graph-container"
          style={{ flex: 1, borderRadius: 0, border: 'none', borderRight: detailNode ? '1px solid var(--border)' : 'none' }}
        >
          {loading ? (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', gap: 16, color: 'var(--text-muted)' }}>
              <FontAwesomeIcon icon={faSpinner} spin size="2x" style={{ color: 'var(--accent)' }} />
              <span style={{ fontSize: 14 }}>Loading Knowledge Graph from Neo4j...</span>
            </div>
          ) : error ? (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', gap: 12, color: 'var(--red)', padding: 32, textAlign: 'center' }}>
              <FontAwesomeIcon icon={faInfoCircle} size="2x" />
              <div style={{ fontWeight: 600 }}>Failed to load graph</div>
              <div style={{ fontSize: 13, color: 'var(--text-muted)' }}>{error}</div>
              <button className="btn btn-ghost" onClick={fetchGraph}>
                <FontAwesomeIcon icon={faRefresh} size="xs" /> Retry
              </button>
            </div>
          ) : (
            <GraphViewer
              data={graphData}
              activeTypes={activeTypes}
              onNodeClick={handleNodeClick}
              expandedTicker={expandedTicker}
              height={graphHeight || 600}
            />
          )}

          {/* Legend */}
          <div className="graph-legend">
            <div className="legend-title">Legend</div>
            {LEGEND_CONFIG.map(item => (
              <div
                key={item.type}
                className={`legend-item ${!activeTypes[item.type] ? 'inactive' : ''}`}
                onClick={() => toggleType(item.type)}
                title={item.desc}
              >
                <div
                  className="legend-dot"
                  style={{ backgroundColor: item.color, color: item.color, opacity: activeTypes[item.type] ? 1 : 0.3 }}
                />
                <div>
                  <div style={{ fontSize: 12, fontWeight: 600, color: activeTypes[item.type] ? 'var(--text-primary)' : 'var(--text-muted)' }}>
                    {item.label}
                  </div>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', lineHeight: 1.3 }}>{item.desc}</div>
                </div>
                <div style={{
                  marginLeft: 'auto', width: 14, height: 14, borderRadius: 3,
                  border: `2px solid ${item.color}`,
                  background: activeTypes[item.type] ? item.color : 'transparent',
                  opacity: item.type === 'Company' ? 0.5 : 1,
                  transition: 'all 0.15s ease', flexShrink: 0,
                }} />
              </div>
            ))}

            <div style={{ marginTop: 12, paddingTop: 10, borderTop: '1px solid var(--border)', fontSize: 10, color: 'var(--text-muted)', lineHeight: 1.8 }}>
              <div>🖱 Scroll: zoom</div>
              <div>✋ Drag canvas: pan</div>
              <div>⬡ Drag node: move</div>
              <div style={{ color: 'var(--cyan)' }}>🔵 Click company: expand</div>
            </div>
          </div>

          {/* Expand loading */}
          {expandLoading && (
            <div style={{
              position: 'absolute', top: 16, left: '50%', transform: 'translateX(-50%)',
              background: 'rgba(10,15,30,0.95)', border: '1px solid var(--border-strong)',
              borderRadius: 8, padding: '8px 20px',
              display: 'flex', alignItems: 'center', gap: 10,
              color: 'var(--text-primary)', fontSize: 13, zIndex: 20,
            }}>
              <FontAwesomeIcon icon={faSpinner} spin style={{ color: 'var(--accent)' }} />
              Expanding {expandedTicker}…
            </div>
          )}
        </div>

        {/* ── Detail Panel ── */}
        {detailNode && (
          <div style={{
            width: 300, flexShrink: 0,
            background: 'var(--bg-surface)',
            borderLeft: '1px solid var(--border)',
            display: 'flex', flexDirection: 'column',
            overflowY: 'auto',
          }}>
            {/* Header */}
            <div style={{
              padding: '14px 16px',
              borderBottom: '1px solid var(--border)',
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              background: 'var(--bg-elevated)',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <div style={{
                  width: 28, height: 28, borderRadius: 8,
                  background: `${NODE_COLORS[detailNode.type]}25`,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}>
                  <FontAwesomeIcon icon={NODE_ICONS[detailNode.type]} style={{ color: NODE_COLORS[detailNode.type], fontSize: 13 }} />
                </div>
                <div>
                  <div style={{ fontWeight: 700, fontSize: 14, color: NODE_COLORS[detailNode.type] }}>
                    {detailNode.ticker || detailNode.id}
                  </div>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                    {detailNode.type}
                  </div>
                </div>
              </div>
              <button
                onClick={() => { setDetailNode(null); setDetailEdges([]); }}
                style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', padding: 4, borderRadius: 4 }}
              >
                <FontAwesomeIcon icon={faTimes} size="sm" />
              </button>
            </div>

            {/* Node info */}
            <div style={{ padding: '12px 16px', borderBottom: '1px solid var(--border)' }}>
              {detailNode.name && (
                <div style={{ marginBottom: 10 }}>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 3 }}>Full Name</div>
                  <div style={{ fontSize: 13, color: 'var(--text-primary)' }}>{detailNode.name}</div>
                </div>
              )}
              {detailNode.degree != null && (
                <div style={{ marginBottom: 10 }}>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 3 }}>Co-occurrence Weight</div>
                  <div style={{ fontSize: 20, fontWeight: 700, color: NODE_COLORS[detailNode.type] }}>
                    {Math.round(detailNode.degree)}
                  </div>
                </div>
              )}
              <div style={{ marginBottom: 6 }}>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 3 }}>Connected Edges</div>
                <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--text-primary)' }}>{detailEdges.length}</div>
              </div>
            </div>

            {/* Expand / Collapse button */}
            {detailNode.type === 'Company' && (
              <div style={{ padding: '10px 16px', borderBottom: '1px solid var(--border)' }}>
                {expandedTicker === detailNode.id ? (
                  <button
                    className="btn btn-ghost"
                    style={{ width: '100%', justifyContent: 'center', fontSize: 12 }}
                    onClick={handleCollapseAll}
                  >
                    <FontAwesomeIcon icon={faCompress} size="xs" />
                    Collapse Expansion
                  </button>
                ) : (
                  <button
                    className="btn btn-primary"
                    style={{ width: '100%', justifyContent: 'center', fontSize: 12 }}
                    onClick={() => handleNodeClick(detailNode)}
                    disabled={expandLoading}
                  >
                    {expandLoading
                      ? <><FontAwesomeIcon icon={faSpinner} spin size="xs" /> Expanding…</>
                      : <><FontAwesomeIcon icon={faExpand} size="xs" /> Expand — Show Articles & Themes</>
                    }
                  </button>
                )}
              </div>
            )}

            {/* Connected nodes list */}
            {detailEdges.length > 0 && (
              <div style={{ padding: '12px 16px', flex: 1 }}>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 8 }}>
                  Connected Nodes ({detailEdges.length})
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                  {detailEdges.slice(0, 30).map((e, i) => {
                    const src = e.source?.id ?? e.source;
                    const tgt = e.target?.id ?? e.target;
                    const peerId = src === detailNode.id ? tgt : src;
                    const peer = graphData?.nodes?.find(n => n.id === peerId);
                    return (
                      <div key={i} style={{
                        display: 'flex', alignItems: 'center', gap: 8,
                        padding: '6px 10px', borderRadius: 6,
                        background: 'var(--bg-elevated)',
                        fontSize: 12,
                      }}>
                        <div style={{
                          width: 8, height: 8, borderRadius: '50%', flexShrink: 0,
                          background: NODE_COLORS[peer?.type] || '#666',
                          boxShadow: `0 0 4px ${NODE_COLORS[peer?.type] || '#666'}`,
                        }} />
                        <span style={{ flex: 1, color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          {peer?.ticker || peer?.name || peerId}
                        </span>
                        {e.weight && (
                          <span style={{ color: 'var(--text-muted)', fontFamily: 'JetBrains Mono, monospace', fontSize: 10 }}>
                            {Math.round(e.weight)}
                          </span>
                        )}
                      </div>
                    );
                  })}
                  {detailEdges.length > 30 && (
                    <div style={{ textAlign: 'center', fontSize: 11, color: 'var(--text-muted)', padding: '4px 0' }}>
                      +{detailEdges.length - 30} more…
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </Layout>
  );
};

export default GraphPage;
