import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import Layout from '../components/Layout';
import { chatAPI } from '../services/api';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faPaperPlane, faCalendar, faBuilding, faChevronDown,
  faChevronUp, faBrain, faRoute, faCircleNotch
} from '@fortawesome/free-solid-svg-icons';

// 30 target firms for the multi-select
const TARGET_FIRMS = [
  'MSFT','GOOGL','AMZN','META','AAPL','NVDA','AMD','AVGO','QCOM','INTC',
  'ADI','TSM','ASML','AMAT','LRCX','KLAC','MU','TXN','ADBE','CRM',
  'ORCL','IBM','SAP','NOW','INTU','PANW','ADSK','CSCO','STX','TSLA'
];

// Generate month options from 2014-01 to 2026-06
const generateMonthOptions = () => {
  const opts = [];
  const MAX_YEAR = 2026;
  const MAX_MONTH = 6; // 2026-06 까지만
  for (let y = MAX_YEAR; y >= 2014; y--) {
    const mMax = y === MAX_YEAR ? MAX_MONTH : 12;
    for (let m = mMax; m >= 1; m--) {
      const val = `${y}-${String(m).padStart(2, '0')}`;
      opts.push({ value: val, label: val });
    }
  }
  return opts;
};
const MONTH_OPTIONS = generateMonthOptions();

// ── CoT Accordion ──────────────────────────────────────────────
const CoTAccordion = ({ text }) => {
  const [open, setOpen] = useState(false);
  if (!text) return null;
  return (
    <div className="cot-accordion" style={{ marginTop: 12 }}>
      <div className="cot-header" onClick={() => setOpen(!open)}>
        <span><FontAwesomeIcon icon={faBrain} style={{ marginRight: 6 }} />Chain-of-Thought Reasoning</span>
        <FontAwesomeIcon icon={open ? faChevronUp : faChevronDown} size="xs" />
      </div>
      {open && <div className="cot-body">{text}</div>}
    </div>
  );
};

// ── Path Cards ─────────────────────────────────────────────────
const PathCards = ({ paths }) => {
  if (!paths || paths.length === 0) return null;
  return (
    <div style={{ marginTop: 12 }}>
      <div style={{ fontSize: 11, color: 'var(--cyan)', fontWeight: 700, letterSpacing: '0.5px', textTransform: 'uppercase', marginBottom: 8 }}>
        <FontAwesomeIcon icon={faRoute} style={{ marginRight: 5 }} />
        Knowledge Paths ({paths.length})
      </div>
      {paths.map((p, i) => (
        <div key={i} className="path-card">
          <div className="path-flow">
            {p.path.map((segment, j) => (
              <React.Fragment key={j}>
                {j > 0 && j % 2 === 0 && <span style={{ color: 'var(--text-muted)' }}>→</span>}
                {j % 2 === 0
                  ? <span style={{ background: 'rgba(59,130,246,0.2)', padding: '2px 7px', borderRadius: 4 }}>{segment}</span>
                  : <span className="path-edge">({segment})</span>
                }
              </React.Fragment>
            ))}
            {p.weight !== null && p.weight !== undefined && (
              <span className="badge badge-cyan" style={{ marginLeft: 4 }}>w={p.weight}</span>
            )}
          </div>
          {p.snippets && p.snippets.length > 0 && (
            <div style={{ marginTop: 6, paddingTop: 6, borderTop: '1px solid rgba(255,255,255,0.05)' }}>
              {p.snippets.map((s, si) => (
                <div key={si} style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 3 }}>
                  └ {s.slice(0, 120)}{s.length > 120 ? '…' : ''}
                </div>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  );
};

// ── Meta Chips ────────────────────────────────────────────────
const MetaChips = ({ mode, cutoff_date, entities_found }) => (
  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 10 }}>
    {mode && (
      <span className={`badge ${mode === 'pathrag' ? 'badge-blue' : mode === 'error' ? 'badge-red' : 'badge-yellow'}`}>
        {mode === 'pathrag' ? 'PathRAG' : mode === 'vector_fallback' ? 'Vector Fallback' : mode}
      </span>
    )}
    {cutoff_date && (
      <span className="badge badge-cyan">
        <FontAwesomeIcon icon={faCalendar} size="xs" /> cutoff: {cutoff_date}
      </span>
    )}
    {entities_found && entities_found.length > 0 && entities_found.map(e => (
      <span key={e} className="badge badge-purple">{e}</span>
    ))}
  </div>
);

// ── Single Message ────────────────────────────────────────────
const ChatMessage = ({ msg }) => {
  const isUser = msg.role === 'user';
  let meta = null;
  if (!isUser && msg.meta_json) {
    try { meta = JSON.parse(msg.meta_json); } catch {}
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: isUser ? 'flex-end' : 'flex-start' }}>
      <div className={`chat-bubble ${msg.role}`}>
        {!isUser && meta && (
          <MetaChips mode={meta.mode} cutoff_date={meta.cutoff_date} entities_found={meta.entities_found} />
        )}
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.message}</ReactMarkdown>
        {!isUser && meta && <PathCards paths={meta.paths} />}
        {!isUser && meta?.reasoning_process && <CoTAccordion text={meta.reasoning_process} />}
      </div>
      <span style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4, paddingLeft: isUser ? 0 : 4, paddingRight: isUser ? 4 : 0 }}>
        {msg.created_at ? new Date(msg.created_at).toLocaleTimeString() : ''}
      </span>
    </div>
  );
};

// ── Main ChatPage ─────────────────────────────────────────────
const ChatPage = () => {
  const { threadId } = useParams();
  const navigate = useNavigate();

  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [threadLoading, setThreadLoading] = useState(false);
  const [currentThread, setCurrentThread] = useState(null);

  // TemporalRAG parameters
  const [cutoffDate, setCutoffDate] = useState('2026-06');
  const [selectedFirms, setSelectedFirms] = useState([]);
  const [showFirmSelect, setShowFirmSelect] = useState(false);
  const [paramsPanelOpen, setParamsPanelOpen] = useState(true);

  const bottomRef = useRef(null);
  const inputRef = useRef(null);

  // Load thread
  useEffect(() => {
    if (!threadId) return;
    setThreadLoading(true);
    chatAPI.getThread(threadId)
      .then(r => {
        setCurrentThread(r.data);
        setMessages(r.data.chats || []);
        if (r.data.last_cutoff_date) setCutoffDate(r.data.last_cutoff_date);
        if (r.data.last_firms) {
          try { setSelectedFirms(JSON.parse(r.data.last_firms) || []); } catch {}
        }
      })
      .catch(() => {})
      .finally(() => setThreadLoading(false));
  }, [threadId]);

  // Auto-scroll
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Send message
  const sendMessage = useCallback(async () => {
    if (!input.trim() || loading) return;
    let tid = threadId;

    // Create thread if none
    if (!tid) {
      try {
        const r = await chatAPI.createThread('New Chat');
        tid = r.data.uuid;
        localStorage.setItem('current_thread_uuid', tid);
        navigate(`/chat/${tid}`, { replace: true });
      } catch { return; }
    }

    const userMsg = {
      id: Date.now(), role: 'user', message: input.trim(),
      created_at: new Date().toISOString(),
    };
    setMessages(prev => [...prev, userMsg]);
    const query = input.trim();
    setInput('');
    setLoading(true);

    try {
      const r = await chatAPI.sendMessage(tid, {
        message: query,
        cutoff_date: cutoffDate,
        firms: selectedFirms.length > 0 ? selectedFirms : null,
      });

      const assistantMsg = {
        id: r.data.assistant_message_id,
        role: 'assistant',
        message: r.data.answer,
        created_at: r.data.created_at,
        meta_json: JSON.stringify({
          mode: r.data.mode,
          cutoff_date: r.data.cutoff_date,
          cutoff_ts: r.data.cutoff_ts,
          entities_found: r.data.entities_found || [],
          paths: r.data.paths || [],
          reasoning_process: r.data.reasoning_process,
        }),
      };
      setMessages(prev => [...prev, assistantMsg]);
    } catch (err) {
      setMessages(prev => [...prev, {
        id: Date.now(), role: 'assistant',
        message: `Error: ${err.response?.data?.detail || err.message}`,
        created_at: new Date().toISOString(),
      }]);
    } finally {
      setLoading(false);
    }
  }, [input, loading, threadId, navigate, cutoffDate, selectedFirms]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  };

  const toggleFirm = (ticker) => {
    setSelectedFirms(prev => prev.includes(ticker) ? prev.filter(f => f !== ticker) : [...prev, ticker]);
  };

  return (
    <Layout title={currentThread?.title || 'Chat'}>
      <div style={{ display: 'flex', height: 'calc(100vh - 56px - 48px)', gap: 0, maxHeight: 'calc(100vh - 104px)' }}>

        {/* ── Parameters Panel ── */}
        <div style={{
          width: paramsPanelOpen ? 240 : 0,
          overflow: 'hidden',
          transition: 'width 0.2s ease',
          borderRight: '1px solid var(--border)',
          background: 'var(--bg-surface)',
          display: 'flex', flexDirection: 'column',
          flexShrink: 0,
        }}>
          <div style={{ padding: '16px', overflow: 'auto', flex: 1 }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.8px', marginBottom: 16 }}>
              RAG Parameters
            </div>

            {/* Cutoff Date */}
            <div style={{ marginBottom: 16 }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--text-secondary)', fontWeight: 600, marginBottom: 6 }}>
                <FontAwesomeIcon icon={faCalendar} size="xs" /> Cutoff Date
              </label>
              <select
                className="input"
                style={{ width: '100%', cursor: 'pointer' }}
                value={cutoffDate}
                onChange={e => setCutoffDate(e.target.value)}
              >
                {MONTH_OPTIONS.map(opt => (
                  <option key={opt.value} value={opt.value}>{opt.label}</option>
                ))}
              </select>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>
                Only data before this date will be used
              </div>
            </div>

            {/* Firm Filter */}
            <div style={{ marginBottom: 16 }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--text-secondary)', fontWeight: 600 }}>
                  <FontAwesomeIcon icon={faBuilding} size="xs" /> Firms
                </label>
                <button
                  className="btn btn-ghost"
                  style={{ padding: '2px 8px', fontSize: 10 }}
                  onClick={() => setSelectedFirms([])}
                >
                  Clear
                </button>
              </div>

              <button
                className="btn btn-ghost"
                style={{ width: '100%', justifyContent: 'space-between', marginBottom: 8, fontSize: 12 }}
                onClick={() => setShowFirmSelect(!showFirmSelect)}
              >
                <span>{selectedFirms.length === 0 ? 'All firms' : `${selectedFirms.length} selected`}</span>
                <FontAwesomeIcon icon={showFirmSelect ? faChevronUp : faChevronDown} size="xs" />
              </button>

              {showFirmSelect && (
                <div style={{
                  background: 'var(--bg-elevated)', border: '1px solid var(--border)',
                  borderRadius: 8, padding: 8, maxHeight: 200, overflowY: 'auto',
                  display: 'flex', flexWrap: 'wrap', gap: 4,
                }}>
                  {TARGET_FIRMS.map(ticker => (
                    <button
                      key={ticker}
                      onClick={() => toggleFirm(ticker)}
                      style={{
                        padding: '3px 8px', borderRadius: 4, border: 'none',
                        fontSize: 11, fontFamily: 'JetBrains Mono, monospace',
                        cursor: 'pointer', transition: 'all 0.15s ease',
                        background: selectedFirms.includes(ticker) ? 'var(--accent)' : 'var(--bg-overlay)',
                        color: selectedFirms.includes(ticker) ? 'white' : 'var(--text-secondary)',
                        fontWeight: selectedFirms.includes(ticker) ? 700 : 400,
                      }}
                    >
                      {ticker}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Collapse toggle */}
          <button
            className="btn btn-ghost"
            style={{ margin: 8, fontSize: 11, justifyContent: 'center' }}
            onClick={() => setParamsPanelOpen(false)}
          >
            Hide Panel
          </button>
        </div>

        {/* ── Chat Area ── */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', background: 'var(--bg-base)' }}>
          {/* Show panel toggle */}
          {!paramsPanelOpen && (
            <div style={{ padding: '8px 16px', borderBottom: '1px solid var(--border)', background: 'var(--bg-surface)' }}>
              <button className="btn btn-ghost" style={{ fontSize: 11 }} onClick={() => setParamsPanelOpen(true)}>
                <FontAwesomeIcon icon={faCalendar} size="xs" /> {cutoffDate}
                {selectedFirms.length > 0 && <> · {selectedFirms.length} firms</>}
              </button>
            </div>
          )}

          {/* Messages */}
          <div className="chat-messages" style={{ flex: 1, overflowY: 'auto', padding: '20px 24px', display: 'flex', flexDirection: 'column', gap: 16 }}>
            {threadLoading ? (
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--text-muted)', fontSize: 13 }}>
                <div className="spinner" style={{ width: 16, height: 16 }} /> Loading thread...
              </div>
            ) : messages.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '60px 20px', color: 'var(--text-muted)' }}>
                <div style={{ fontSize: 40, marginBottom: 16 }}>🧠</div>
                <div style={{ fontSize: 18, fontWeight: 600, marginBottom: 8, color: 'var(--text-primary)' }}>
                  TemporalRAG Chat
                </div>
                <p style={{ fontSize: 13, lineHeight: 1.7, maxWidth: 400, margin: '0 auto' }}>
                  Ask about AI/IT company relationships, stock movements, or request
                  explanations for B-MTGNN predictions using the knowledge graph.
                </p>
                <div style={{ marginTop: 20, display: 'flex', flexDirection: 'column', gap: 8, alignItems: 'center' }}>
                  {[
                    'Why did NVIDIA outperform AMD in late 2024?',
                    'Explain the relationship between TSMC and Apple before 2025-01',
                    'What factors connected Microsoft and OpenAI in 2024?',
                  ].map(q => (
                    <button
                      key={q}
                      className="btn btn-ghost"
                      style={{ fontSize: 12, textAlign: 'left', maxWidth: 400 }}
                      onClick={() => setInput(q)}
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              messages.map((msg, i) => <ChatMessage key={msg.id || i} msg={msg} />)
            )}

            {loading && (
              <div className="chat-bubble assistant" style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <FontAwesomeIcon icon={faCircleNotch} spin style={{ color: 'var(--accent)' }} />
                <span style={{ color: 'var(--text-muted)', fontSize: 13 }}>
                  TemporalRAG reasoning... (cutoff: {cutoffDate})
                </span>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          {/* Input Bar */}
          <div className="chat-input-bar">
            <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end' }}>
              <textarea
                ref={inputRef}
                className="input"
                style={{ flex: 1, resize: 'none', minHeight: 44, maxHeight: 120, lineHeight: 1.5, paddingTop: 10 }}
                rows={1}
                placeholder={`Ask about company relationships (cutoff: ${cutoffDate})...`}
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                onInput={e => {
                  e.target.style.height = 'auto';
                  e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px';
                }}
              />
              <button
                className="btn btn-primary"
                style={{ height: 44, width: 44, padding: 0, justifyContent: 'center', flexShrink: 0 }}
                onClick={sendMessage}
                disabled={loading || !input.trim()}
              >
                {loading
                  ? <div className="spinner" style={{ width: 16, height: 16 }} />
                  : <FontAwesomeIcon icon={faPaperPlane} />
                }
              </button>
            </div>
            <div style={{ marginTop: 6, fontSize: 10, color: 'var(--text-muted)', display: 'flex', gap: 12 }}>
              <span>Enter to send · Shift+Enter for newline</span>
              <span style={{ marginLeft: 'auto' }}>
                Cutoff: <strong style={{ color: 'var(--cyan)' }}>{cutoffDate}</strong>
                {selectedFirms.length > 0 && <> · Firms: <strong style={{ color: 'var(--accent-light)' }}>{selectedFirms.join(', ')}</strong></>}
              </span>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default ChatPage;
