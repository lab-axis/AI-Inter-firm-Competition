import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { chatAPI } from '../services/api';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faTachometerAlt, faComments, faProjectDiagram,
  faBars, faPlus, faChevronRight
} from '@fortawesome/free-solid-svg-icons';

const NAV_ITEMS = [
  { key: '/', label: 'Dashboard', icon: faTachometerAlt },
  { key: '/chat', label: 'Chat', icon: faComments },
  { key: '/graph', label: 'Knowledge Graph', icon: faProjectDiagram },
];

const PAGE_TITLES = {
  '/': 'Dashboard',
  '/chat': 'TemporalRAG Chat',
  '/graph': 'Knowledge Graph',
};

const Layout = ({ children, title }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [recentThreads, setRecentThreads] = useState([]);
  const [mobile, setMobile] = useState(false);

  useEffect(() => {
    const check = () => {
      const m = window.innerWidth < 768;
      setMobile(m);
      if (m) setSidebarOpen(false);
    };
    check();
    window.addEventListener('resize', check);
    return () => window.removeEventListener('resize', check);
  }, []);

  useEffect(() => {
    chatAPI.getRecentThreads()
      .then(r => setRecentThreads(r.data.threads || []))
      .catch(() => setRecentThreads([]));
  }, [location.pathname]);

  const currentPath = Object.keys(PAGE_TITLES).find(k =>
    k !== '/' ? location.pathname.startsWith(k) : location.pathname === '/'
  ) || '/';
  const pageTitle = title || PAGE_TITLES[currentPath] || 'TemporalRAG';

  const sidebarWidth = sidebarOpen && !mobile ? 260 : 0;

  return (
    <div style={{ display: 'flex', minHeight: '100vh', background: 'var(--bg-base)' }}>

      {/* ── Sidebar ── */}
      <aside style={{
        width: sidebarOpen && !mobile ? 260 : 0,
        minHeight: '100vh',
        background: 'var(--sidebar-bg)',
        borderRight: '1px solid var(--sidebar-border)',
        display: 'flex', flexDirection: 'column',
        position: 'fixed', top: 0, left: 0, zIndex: 100,
        overflow: 'hidden',
        transition: 'width 0.25s ease',
      }}>
        {/* Logo */}
        <div className="sidebar-logo">
          <div className="logo-icon">T</div>
          <span className="logo-text">TemporalRAG</span>
        </div>

        {/* Nav */}
        <nav style={{ padding: '12px 0', flex: 1 }}>
          {NAV_ITEMS.map(item => (
            <div
              key={item.key}
              className={`nav-item ${currentPath === item.key ? 'active' : ''}`}
              onClick={() => {
                if (item.key === '/chat') {
                  const saved = localStorage.getItem('current_thread_uuid');
                  navigate(saved ? `/chat/${saved}` : '/chat');
                } else {
                  navigate(item.key);
                }
                if (mobile) setSidebarOpen(false);
              }}
            >
              <FontAwesomeIcon icon={item.icon} style={{ width: 14 }} />
              <span style={{ whiteSpace: 'nowrap' }}>{item.label}</span>
            </div>
          ))}

          {/* Recent Chats */}
          <div style={{
            margin: '16px 0 8px',
            padding: '0 24px',
            fontSize: 11, fontWeight: 700, letterSpacing: '0.8px',
            color: 'var(--text-muted)', textTransform: 'uppercase',
          }}>
            Recent Chats
          </div>

          <div style={{ padding: '0 8px' }}>
            {/* New Chat button */}
            <div
              className="nav-item"
              onClick={async () => {
                try {
                  const r = await chatAPI.createThread('New Chat');
                  const uuid = r.data.uuid;
                  localStorage.setItem('current_thread_uuid', uuid);
                  navigate(`/chat/${uuid}`);
                  if (mobile) setSidebarOpen(false);
                } catch {}
              }}
              style={{ color: 'var(--accent-light)', marginBottom: 4 }}
            >
              <FontAwesomeIcon icon={faPlus} style={{ width: 12 }} />
              <span style={{ fontSize: 12 }}>New Chat</span>
            </div>

            {recentThreads.map(t => (
              <div
                key={t.id}
                className="nav-item"
                style={{ fontSize: 12, padding: '7px 12px' }}
                onClick={() => {
                  localStorage.setItem('current_thread_uuid', t.uuid);
                  navigate(`/chat/${t.uuid}`);
                  if (mobile) setSidebarOpen(false);
                }}
              >
                <FontAwesomeIcon icon={faChevronRight} style={{ width: 10, opacity: 0.4 }} />
                <span style={{
                  overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                  maxWidth: 170,
                }}>
                  {t.title || 'New Chat'}
                </span>
              </div>
            ))}
          </div>
        </nav>
      </aside>

      {/* ── Main ── */}
      <div style={{ marginLeft: sidebarWidth, width: `calc(100% - ${sidebarWidth}px)`, transition: 'all 0.25s ease', display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>

        {/* Top header */}
        <header className="top-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <button
              className="sidebar-toggle"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              style={{ position: 'static' }}
            >
              <FontAwesomeIcon icon={faBars} size="sm" />
            </button>
            <span className="header-title">{pageTitle}</span>
          </div>
        </header>

        {/* Content */}
        <main className="page-content" style={{ flex: 1 }}>
          {children}
        </main>
      </div>
    </div>
  );
};

export default Layout;
