import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const Login = () => {
  const { login, error, isLoading } = useAuth();
  const navigate = useNavigate();
  const [form, setForm] = useState({ username: '', password: '' });

  const handleSubmit = async (e) => {
    e.preventDefault();
    const ok = await login(form.username, form.password);
    if (ok) navigate('/');
  };

  return (
    <div className="login-page">
      <div className="login-card">
        {/* Logo */}
        <div style={{ textAlign: 'center', marginBottom: 32 }}>
          <div style={{
            width: 56, height: 56, borderRadius: 16,
            background: 'linear-gradient(135deg, var(--accent), var(--cyan))',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            margin: '0 auto 16px', fontSize: 24, fontWeight: 800, color: 'white',
            boxShadow: '0 0 30px rgba(59,130,246,0.4)',
          }}>T</div>
          <h1 style={{ fontSize: 24, fontWeight: 700, margin: '0 0 6px' }}>
            TemporalRAG
          </h1>
          <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>
            Temporal Knowledge Graph RAG System
          </p>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit}>
          <div style={{ marginBottom: 16 }}>
            <label style={{ display: 'block', marginBottom: 6, fontSize: 12, color: 'var(--text-secondary)', fontWeight: 500 }}>
              Username
            </label>
            <input
              className="input"
              style={{ width: '100%' }}
              type="text"
              placeholder="Enter username"
              value={form.username}
              onChange={e => setForm({ ...form, username: e.target.value })}
              required
            />
          </div>

          <div style={{ marginBottom: 24 }}>
            <label style={{ display: 'block', marginBottom: 6, fontSize: 12, color: 'var(--text-secondary)', fontWeight: 500 }}>
              Password
            </label>
            <input
              className="input"
              style={{ width: '100%' }}
              type="password"
              placeholder="Enter password"
              value={form.password}
              onChange={e => setForm({ ...form, password: e.target.value })}
              required
            />
          </div>

          {error && (
            <div style={{
              background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)',
              borderRadius: 8, padding: '10px 14px', marginBottom: 16,
              color: 'var(--red)', fontSize: 13,
            }}>
              {error}
            </div>
          )}

          <button
            className="btn btn-primary"
            type="submit"
            disabled={isLoading}
            style={{ width: '100%', justifyContent: 'center', padding: '10px', fontSize: 14 }}
          >
            {isLoading ? (
              <><div className="spinner" style={{ width: 16, height: 16 }} /> Signing in...</>
            ) : 'Sign In'}
          </button>
        </form>

        {/* Default credentials hint */}
        <div style={{
          marginTop: 24, padding: 14,
          background: 'rgba(59,130,246,0.06)', borderRadius: 8,
          border: '1px solid rgba(59,130,246,0.15)',
        }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 6, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
            Default Accounts
          </div>
          <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.8 }}>
            admin / Admin@123<br />
            user1 / Pass@123
          </div>
        </div>
      </div>
    </div>
  );
};

export default Login;
