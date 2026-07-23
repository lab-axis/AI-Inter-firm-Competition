import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8002';

const api = axios.create({
  baseURL: API_URL,
  headers: { 'Content-Type': 'application/json' },
});

// ── Chat API ──────────────────────────────────────────────────
export const chatAPI = {
  // Thread CRUD
  getThreads: () => api.get('/chats/threads'),
  getRecentThreads: () => api.get('/chats/recent'),
  createThread: (title) => api.post('/chats/threads', { title }),
  getThread: (uuid) => api.get(`/chats/threads/${uuid}`),
  updateThread: (uuid, title) => api.put(`/chats/threads/${uuid}`, { title }),
  deleteThread: (uuid) => api.delete(`/chats/threads/${uuid}`),

  // Chat message
  sendMessage: (threadUuid, { message, cutoff_date, firms }) =>
    api.post(`/chats/chat/${threadUuid}`, { message, cutoff_date, firms }),
};

// ── Graph API ─────────────────────────────────────────────────
export const graphAPI = {
  getStats: () => api.get('/graph/stats'),
  getGraph: (params = {}) => api.get('/graph/', { params }),
  expandCompany: (ticker, cutoff_date) =>
    api.get(`/graph/expand/${ticker}`, { params: { cutoff_date } }),
};

export default api;
