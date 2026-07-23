import React, { createContext, useState, useContext, useEffect } from 'react';
import { authAPI } from '../services/api';

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [currentUser, setCurrentUser] = useState(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  const isTokenValid = (token) => {
    if (!token) return false;
    try {
      // Simple JWT expiry check without jwt-decode v3 dependency issues
      const payload = JSON.parse(atob(token.split('.')[1]));
      return payload.exp * 1000 > Date.now();
    } catch {
      return false;
    }
  };

  const loadUser = async () => {
    setIsLoading(true);
    const token = localStorage.getItem('token');
    if (token && isTokenValid(token)) {
      try {
        const res = await authAPI.getCurrentUser();
        setCurrentUser(res.data);
        setIsAuthenticated(true);
      } catch {
        localStorage.removeItem('token');
        setCurrentUser(null);
        setIsAuthenticated(false);
      }
    } else {
      localStorage.removeItem('token');
      setCurrentUser(null);
      setIsAuthenticated(false);
    }
    setIsLoading(false);
  };

  const login = async (username, password) => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await authAPI.login(username, password);
      localStorage.setItem('token', res.data.access_token);
      await loadUser();
      return true;
    } catch (err) {
      setError(err.response?.data?.detail || 'Login failed. Please try again.');
      setIsLoading(false);
      return false;
    }
  };

  const logout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('current_thread_uuid');
    setCurrentUser(null);
    setIsAuthenticated(false);
  };

  useEffect(() => { loadUser(); }, []); // eslint-disable-line

  return (
    <AuthContext.Provider value={{ currentUser, isAuthenticated, isLoading, error, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
};
