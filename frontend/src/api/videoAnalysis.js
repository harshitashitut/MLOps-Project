import { supabase } from '../lib/supabase';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const getUserId = async () => {
  const { data: { session } } = await supabase.auth.getSession();
  return session?.user?.id || null;
};

export const analyzeVideo = async (videoFile) => {
  const userId = await getUserId();
  const formData = new FormData();
  formData.append('video', videoFile);

  const headers = {};
  if (userId) {
    headers['X-User-ID'] = userId;
  }

  const response = await fetch(`${API_BASE_URL}/api/analyze-video`, {
    method: 'POST',
    headers,
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || `Upload failed: ${response.status}`);
  }

  return response.json();
};

export const pollStatus = async (videoId) => {
  const response = await fetch(`${API_BASE_URL}/api/status/${videoId}`);
  if (!response.ok) {
    throw new Error(`Status check failed: ${response.status}`);
  }
  return response.json();
};

export const getResults = async (videoId) => {
  const response = await fetch(`${API_BASE_URL}/api/results/${videoId}`);
  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || `Failed to fetch results: ${response.status}`);
  }
  return response.json();
};

export const getUserVideos = async () => {
  const userId = await getUserId();
  const headers = {};
  if (userId) {
    headers['X-User-ID'] = userId;
  }

  const response = await fetch(`${API_BASE_URL}/api/videos`, { headers });
  if (!response.ok) {
    throw new Error(`Failed to fetch videos: ${response.status}`);
  }
  return response.json();
};

export const getAdminAnalytics = async () => {
  const userId = await getUserId();
  const response = await fetch(`${API_BASE_URL}/api/admin/analytics`, {
    headers: {
      'X-User-ID': userId,
    },
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || `Failed to fetch analytics: ${response.status}`);
  }
  return response.json();
};