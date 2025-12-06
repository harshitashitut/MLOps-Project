# Frontend Implementation Guide (Next.js)

**Owner:** Person 2 (Frontend + Database Lead)  
**Timeline:** Day 1-3  
**Tech Stack:** Next.js 14, React, TypeScript, Tailwind CSS, Recharts

---

## Overview

You're building the user interface that allows users to:
- Upload pitch videos with drag-and-drop
- Monitor analysis progress in real-time
- View detailed results with visualizations
- Access monitoring dashboard for system health

---

## Project Structure

```
frontend/
├── app/
│   ├── layout.tsx              # Root layout
│   ├── page.tsx                # Homepage (upload form)
│   ├── results/
│   │   └── [id]/
│   │       └── page.tsx        # Results dashboard
│   └── report/
│       └── page.tsx            # Monitoring dashboard
│
├── components/
│   ├── VideoUpload.tsx         # Drag-and-drop upload
│   ├── StatusPoller.tsx        # Progress tracker
│   ├── ScoreCard.tsx           # Score display component
│   ├── FeedbackSection.tsx     # Feedback display
│   └── MonitoringCharts.tsx    # Drift/health charts
│
├── lib/
│   ├── api.ts                  # API client
│   ├── types.ts                # TypeScript interfaces
│   └── utils.ts                # Helper functions
│
├── public/
│   └── logo.svg
│
├── tailwind.config.ts
├── package.json
└── Dockerfile
```

---

## Day 1: Setup + Upload Page (4-6 hours)

### Step 1: Project Setup (30 min)

```bash
npx create-next-app@latest frontend --typescript --tailwind --app
cd frontend
npm install axios recharts lucide-react
```

### Step 2: API Client (30 min)

**File: `lib/api.ts`**

```typescript
import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: `${API_BASE_URL}/api`,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface VideoUploadResponse {
  video_id: string;
  filename: string;
  status: string;
  message: string;
  uploaded_at: string;
}

export interface AnalysisResponse {
  job_id: string;
  video_id: string;
  status: string;
  message: string;
  estimated_time_seconds: number;
}

export interface StatusResponse {
  status: 'uploaded' | 'processing' | 'completed' | 'failed';
  progress: number;
  current_task?: string;
  estimated_time_remaining?: number;
}

export interface ResultsResponse {
  video_id: string;
  overall_score: number;
  results: {
    content_analysis: any;
    delivery_analysis: any;
    visual_analysis: any;
  };
  feedback: {
    strengths: string[];
    improvements: string[];
    detailed_feedback: string;
  };
  metadata: any;
  processing_time_seconds: number;
  created_at: string;
}

export interface MonitoringResponse {
  total_analyses: number;
  avg_score: number;
  system_health: 'healthy' | 'warning' | 'critical';
  drift_detected: boolean;
  score_history: Array<{ overall_score: number; created_at: string }>;
  last_updated: string;
}

// API Methods
export const api = {
  uploadVideo: async (file: File): Promise<VideoUploadResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await apiClient.post('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return response.data;
  },

  triggerAnalysis: async (videoId: string): Promise<AnalysisResponse> => {
    const response = await apiClient.post(`/analyze/${videoId}`);
    return response.data;
  },

  getStatus: async (videoId: string): Promise<StatusResponse> => {
    const response = await apiClient.get(`/status/${videoId}`);
    return response.data;
  },

  getResults: async (videoId: string): Promise<ResultsResponse> => {
    const response = await apiClient.get(`/results/${videoId}`);
    return response.data;
  },

  getMonitoring: async (): Promise<MonitoringResponse> => {
    const response = await apiClient.get('/monitoring');
    return response.data;
  },
};
```

### Step 3: TypeScript Types (15 min)

**File: `lib/types.ts`**

```typescript
export type AnalysisStatus = 'uploaded' | 'processing' | 'completed' | 'failed';

export interface Video {
  video_id: string;
  filename: string;
  status: AnalysisStatus;
  uploaded_at: string;
}

export interface Analysis {
  content_analysis: {
    problem_clarity: number;
    solution_fit: number;
    market_sizing: number;
    business_model: number;
    competitive_advantage: number;
  };
  delivery_analysis: {
    speaking_pace: string;
    vocal_confidence: number;
    filler_words_count: number;
    emotional_tone: string;
    articulation_score: number;
  };
  visual_analysis: {
    body_language_score: number;
    posture_quality: string;
    eye_contact: string;
    hand_gestures: string;
    facial_confidence: number;
  };
}

export interface Feedback {
  strengths: string[];
  improvements: string[];
  detailed_feedback: string;
  next_steps?: string[];
}
```

### Step 4: Root Layout (30 min)

**File: `app/layout.tsx`**

```typescript
import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'PitchQuest - AI Pitch Analysis',
  description: 'Get AI-powered feedback on your startup pitch',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <nav className="bg-indigo-600 text-white shadow-lg">
          <div className="max-w-7xl mx-auto px-4 py-4 flex justify-between items-center">
            <div className="flex items-center space-x-2">
              <span className="text-2xl font-bold">🎯 PitchQuest</span>
            </div>
            <div className="flex space-x-6">
              <a href="/" className="hover:underline">Upload</a>
              <a href="/report" className="hover:underline">Monitoring</a>
            </div>
          </div>
        </nav>
        <main>{children}</main>
        <footer className="bg-gray-100 py-6 mt-12">
          <div className="max-w-7xl mx-auto px-4 text-center text-gray-600">
            <p>© 2025 PitchQuest. AI-Powered Pitch Analysis.</p>
          </div>
        </footer>
      </body>
    </html>
  );
}
```

### Step 5: Video Upload Component (1 hour)

**File: `components/VideoUpload.tsx`**

```typescript
'use client';

import React, { useState, useCallback } from 'react';
import { Upload, Video, AlertCircle } from 'lucide-react';

interface VideoUploadProps {
  onUploadComplete: (videoId: string, filename: string) => void;
}

export default function VideoUpload({ onUploadComplete }: VideoUploadProps) {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type.startsWith('video/')) {
      setFile(droppedFile);
      setError(null);
    } else {
      setError('Please upload a valid video file (MP4, MOV, AVI)');
    }
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    
    setUploading(true);
    setError(null);
    
    try {
      const { api } = await import('@/lib/api');
      
      // Upload video
      const uploadResponse = await api.uploadVideo(file);
      
      // Trigger analysis
      await api.triggerAnalysis(uploadResponse.video_id);
      
      // Notify parent component
      onUploadComplete(uploadResponse.video_id, uploadResponse.filename);
    } catch (err: any) {
      console.error('Upload error:', err);
      setError(err.response?.data?.detail || 'Upload failed. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      {/* Drag & Drop Area */}
      <div
        className={`relative border-2 border-dashed rounded-lg p-12 text-center transition-colors ${
          dragActive
            ? 'border-indigo-500 bg-indigo-50'
            : 'border-gray-300 hover:border-gray-400'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          type="file"
          accept="video/*"
          onChange={handleFileChange}
          className="hidden"
          id="video-upload"
          disabled={uploading}
        />
        
        <label
          htmlFor="video-upload"
          className="cursor-pointer flex flex-col items-center"
        >
          {file ? (
            <>
              <Video className="w-16 h-16 text-indigo-600 mb-4" />
              <p className="text-lg font-medium text-gray-900">{file.name}</p>
              <p className="text-sm text-gray-500">
                {(file.size / 1024 / 1024).toFixed(2)} MB
              </p>
            </>
          ) : (
            <>
              <Upload className="w-16 h-16 text-gray-400 mb-4" />
              <p className="text-lg font-medium text-gray-900 mb-2">
                Drag and drop your pitch video
              </p>
              <p className="text-sm text-gray-500 mb-4">
                or click to browse (MP4, MOV, AVI - Max 100MB)
              </p>
              <span className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors">
                Select Video
              </span>
            </>
          )}
        </label>
      </div>

      {/* Error Message */}
      {error && (
        <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start">
          <AlertCircle className="w-5 h-5 text-red-600 mr-3 flex-shrink-0 mt-0.5" />
          <p className="text-red-800">{error}</p>
        </div>
      )}

      {/* Upload Button */}
      {file && (
        <button
          onClick={handleUpload}
          disabled={uploading}
          className={`mt-6 w-full py-3 px-6 rounded-lg font-medium text-white transition-colors ${
            uploading
              ? 'bg-gray-400 cursor-not-allowed'
              : 'bg-indigo-600 hover:bg-indigo-700'
          }`}
        >
          {uploading ? (
            <span className="flex items-center justify-center">
              <svg className="animate-spin h-5 w-5 mr-3" viewBox="0 0 24 24">
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                  fill="none"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </svg>
              Uploading & Analyzing...
            </span>
          ) : (
            'Upload & Analyze Pitch'
          )}
        </button>
      )}
    </div>
  );
}
```

### Step 6: Homepage (30 min)

**File: `app/page.tsx`**

```typescript
'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import VideoUpload from '@/components/VideoUpload';

export default function HomePage() {
  const router = useRouter();

  const handleUploadComplete = (videoId: string, filename: string) => {
    // Redirect to results page
    router.push(`/results/${videoId}`);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
      <div className="max-w-5xl mx-auto px-4 py-16">
        {/* Hero Section */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-gray-900 mb-4">
            Perfect Your Pitch with AI
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Get instant, comprehensive feedback on your startup pitch.
            Our AI analyzes your content, delivery, and visual presence.
          </p>
        </div>

        {/* Features */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          <div className="bg-white p-6 rounded-lg shadow-md">
            <div className="text-3xl mb-3">📝</div>
            <h3 className="text-lg font-semibold mb-2">Content Analysis</h3>
            <p className="text-gray-600 text-sm">
              Problem clarity, solution fit, market sizing, and business model evaluation
            </p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <div className="text-3xl mb-3">🎤</div>
            <h3 className="text-lg font-semibold mb-2">Delivery Feedback</h3>
            <p className="text-gray-600 text-sm">
              Speaking pace, vocal confidence, emotional tone, and articulation analysis
            </p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <div className="text-3xl mb-3">👁️</div>
            <h3 className="text-lg font-semibold mb-2">Visual Presence</h3>
            <p className="text-gray-600 text-sm">
              Body language, posture, eye contact, and facial confidence assessment
            </p>
          </div>
        </div>

        {/* Upload Component */}
        <div className="bg-white p-8 rounded-xl shadow-xl">
          <VideoUpload onUploadComplete={handleUploadComplete} />
        </div>

        {/* How It Works */}
        <div className="mt-16 text-center">
          <h2 className="text-2xl font-bold text-gray-900 mb-8">How It Works</h2>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div>
              <div className="w-12 h-12 bg-indigo-600 text-white rounded-full flex items-center justify-center mx-auto mb-3 font-bold">
                1
              </div>
              <p className="text-sm text-gray-600">Upload your pitch video</p>
            </div>
            <div>
              <div className="w-12 h-12 bg-indigo-600 text-white rounded-full flex items-center justify-center mx-auto mb-3 font-bold">
                2
              </div>
              <p className="text-sm text-gray-600">AI analyzes content, delivery & visuals</p>
            </div>
            <div>
              <div className="w-12 h-12 bg-indigo-600 text-white rounded-full flex items-center justify-center mx-auto mb-3 font-bold">
                3
              </div>
              <p className="text-sm text-gray-600">Get detailed scores and feedback</p>
            </div>
            <div>
              <div className="w-12 h-12 bg-indigo-600 text-white rounded-full flex items-center justify-center mx-auto mb-3 font-bold">
                4
              </div>
              <p className="text-sm text-gray-600">Improve and iterate on your pitch</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
```

### End of Day 1 Deliverable
- [ ] Frontend runs on :3000
- [ ] Upload page looks professional
- [ ] Drag-and-drop works
- [ ] Can upload video (test with backend)
- [ ] Redirects to results page

---

## Day 2: Results Dashboard (6-8 hours)

### Step 1: Status Poller Component (1 hour)

**File: `components/StatusPoller.tsx`**

```typescript
'use client';

import { useState, useEffect } from 'react';
import { api, StatusResponse } from '@/lib/api';
import { Loader2, CheckCircle, XCircle } from 'lucide-react';

interface StatusPollerProps {
  videoId: string;
  onComplete: () => void;
  onError: () => void;
}

export default function StatusPoller({ videoId, onComplete, onError }: StatusPollerProps) {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let interval: NodeJS.Timeout;

    const pollStatus = async () => {
      try {
        const statusData = await api.getStatus(videoId);
        setStatus(statusData);

        if (statusData.status === 'completed') {
          clearInterval(interval);
          onComplete();
        } else if (statusData.status === 'failed') {
          clearInterval(interval);
          onError();
        }
      } catch (err: any) {
        console.error('Status polling error:', err);
        setError('Failed to check status');
      }
    };

    // Poll immediately
    pollStatus();

    // Then poll every 3 seconds
    interval = setInterval(pollStatus, 3000);

    return () => clearInterval(interval);
  }, [videoId, onComplete, onError]);

  if (error) {
    return (
      <div className="text-center py-12">
        <XCircle className="w-16 h-16 text-red-600 mx-auto mb-4" />
        <p className="text-xl text-red-600">{error}</p>
      </div>
    );
  }

  if (!status) {
    return (
      <div className="text-center py-12">
        <Loader2 className="w-16 h-16 text-indigo-600 mx-auto mb-4 animate-spin" />
        <p className="text-xl text-gray-600">Initializing...</p>
      </div>
    );
  }

  return (
    <div className="text-center py-12">
      {status.status === 'processing' ? (
        <>
          <Loader2 className="w-16 h-16 text-indigo-600 mx-auto mb-4 animate-spin" />
          <h2 className="text-2xl font-bold text-gray-900 mb-2">
            Analyzing Your Pitch
          </h2>
          <p className="text-gray-600 mb-6">{status.current_task || 'Processing...'}</p>
          
          {/* Progress Bar */}
          <div className="max-w-md mx-auto">
            <div className="w-full bg-gray-200 rounded-full h-4 mb-2">
              <div
                className="bg-indigo-600 h-4 rounded-full transition-all duration-500"
                style={{ width: `${status.progress}%` }}
              />
            </div>
            <p className="text-sm text-gray-500">
              {status.progress}% complete
              {status.estimated_time_remaining && 
                ` • ~${Math.ceil(status.estimated_time_remaining / 60)} min remaining`
              }
            </p>
          </div>

          {/* Task List */}
          <div className="mt-8 max-w-md mx-auto text-left">
            <div className="space-y-2">
              {['Video Processing', 'Visual Analysis', 'Audio Analysis', 'Content Evaluation'].map((task, idx) => {
                const taskProgress = (status.progress / 100) * 4;
                const isComplete = taskProgress > idx;
                const isCurrent = Math.floor(taskProgress) === idx;

                return (
                  <div key={task} className="flex items-center">
                    {isComplete ? (
                      <CheckCircle className="w-5 h-5 text-green-600 mr-3" />
                    ) : isCurrent ? (
                      <Loader2 className="w-5 h-5 text-indigo-600 mr-3 animate-spin" />
                    ) : (
                      <div className="w-5 h-5 border-2 border-gray-300 rounded-full mr-3" />
                    )}
                    <span className={isComplete ? 'text-green-600' : 'text-gray-600'}>
                      {task}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </>
      ) : (
        <>
          <CheckCircle className="w-16 h-16 text-green-600 mx-auto mb-4" />
          <p className="text-xl text-gray-600">Analysis Complete!</p>
        </>
      )}
    </div>
  );
}
```

### Step 2: Score Card Component (30 min)

**File: `components/ScoreCard.tsx`**

```typescript
interface ScoreCardProps {
  title: string;
  score: number;
  details: Record<string, any>;
  icon?: string;
}

export default function ScoreCard({ title, score, details, icon = '📊' }: ScoreCardProps) {
  const getScoreColor = (score: number) => {
    if (score >= 8) return 'text-green-600';
    if (score >= 6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getScoreBg = (score: number) => {
    if (score >= 8) return 'bg-green-50';
    if (score >= 6) return 'bg-yellow-50';
    return 'bg-red-50';
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
        <span className="text-2xl">{icon}</span>
      </div>
      
      <div className={`${getScoreBg(score)} rounded-lg p-4 mb-4`}>
        <div className={`text-4xl font-bold ${getScoreColor(score)}`}>
          {score.toFixed(1)}
        </div>
        <p className="text-sm text-gray-600">out of 10</p>
      </div>

      <div className="space-y-2">
        {Object.entries(details).map(([key, value]) => (
          <div key={key} className="flex justify-between text-sm">
            <span className="text-gray-600 capitalize">
              {key.replace(/_/g, ' ')}:
            </span>
            <span className="font-medium text-gray-900">
              {typeof value === 'number' ? value.toFixed(1) : value}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
```

### Step 3: Results Page (2 hours)

**File: `app/results/[id]/page.tsx`**

```typescript
'use client';

import { useState, useEffect } from 'react';
import { useParams } from 'next/navigation';
import { api, ResultsResponse } from '@/lib/api';
import StatusPoller from '@/components/StatusPoller';
import ScoreCard from '@/components/ScoreCard';
import { TrendingUp, TrendingDown, AlertCircle } from 'lucide-react';

export default function ResultsPage() {
  const params = useParams();
  const videoId = params.id as string;
  
  const [loading, setLoading] = useState(true);
  const [results, setResults] = useState<ResultsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchResults = async () => {
    try {
      const data = await api.getResults(videoId);
      setResults(data);
      setLoading(false);
    } catch (err: any) {
      console.error('Failed to fetch results:', err);
      if (err.response?.status === 202) {
        // Still processing, keep waiting
        return;
      }
      setError('Failed to load results');
      setLoading(false);
    }
  };

  const handleAnalysisComplete = () => {
    fetchResults();
  };

  const handleAnalysisError = () => {
    setError('Analysis failed. Please try again.');
    setLoading(false);
  };

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="bg-red-50 p-8 rounded-lg max-w-md">
          <AlertCircle className="w-16 h-16 text-red-600 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-red-900 mb-2 text-center">
            Analysis Failed
          </h2>
          <p className="text-red-700 text-center mb-4">{error}</p>
          <button
            onClick={() => window.location.href = '/'}
            className="w-full py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  if (loading || !results) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="bg-white p-8 rounded-xl shadow-xl max-w-2xl w-full">
          <StatusPoller
            videoId={videoId}
            onComplete={handleAnalysisComplete}
            onError={handleAnalysisError}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4">
        {/* Overall Score */}
        <div className="bg-gradient-to-r from-indigo-600 to-purple-600 rounded-xl shadow-xl p-8 mb-8 text-white">
          <h1 className="text-3xl font-bold mb-2">Analysis Complete!</h1>
          <div className="flex items-center justify-between">
            <div>
              <div className="text-6xl font-bold mb-2">{results.overall_score}</div>
              <p className="text-xl opacity-90">Overall Score</p>
            </div>
            <div className="text-right">
              <p className="text-sm opacity-75">Processing Time</p>
              <p className="text-2xl font-semibold">
                {results.processing_time_seconds.toFixed(1)}s
              </p>
            </div>
          </div>
        </div>

        {/* Score Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <ScoreCard
            title="Content Quality"
            score={
              Object.values(results.results.content_analysis).reduce((a: number, b: any) => a + Number(b), 0) /
              Object.keys(results.results.content_analysis).length
            }
            details={results.results.content_analysis}
            icon="📝"
          />
          <ScoreCard
            title="Delivery"
            score={results.results.delivery_analysis.vocal_confidence}
            details={results.results.delivery_analysis}
            icon="🎤"
          />
          <ScoreCard
            title="Visual Presence"
            score={results.results.visual_analysis.body_language_score}
            details={results.results.visual_analysis}
            icon="👁️"
          />
        </div>

        {/* Feedback Section */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          {/* Strengths */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex items-center mb-4">
              <TrendingUp className="w-6 h-6 text-green-600 mr-2" />
              <h2 className="text-xl font-bold text-gray-900">Strengths</h2>
            </div>
            <ul className="space-y-2">
              {results.feedback.strengths.map((strength, idx) => (
                <li key={idx} className="flex items-start">
                  <span className="text-green-600 mr-2">✓</span>
                  <span className="text-gray-700">{strength}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Areas to Improve */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex items-center mb-4">
              <TrendingDown className="w-6 h-6 text-orange-600 mr-2" />
              <h2 className="text-xl font-bold text-gray-900">Areas to Improve</h2>
            </div>
            <ul className="space-y-2">
              {results.feedback.improvements.map((improvement, idx) => (
                <li key={idx} className="flex items-start">
                  <span className="text-orange-600 mr-2">→</span>
                  <span className="text-gray-700">{improvement}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Detailed Feedback */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-bold text-gray-900 mb-4">Detailed Feedback</h2>
          <p className="text-gray-700 leading-relaxed whitespace-pre-wrap">
            {results.feedback.detailed_feedback}
          </p>
        </div>

        {/* Actions */}
        <div className="mt-8 flex justify-center space-x-4">
          <button
            onClick={() => window.location.href = '/'}
            className="px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 font-medium"
          >
            Analyze Another Pitch
          </button>
          <button
            onClick={() => window.print()}
            className="px-6 py-3 bg-gray-200 text-gray-800 rounded-lg hover:bg-gray-300 font-medium"
          >
            Download Report
          </button>
        </div>
      </div>
    </div>
  );
}
```

### End of Day 2 Deliverable
- [ ] Results page shows live progress
- [ ] Scores displayed with color coding
- [ ] Feedback sections populated
- [ ] Responsive design works on mobile
- [ ] Can analyze multiple videos

---

## Day 3: Monitoring Dashboard + Polish (4-6 hours)

### Step 1: Monitoring Charts Component (1 hour)

**File: `components/MonitoringCharts.tsx`**

```typescript
'use client';

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface MonitoringChartsProps {
  scoreHistory: Array<{ overall_score: number; created_at: string }>;
}

export default function MonitoringCharts({ scoreHistory }: MonitoringChartsProps) {
  const chartData = scoreHistory.map(item => ({
    date: new Date(item.created_at).toLocaleDateString(),
    score: item.overall_score
  }));

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-xl font-bold text-gray-900 mb-4">Score Trend</h2>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis domain={[0, 100]} />
          <Tooltip />
          <Line type="monotone" dataKey="score" stroke="#4f46e5" strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
```

### Step 2: Monitoring Page (1.5 hours)

**File: `app/report/page.tsx`**

```typescript
'use client';

import { useState, useEffect } from 'react';
import { api, MonitoringResponse } from '@/lib/api';
import MonitoringCharts from '@/components/MonitoringCharts';
import { Activity, AlertTriangle, CheckCircle, BarChart3 } from 'lucide-react';

export default function MonitoringPage() {
  const [data, setData] = useState<MonitoringResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const monitoringData = await api.getMonitoring();
        setData(monitoringData);
      } catch (err) {
        console.error('Failed to fetch monitoring data:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 30000); // Refresh every 30s

    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-indigo-600" />
      </div>
    );
  }

  if (!data) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <p className="text-xl text-gray-600">No monitoring data available</p>
      </div>
    );
  }

  const getHealthColor = (health: string) => {
    switch (health) {
      case 'healthy': return 'text-green-600';
      case 'warning': return 'text-yellow-600';
      case 'critical': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  const getHealthBg = (health: string) => {
    switch (health) {
      case 'healthy': return 'bg-green-50';
      case 'warning': return 'bg-yellow-50';
      case 'critical': return 'bg-red-50';
      default: return 'bg-gray-50';
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4">
        <h1 className="text-3xl font-bold text-gray-900 mb-8">System Monitoring</h1>

        {/* Metric Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex items-center justify-between mb-2">
              <BarChart3 className="w-8 h-8 text-indigo-600" />
            </div>
            <div className="text-3xl font-bold text-gray-900">
              {data.total_analyses}
            </div>
            <p className="text-sm text-gray-600">Total Analyses</p>
          </div>

          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex items-center justify-between mb-2">
              <Activity className="w-8 h-8 text-blue-600" />
            </div>
            <div className="text-3xl font-bold text-gray-900">
              {data.avg_score.toFixed(1)}
            </div>
            <p className="text-sm text-gray-600">Average Score</p>
          </div>

          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex items-center justify-between mb-2">
              {data.system_health === 'healthy' ? (
                <CheckCircle className="w-8 h-8 text-green-600" />
              ) : (
                <AlertTriangle className="w-8 h-8 text-yellow-600" />
              )}
            </div>
            <div className={`text-2xl font-bold ${getHealthColor(data.system_health)} capitalize`}>
              {data.system_health}
            </div>
            <p className="text-sm text-gray-600">System Health</p>
          </div>

          <div className={`rounded-lg shadow-lg p-6 ${getHealthBg(data.drift_detected ? 'warning' : 'healthy')}`}>
            <div className="flex items-center justify-between mb-2">
              <AlertTriangle className={`w-8 h-8 ${data.drift_detected ? 'text-yellow-600' : 'text-green-600'}`} />
            </div>
            <div className={`text-2xl font-bold ${data.drift_detected ? 'text-yellow-600' : 'text-green-600'}`}>
              {data.drift_detected ? 'Detected' : 'None'}
            </div>
            <p className="text-sm text-gray-600">Data Drift</p>
          </div>
        </div>

        {/* Drift Warning */}
        {data.drift_detected && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-8">
            <div className="flex items-start">
              <AlertTriangle className="w-5 h-5 text-yellow-600 mr-3 flex-shrink-0 mt-0.5" />
              <div>
                <h3 className="font-semibold text-yellow-900 mb-1">Data Drift Detected</h3>
                <p className="text-yellow-800 text-sm">
                  Recent analysis scores are significantly different from historical patterns.
                  Consider retraining the model or investigating input data quality.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Score History Chart */}
        {data.score_history.length > 0 && (
          <MonitoringCharts scoreHistory={data.score_history} />
        )}

        {/* System Info */}
        <div className="mt-8 bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-bold text-gray-900 mb-4">System Information</h2>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-gray-600">Last Updated:</span>
              <span className="ml-2 font-medium text-gray-900">
                {new Date(data.last_updated).toLocaleString()}
              </span>
            </div>
            <div>
              <span className="text-gray-600">Status:</span>
              <span className="ml-2 font-medium text-green-600">Operational</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
```

### Step 3: UI Polish (2 hours)

- Add transitions and animations
- Improve responsive design
- Test on mobile devices
- Add error boundaries
- Optimize images and fonts

**File: `tailwind.config.ts`**

```typescript
import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      animation: {
        'spin-slow': 'spin 3s linear infinite',
      },
    },
  },
  plugins: [],
}
export default config
```

### Step 4: Dockerfile (30 min)

**File: `Dockerfile`**

```dockerfile
FROM node:18-alpine AS base

# Install dependencies only when needed
FROM base AS deps
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

# Rebuild the source code only when needed
FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Set environment variable for production build
ENV NEXT_PUBLIC_API_URL=http://backend:8000

RUN npm run build

# Production image, copy all the files and run next
FROM base AS runner
WORKDIR /app

ENV NODE_ENV production

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

ENV PORT 3000
ENV HOSTNAME "0.0.0.0"

CMD ["node", "server.js"]
```

### End of Day 3 Deliverable
- [ ] Monitoring dashboard complete
- [ ] All pages responsive
- [ ] Professional UI with animations
- [ ] Error states handled
- [ ] Docker image builds

---

## Testing Checklist

- [ ] Upload various video formats (MP4, MOV, AVI)
- [ ] Upload large files (>50MB)
- [ ] Test on Chrome, Firefox, Safari
- [ ] Test on mobile devices
- [ ] Test slow network conditions
- [ ] Test with backend offline (graceful errors)
- [ ] Test concurrent uploads
- [ ] Verify all charts render correctly

---

## Quick Reference

### Start Frontend
```bash
cd frontend
npm run dev
```

### Build for Production
```bash
npm run build
npm start
```

### Common Issues

**Issue:** API calls fail with CORS error  
**Fix:** Ensure backend has `http://localhost:3000` in CORS origins

**Issue:** Charts not rendering  
**Fix:** Check recharts is installed: `npm install recharts`

**Issue:** Environment variables not working  
**Fix:** Prefix with `NEXT_PUBLIC_` for client-side vars

---

## Success Criteria

- [ ] Professional, modern UI design
- [ ] Responsive on all devices
- [ ] Real-time status updates
- [ ] Clear data visualization
- [ ] Graceful error handling
- [ ] Fast page loads (<3s)
- [ ] Accessible (keyboard navigation, ARIA labels)

**Ready for integration testing! 🎨**