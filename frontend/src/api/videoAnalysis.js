// src/api/videoAnalysis.js
// FIXED VERSION - Matches your backend API exactly

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Step 1: Upload video file
 * Returns: { video_id, filename, status, message }
 */
export const uploadVideo = async (videoFile) => {
  const formData = new FormData();
  formData.append('file', videoFile); // Backend expects 'file', not 'video'

  try {
    const response = await fetch(`${API_BASE_URL}/api/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Upload failed: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Upload error:', error);
    throw error;
  }
};

/**
 * Step 2: Trigger analysis for uploaded video
 * Returns: { video_id, job_id, status, message }
 */
export const triggerAnalysis = async (videoId) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/analyze/${videoId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Analysis trigger failed: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Analysis trigger error:', error);
    throw error;
  }
};

/**
 * Step 3: Poll status (call this repeatedly until completed)
 * Returns: { status, progress, current_task, airflow_state }
 */
export const getAnalysisStatus = async (videoId) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/status/${videoId}`, {
      method: 'GET',
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Status check failed: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Status check error:', error);
    throw error;
  }
};

/**
 * Step 4: Get final results (only after status === 'completed')
 * Returns: { video_id, overall_score, performance_level, results: {...} }
 */
export const getResults = async (videoId) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/results/${videoId}`, {
      method: 'GET',
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      
      // 202 = Still processing
      if (response.status === 202) {
        throw new Error('STILL_PROCESSING');
      }
      
      throw new Error(errorData.detail || `Results fetch failed: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Results fetch error:', error);
    throw error;
  }
};

/**
 * CONVENIENCE FUNCTION: Complete flow from upload to results
 * This handles the entire process with status polling
 */
export const analyzeVideoComplete = async (videoFile, onProgress) => {
  try {
    // Step 1: Upload
    onProgress?.({ stage: 'uploading', progress: 0, message: 'Uploading video...' });
    const uploadResponse = await uploadVideo(videoFile);
    const { video_id } = uploadResponse;
    
    // Step 2: Trigger analysis
    onProgress?.({ stage: 'triggering', progress: 10, message: 'Starting analysis...' });
    await triggerAnalysis(video_id);
    
    // Step 3: Poll status until completed
    let status = 'processing';
    let attempts = 0;
    const maxAttempts = 120; // 6 minutes max (3 sec intervals)
    
    while (status === 'processing' && attempts < maxAttempts) {
      await new Promise(resolve => setTimeout(resolve, 3000)); // Wait 3 seconds
      
      const statusResponse = await getAnalysisStatus(video_id);
      status = statusResponse.status;
      
      onProgress?.({
        stage: 'analyzing',
        progress: statusResponse.progress || 50,
        message: statusResponse.current_task || 'Processing...',
      });
      
      if (status === 'failed') {
        throw new Error('Analysis failed. Please try again.');
      }
      
      attempts++;
    }
    
    if (status !== 'completed') {
      throw new Error('Analysis timeout. Please check results later.');
    }
    
    // Step 4: Get results
    onProgress?.({ stage: 'fetching', progress: 95, message: 'Fetching results...' });
    const results = await getResults(video_id);
    
    onProgress?.({ stage: 'done', progress: 100, message: 'Complete!' });
    return results;
    
  } catch (error) {
    console.error('Complete analysis error:', error);
    throw error;
  }
};

// Health check (optional but useful)
export const checkBackendHealth = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      signal: AbortSignal.timeout(5000), // 5 second timeout
    });
    return response.ok;
  } catch {
    return false;
  }
};