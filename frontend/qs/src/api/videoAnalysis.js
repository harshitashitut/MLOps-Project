const API_BASE_URL = 'http://localhost:8000';

export const analyzeVideo = async (videoFile) => {
  const formData = new FormData();
  formData.append('video', videoFile);

  try {
    const response = await fetch(`${API_BASE_URL}/api/analyze-video`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error analyzing video:', error);
    throw error;
  }
};