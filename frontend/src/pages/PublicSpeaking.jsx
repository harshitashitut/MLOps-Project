import React, { useState, useEffect, useRef } from "react";
import { Upload, Mic, ArrowLeft, CheckCircle2, AlertCircle, Loader2, TrendingUp, Target, Award } from "lucide-react";
import { analyzeVideo, pollStatus, getResults } from "../api/videoAnalysis";

export default function PublicSpeaking() {
  const [file, setFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState("");
  const [videoId, setVideoId] = useState(null);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const pollingRef = useRef(null);

  useEffect(() => {
    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
      }
    };
  }, []);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setResults(null);
      setError(null);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type.startsWith('video/')) {
      setFile(droppedFile);
      setResults(null);
      setError(null);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const startPolling = (id) => {
    pollingRef.current = setInterval(async () => {
      try {
        const status = await pollStatus(id);
        setProgress(status.progress || 0);
        setCurrentStep(status.current_step || "Processing...");

        if (status.status === "completed") {
          clearInterval(pollingRef.current);
          pollingRef.current = null;
          const resultsData = await getResults(id);
          console.log('Raw API response:', resultsData);

          // Normalize - handle flat scores and single-nested results
          const normalized = {
            overall_score: resultsData.overall_score,
            performance_level: resultsData.performance_level,
            // Reconstruct category_scores from flat values
            category_scores: {
              content: { score: resultsData.content_score },
              vocal_delivery: { score: resultsData.vocal_delivery_score },
              visual_presentation: { score: resultsData.visual_presentation_score },
              tone_emotion: { score: resultsData.tone_emotion_score },
            },
            key_metrics: resultsData.results?.key_metrics || {
              words_per_minute: resultsData.words_per_minute,
              speech_duration_seconds: resultsData.speech_duration_seconds,
              filler_words_count: resultsData.filler_words_count,
            },
            strengths: resultsData.results?.strengths || [],
            improvements: resultsData.results?.improvements || [],
            next_steps: resultsData.results?.next_steps || [],
          };
          console.log('Normalized:', normalized);
          setResults(normalized);
          setProcessing(false);
        }
      } catch (err) {
        console.error("Polling error:", err);
        setError(err.message || "Failed to get analysis status");
        setProcessing(false);
      }
    }, 2000);
  };

  const handleUpload = async () => {
    if (!file) return;

    setUploading(true);
    setError(null);
    setProgress(0);
    setCurrentStep("Uploading video...");

    try {
      const response = await analyzeVideo(file);
      setVideoId(response.video_id);
      setUploading(false);
      setProcessing(true);
      setProgress(5);
      setCurrentStep("Analysis started...");
      startPolling(response.video_id);
    } catch (err) {
      console.error("Upload error:", err);
      setError(err.message || "Failed to upload video");
      setUploading(false);
    }
  };

  const handleReset = () => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
    setFile(null);
    setResults(null);
    setError(null);
    setUploading(false);
    setProcessing(false);
    setProgress(0);
    setCurrentStep("");
    setVideoId(null);
  };

  const getScoreColor = (score) => {
    if (score >= 70) return "text-green-500";
    if (score >= 50) return "text-yellow-500";
    return "text-red-500";
  };

  const isLoading = uploading || processing;

  return (
    <div className="min-h-screen bg-black">
      <header className="bg-gradient-to-r from-emerald-600 to-green-600 shadow-lg">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center gap-4">
            <button
              onClick={() => window.history.back()}
              className="text-white/90 hover:text-white transition-colors flex items-center gap-2 group"
            >
              <ArrowLeft className="w-5 h-5 group-hover:-translate-x-1 transition-transform" />
              <span className="font-medium">Back</span>
            </button>
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-white/20 backdrop-blur-sm flex items-center justify-center">
                <Mic className="w-5 h-5 text-white" />
              </div>
              <h1 className="text-2xl font-bold text-white">Public Speaking Analysis</h1>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-12">

        {!results ? (
          <>
            <div className="mb-8 p-6 bg-gradient-to-br from-gray-900 to-black border border-green-500/20 rounded-2xl max-w-4xl mx-auto">
              <h2 className="text-xl font-bold text-white mb-2">How it works</h2>
              <p className="text-gray-400 leading-relaxed">
                Upload a video of your public speaking practice. Our AI will analyze your delivery,
                body language, vocal tone, pacing, and provide detailed feedback to help you improve.
              </p>
            </div>

            <div className="bg-gradient-to-br from-gray-900 to-black border border-green-500/20 rounded-2xl p-8 shadow-xl max-w-4xl mx-auto">
              <h3 className="text-lg font-bold text-white mb-6">Upload Your Video</h3>

              {!isLoading ? (
                <div
                  onDrop={handleDrop}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  className={`
                    relative border-2 border-dashed rounded-xl p-12 text-center transition-all duration-300
                    ${isDragging
                      ? 'border-green-500 bg-green-500/10'
                      : 'border-gray-700 hover:border-green-500/50 bg-gray-800/50'
                    }
                  `}
                >
                  <input
                    type="file"
                    accept="video/*"
                    onChange={handleFileChange}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  />

                  <div className="pointer-events-none">
                    <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gradient-to-br from-emerald-500 to-green-600 flex items-center justify-center">
                      <Upload className="w-8 h-8 text-white" />
                    </div>

                    {file ? (
                      <div className="space-y-2">
                        <p className="text-white font-medium">{file.name}</p>
                        <p className="text-sm text-gray-400">
                          {(file.size / (1024 * 1024)).toFixed(2)} MB
                        </p>
                      </div>
                    ) : (
                      <>
                        <p className="text-white font-medium mb-2">
                          Drop your video here or click to browse
                        </p>
                        <p className="text-sm text-gray-400">
                          Supports MP4, MOV, AVI (Max 500MB)
                        </p>
                      </>
                    )}
                  </div>
                </div>
              ) : (
                <div className="border-2 border-green-500/30 rounded-xl p-12 text-center bg-gray-800/50">
                  <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gradient-to-br from-emerald-500 to-green-600 flex items-center justify-center">
                    <Loader2 className="w-8 h-8 text-white animate-spin" />
                  </div>
                  <p className="text-white font-medium mb-2">{currentStep}</p>
                  <div className="max-w-md mx-auto mt-4">
                    <div className="flex justify-between text-sm text-gray-400 mb-2">
                      <span>Progress</span>
                      <span>{progress}%</span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-3">
                      <div
                        className="bg-gradient-to-r from-emerald-500 to-green-600 h-3 rounded-full transition-all duration-500"
                        style={{ width: `${progress}%` }}
                      />
                    </div>
                    <p className="text-xs text-gray-500 mt-3">
                      Analysis typically takes 1-2 minutes
                    </p>
                  </div>
                </div>
              )}

              {error && (
                <div className="mt-4 p-4 bg-red-500/10 border border-red-500/20 rounded-lg">
                  <p className="text-red-400 text-sm flex items-center gap-2">
                    <AlertCircle className="w-4 h-4" />
                    {error}
                  </p>
                </div>
              )}

              <div className="mt-6 flex gap-3">
                {file && !isLoading && (
                  <>
                    <button
                      onClick={handleUpload}
                      className="flex-1 bg-gradient-to-r from-emerald-600 to-green-600 text-white font-semibold py-3 px-6 rounded-lg hover:from-emerald-700 hover:to-green-700 transition-all duration-300 flex items-center justify-center gap-2"
                    >
                      <Upload className="w-5 h-5" />
                      Upload & Analyze
                    </button>
                    <button
                      onClick={handleReset}
                      className="px-6 py-3 border border-gray-600 text-gray-300 rounded-lg hover:border-gray-500 hover:text-white transition-colors"
                    >
                      Cancel
                    </button>
                  </>
                )}
                {isLoading && (
                  <button
                    onClick={handleReset}
                    className="px-6 py-3 border border-gray-600 text-gray-300 rounded-lg hover:border-gray-500 hover:text-white transition-colors"
                  >
                    Cancel
                  </button>
                )}
              </div>
            </div>

            <div className="mt-8 grid md:grid-cols-2 gap-4 max-w-4xl mx-auto">
              <div className="p-6 bg-gradient-to-br from-gray-900 to-black border border-green-500/20 rounded-xl">
                <h4 className="font-semibold text-white mb-2 flex items-center gap-2">
                  <CheckCircle2 className="w-5 h-5 text-green-500" />
                  Best Practices
                </h4>
                <ul className="text-sm text-gray-400 space-y-1">
                  <li>• Record in good lighting</li>
                  <li>• Face the camera directly</li>
                  <li>• Ensure clear audio quality</li>
                  <li>• Keep video under 10 minutes</li>
                </ul>
              </div>

              <div className="p-6 bg-gradient-to-br from-gray-900 to-black border border-green-500/20 rounded-xl">
                <h4 className="font-semibold text-white mb-2 flex items-center gap-2">
                  <AlertCircle className="w-5 h-5 text-yellow-500" />
                  What We Analyze
                </h4>
                <ul className="text-sm text-gray-400 space-y-1">
                  <li>• Body language & posture</li>
                  <li>• Vocal tone & clarity</li>
                  <li>• Pacing & confidence</li>
                  <li>• Eye contact & engagement</li>
                </ul>
              </div>
            </div>
          </>
        ) : (
          <div className="space-y-6">
            <div className="bg-gradient-to-r from-emerald-600 to-green-600 rounded-2xl p-8 text-center">
              <div className="flex items-center justify-center gap-2 mb-2">
                <Award className="w-6 h-6 text-white" />
                <h2 className="text-2xl font-bold text-white">Overall Performance</h2>
              </div>
              <div className="text-6xl font-bold text-white mb-2">
                {results.overall_score}/100
              </div>
              <p className="text-white/90 text-lg capitalize">
                {results.performance_level} Level
              </p>
            </div>

            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
              {results.category_scores && Object.entries(results.category_scores).map(([category, data]) => (
                <div key={category} className="bg-gradient-to-br from-gray-900 to-black border border-green-500/20 rounded-xl p-6">
                  <h3 className="text-sm font-medium text-gray-400 mb-2 capitalize">
                    {category.replace(/_/g, ' ')}
                  </h3>
                  <div className={`text-3xl font-bold mb-2 ${getScoreColor(data.score)}`}>
                    {data.score}
                  </div>
                  <div className="w-full bg-gray-800 rounded-full h-2">
                    <div
                      className="bg-gradient-to-r from-emerald-500 to-green-600 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${data.score}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>

            {results.key_metrics && (
              <div className="bg-gradient-to-br from-gray-900 to-black border border-green-500/20 rounded-xl p-6">
                <h3 className="text-xl font-bold text-white mb-4">Key Metrics</h3>
                <div className="grid md:grid-cols-3 gap-4">
                  <div>
                    <p className="text-gray-400 text-sm">Speaking Pace</p>
                    <p className="text-2xl font-bold text-white">
                      {Math.round(results.key_metrics.words_per_minute)} WPM
                    </p>
                  </div>
                  <div>
                    <p className="text-gray-400 text-sm">Duration</p>
                    <p className="text-2xl font-bold text-white">
                      {Math.round(results.key_metrics.speech_duration_seconds)}s
                    </p>
                  </div>
                  <div>
                    <p className="text-gray-400 text-sm">Filler Words</p>
                    <p className="text-2xl font-bold text-white">
                      {results.key_metrics.filler_words_count}
                    </p>
                  </div>
                </div>
              </div>
            )}

            {results.strengths && results.strengths.length > 0 && (
              <div className="bg-gradient-to-br from-gray-900 to-black border border-green-500/20 rounded-xl p-6">
                <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                  <TrendingUp className="w-6 h-6 text-green-500" />
                  Strengths
                </h3>
                <div className="space-y-3">
                  {results.strengths.map((strength, idx) => (
                    <div key={idx} className="flex gap-3 p-4 bg-green-500/10 border border-green-500/20 rounded-lg">
                      <CheckCircle2 className="w-5 h-5 text-green-500 flex-shrink-0 mt-0.5" />
                      <div>
                        <p className="text-white font-medium">{strength.strength}</p>
                        {strength.evidence && (
                          <p className="text-sm text-gray-400">{strength.evidence}</p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {results.improvements && results.improvements.length > 0 && (
              <div className="bg-gradient-to-br from-gray-900 to-black border border-green-500/20 rounded-xl p-6">
                <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                  <Target className="w-6 h-6 text-yellow-500" />
                  Areas for Improvement
                </h3>
                <div className="space-y-3">
                  {results.improvements.map((improvement, idx) => (
                    <div key={idx} className="p-4 bg-yellow-500/10 border border-yellow-500/20 rounded-lg">
                      <div className="flex items-start justify-between mb-2">
                        <p className="text-white font-medium">{improvement.issue}</p>
                        {improvement.priority && (
                          <span className="text-xs px-2 py-1 bg-yellow-500/20 text-yellow-400 rounded">
                            {improvement.priority}
                          </span>
                        )}
                      </div>
                      {improvement.recommendation && (
                        <p className="text-sm text-gray-400 mb-2">{improvement.recommendation}</p>
                      )}
                      {(improvement.current_metric || improvement.target_metric) && (
                        <div className="flex gap-4 text-xs text-gray-500">
                          {improvement.current_metric && <span>Current: {improvement.current_metric}</span>}
                          {improvement.target_metric && <span>Target: {improvement.target_metric}</span>}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {results.next_steps && results.next_steps.length > 0 && (
              <div className="bg-gradient-to-br from-gray-900 to-black border border-green-500/20 rounded-xl p-6">
                <h3 className="text-xl font-bold text-white mb-4">Next Steps</h3>
                <ul className="space-y-2">
                  {results.next_steps.map((step, idx) => (
                    <li key={idx} className="flex gap-3 text-gray-300">
                      <span className="text-green-500 font-bold">{idx + 1}.</span>
                      <span>{step}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <div className="flex gap-4">
              <button
                onClick={handleReset}
                className="flex-1 bg-gradient-to-r from-emerald-600 to-green-600 text-white font-semibold py-3 px-6 rounded-lg hover:from-emerald-700 hover:to-green-700 transition-all duration-300"
              >
                Analyze Another Video
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}