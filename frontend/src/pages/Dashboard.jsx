import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { ArrowLeft, Video, Clock, TrendingUp, AlertCircle, Loader2, Play } from "lucide-react";
import { getUserVideos, getResults } from "../api/videoAnalysis";

export default function Dashboard() {
    const [videos, setVideos] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [selectedVideo, setSelectedVideo] = useState(null);
    const [selectedResults, setSelectedResults] = useState(null);
    const [loadingResults, setLoadingResults] = useState(false);

    useEffect(() => {
        fetchVideos();
    }, []);

    const fetchVideos = async () => {
        try {
            setLoading(true);
            const data = await getUserVideos();
            setVideos(data.videos || []);
        } catch (err) {
            console.error("Failed to fetch videos:", err);
            setError("Failed to load your videos. Please try again.");
        } finally {
            setLoading(false);
        }
    };

    const handleVideoClick = async (video) => {
        if (video.status !== "completed") return;

        setSelectedVideo(video);
        setLoadingResults(true);

        try {
            const resultsData = await getResults(video.video_id);
            const normalized = {
                overall_score: resultsData.overall_score,
                performance_level: resultsData.performance_level,
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
            };
            setSelectedResults(normalized);
        } catch (err) {
            console.error("Failed to fetch results:", err);
        } finally {
            setLoadingResults(false);
        }
    };

    const closeModal = () => {
        setSelectedVideo(null);
        setSelectedResults(null);
    };

    const getScoreColor = (score) => {
        if (score >= 70) return "text-green-500";
        if (score >= 50) return "text-yellow-500";
        return "text-red-500";
    };

    const getScoreBg = (score) => {
        if (score >= 70) return "bg-green-500";
        if (score >= 50) return "bg-yellow-500";
        return "bg-red-500";
    };

    const getStatusBadge = (status) => {
        switch (status) {
            case "completed":
                return <span className="px-2 py-1 text-xs rounded bg-green-500/20 text-green-400">Completed</span>;
            case "processing":
                return <span className="px-2 py-1 text-xs rounded bg-blue-500/20 text-blue-400">Processing</span>;
            case "failed":
                return <span className="px-2 py-1 text-xs rounded bg-red-500/20 text-red-400">Failed</span>;
            default:
                return <span className="px-2 py-1 text-xs rounded bg-gray-500/20 text-gray-400">{status}</span>;
        }
    };

    const formatDate = (dateString) => {
        if (!dateString) return "—";
        const date = new Date(dateString);
        return date.toLocaleDateString("en-US", {
            month: "short",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit",
        });
    };

    const completedVideos = videos.filter(v => v.status === "completed");
    const averageScore = completedVideos.length > 0
        ? Math.round(completedVideos.reduce((sum, v) => sum + (v.overall_score || 0), 0) / completedVideos.length)
        : 0;

    return (
        <div className="min-h-screen bg-black">
            <header className="bg-gradient-to-r from-emerald-600 to-green-600 shadow-lg">
                <div className="max-w-7xl mx-auto px-6 py-6">
                    <div className="flex items-center gap-4">
                        <Link
                            to="/"
                            className="text-white/90 hover:text-white transition-colors flex items-center gap-2 group"
                        >
                            <ArrowLeft className="w-5 h-5 group-hover:-translate-x-1 transition-transform" />
                            <span className="font-medium">Back</span>
                        </Link>
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-xl bg-white/20 backdrop-blur-sm flex items-center justify-center">
                                <TrendingUp className="w-5 h-5 text-white" />
                            </div>
                            <h1 className="text-2xl font-bold text-white">Your Dashboard</h1>
                        </div>
                    </div>
                </div>
            </header>

            <main className="max-w-7xl mx-auto px-6 py-12">
                {/* Stats Summary */}
                <div className="grid md:grid-cols-3 gap-4 mb-8">
                    <div className="bg-gradient-to-br from-gray-900 to-black border border-green-500/20 rounded-xl p-6">
                        <p className="text-gray-400 text-sm mb-1">Total Videos</p>
                        <p className="text-3xl font-bold text-white">{videos.length}</p>
                    </div>
                    <div className="bg-gradient-to-br from-gray-900 to-black border border-green-500/20 rounded-xl p-6">
                        <p className="text-gray-400 text-sm mb-1">Completed</p>
                        <p className="text-3xl font-bold text-green-500">{completedVideos.length}</p>
                    </div>
                    <div className="bg-gradient-to-br from-gray-900 to-black border border-green-500/20 rounded-xl p-6">
                        <p className="text-gray-400 text-sm mb-1">Average Score</p>
                        <p className={`text-3xl font-bold ${getScoreColor(averageScore)}`}>{averageScore}/100</p>
                    </div>
                </div>

                {/* Videos List */}
                <div className="bg-gradient-to-br from-gray-900 to-black border border-green-500/20 rounded-xl p-6">
                    <h2 className="text-xl font-bold text-white mb-6">Your Videos</h2>

                    {loading ? (
                        <div className="flex items-center justify-center py-12">
                            <Loader2 className="w-8 h-8 text-green-500 animate-spin" />
                        </div>
                    ) : error ? (
                        <div className="flex items-center justify-center py-12 text-red-400">
                            <AlertCircle className="w-5 h-5 mr-2" />
                            {error}
                        </div>
                    ) : videos.length === 0 ? (
                        <div className="text-center py-12">
                            <Video className="w-12 h-12 text-gray-600 mx-auto mb-4" />
                            <p className="text-gray-400 mb-4">No videos yet</p>
                            <Link
                                to="/public-speaking"
                                className="inline-block bg-gradient-to-r from-emerald-600 to-green-600 text-white font-semibold py-2 px-6 rounded-lg hover:from-emerald-700 hover:to-green-700 transition-all"
                            >
                                Analyze Your First Video
                            </Link>
                        </div>
                    ) : (
                        <div className="space-y-3">
                            {videos.map((video) => (
                                <div
                                    key={video.video_id}
                                    onClick={() => handleVideoClick(video)}
                                    className={`flex items-center justify-between p-4 rounded-lg border transition-all ${video.status === "completed"
                                            ? "border-gray-700 hover:border-green-500/50 cursor-pointer hover:bg-gray-800/50"
                                            : "border-gray-800 opacity-60"
                                        }`}
                                >
                                    <div className="flex items-center gap-4">
                                        <div className="w-10 h-10 rounded-lg bg-gray-800 flex items-center justify-center">
                                            {video.status === "completed" ? (
                                                <Play className="w-5 h-5 text-green-500" />
                                            ) : (
                                                <Video className="w-5 h-5 text-gray-500" />
                                            )}
                                        </div>
                                        <div>
                                            <p className="text-white font-medium">{video.filename}</p>
                                            <div className="flex items-center gap-3 text-sm text-gray-400">
                                                <span className="flex items-center gap-1">
                                                    <Clock className="w-3 h-3" />
                                                    {formatDate(video.uploaded_at)}
                                                </span>
                                                {getStatusBadge(video.status)}
                                            </div>
                                        </div>
                                    </div>

                                    {video.status === "completed" && video.overall_score !== undefined && (
                                        <div className="text-right">
                                            <p className={`text-2xl font-bold ${getScoreColor(video.overall_score)}`}>
                                                {video.overall_score}
                                            </p>
                                            <p className="text-xs text-gray-400">Overall Score</p>
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                {/* Link to analyze more */}
                <div className="mt-6 text-center">
                    <Link
                        to="/public-speaking"
                        className="inline-block bg-gradient-to-r from-emerald-600 to-green-600 text-white font-semibold py-3 px-8 rounded-lg hover:from-emerald-700 hover:to-green-700 transition-all"
                    >
                        Analyze New Video
                    </Link>
                </div>
            </main>

            {/* Results Modal */}
            {selectedVideo && (
                <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4" onClick={closeModal}>
                    <div
                        className="bg-gray-900 border border-green-500/20 rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto"
                        onClick={(e) => e.stopPropagation()}
                    >
                        {loadingResults ? (
                            <div className="flex items-center justify-center py-20">
                                <Loader2 className="w-8 h-8 text-green-500 animate-spin" />
                            </div>
                        ) : selectedResults ? (
                            <div className="p-6 space-y-6">
                                <div className="flex items-center justify-between">
                                    <h3 className="text-xl font-bold text-white">{selectedVideo.filename}</h3>
                                    <button onClick={closeModal} className="text-gray-400 hover:text-white text-2xl">&times;</button>
                                </div>

                                {/* Score */}
                                <div className="bg-gradient-to-r from-emerald-600 to-green-600 rounded-xl p-6 text-center">
                                    <p className="text-white/80 mb-1">Overall Performance</p>
                                    <p className="text-5xl font-bold text-white">{selectedResults.overall_score}/100</p>
                                    <p className="text-white/80 capitalize">{selectedResults.performance_level} Level</p>
                                </div>

                                {/* Category Scores */}
                                <div className="grid grid-cols-2 gap-3">
                                    {Object.entries(selectedResults.category_scores).map(([cat, data]) => (
                                        <div key={cat} className="bg-gray-800 rounded-lg p-4">
                                            <p className="text-gray-400 text-sm capitalize mb-1">{cat.replace(/_/g, " ")}</p>
                                            <p className={`text-2xl font-bold ${getScoreColor(data.score)}`}>{data.score}</p>
                                            <div className="w-full bg-gray-700 rounded-full h-1.5 mt-2">
                                                <div className={`${getScoreBg(data.score)} h-1.5 rounded-full`} style={{ width: `${data.score}%` }} />
                                            </div>
                                        </div>
                                    ))}
                                </div>

                                {/* Strengths */}
                                {selectedResults.strengths?.length > 0 && (
                                    <div>
                                        <h4 className="text-white font-semibold mb-2">Strengths</h4>
                                        <ul className="space-y-2">
                                            {selectedResults.strengths.slice(0, 3).map((s, i) => (
                                                <li key={i} className="text-gray-300 text-sm flex gap-2">
                                                    <span className="text-green-500">✓</span>
                                                    {s.strength}
                                                </li>
                                            ))}
                                        </ul>
                                    </div>
                                )}

                                {/* Improvements */}
                                {selectedResults.improvements?.length > 0 && (
                                    <div>
                                        <h4 className="text-white font-semibold mb-2">Areas to Improve</h4>
                                        <ul className="space-y-2">
                                            {selectedResults.improvements.slice(0, 3).map((imp, i) => (
                                                <li key={i} className="text-gray-300 text-sm flex gap-2">
                                                    <span className="text-yellow-500">→</span>
                                                    {imp.issue}
                                                </li>
                                            ))}
                                        </ul>
                                    </div>
                                )}

                                <button
                                    onClick={closeModal}
                                    className="w-full bg-gray-800 text-white py-3 rounded-lg hover:bg-gray-700 transition"
                                >
                                    Close
                                </button>
                            </div>
                        ) : null}
                    </div>
                </div>
            )}
        </div>
    );
}