import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { ArrowLeft, Users, Video, TrendingUp, Award, Loader2, AlertCircle } from "lucide-react";
import { getAdminAnalytics } from "../api/videoAnalysis";

export default function Admin() {
    const [analytics, setAnalytics] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        fetchAnalytics();
    }, []);

    const fetchAnalytics = async () => {
        try {
            setLoading(true);
            const data = await getAdminAnalytics();
            setAnalytics(data);
        } catch (err) {
            console.error("Failed to fetch analytics:", err);
            setError(err.message || "Failed to load analytics");
        } finally {
            setLoading(false);
        }
    };

    const getScoreColor = (score) => {
        if (score >= 70) return "text-green-500";
        if (score >= 50) return "text-yellow-500";
        return "text-red-500";
    };

    if (loading) {
        return (
            <div className="min-h-screen bg-black flex items-center justify-center">
                <Loader2 className="w-8 h-8 text-green-500 animate-spin" />
            </div>
        );
    }

    if (error) {
        return (
            <div className="min-h-screen bg-black flex items-center justify-center">
                <div className="text-center">
                    <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
                    <p className="text-red-400">{error}</p>
                    <Link to="/" className="text-green-500 hover:underline mt-4 block">
                        Go back home
                    </Link>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-black">
            <header className="bg-gradient-to-r from-purple-600 to-indigo-600 shadow-lg">
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
                                <Award className="w-5 h-5 text-white" />
                            </div>
                            <h1 className="text-2xl font-bold text-white">Admin Dashboard</h1>
                        </div>
                    </div>
                </div>
            </header>

            <main className="max-w-7xl mx-auto px-6 py-12">
                {/* Overview Stats */}
                <div className="grid md:grid-cols-4 gap-4 mb-8">
                    <div className="bg-gradient-to-br from-gray-900 to-black border border-purple-500/20 rounded-xl p-6">
                        <div className="flex items-center gap-3 mb-2">
                            <Users className="w-5 h-5 text-purple-400" />
                            <p className="text-gray-400 text-sm">Total Users</p>
                        </div>
                        <p className="text-3xl font-bold text-white">{analytics?.total_users || 0}</p>
                    </div>
                    <div className="bg-gradient-to-br from-gray-900 to-black border border-purple-500/20 rounded-xl p-6">
                        <div className="flex items-center gap-3 mb-2">
                            <Video className="w-5 h-5 text-purple-400" />
                            <p className="text-gray-400 text-sm">Total Videos</p>
                        </div>
                        <p className="text-3xl font-bold text-white">{analytics?.total_videos || 0}</p>
                    </div>
                    <div className="bg-gradient-to-br from-gray-900 to-black border border-purple-500/20 rounded-xl p-6">
                        <div className="flex items-center gap-3 mb-2">
                            <TrendingUp className="w-5 h-5 text-purple-400" />
                            <p className="text-gray-400 text-sm">Completed</p>
                        </div>
                        <p className="text-3xl font-bold text-green-500">{analytics?.completed_videos || 0}</p>
                    </div>
                    <div className="bg-gradient-to-br from-gray-900 to-black border border-purple-500/20 rounded-xl p-6">
                        <div className="flex items-center gap-3 mb-2">
                            <Award className="w-5 h-5 text-purple-400" />
                            <p className="text-gray-400 text-sm">Avg Score</p>
                        </div>
                        <p className={`text-3xl font-bold ${getScoreColor(analytics?.average_score || 0)}`}>
                            {Math.round(analytics?.average_score || 0)}/100
                        </p>
                    </div>
                </div>

                {/* Score Distribution */}
                <div className="grid md:grid-cols-2 gap-6 mb-8">
                    <div className="bg-gradient-to-br from-gray-900 to-black border border-purple-500/20 rounded-xl p-6">
                        <h3 className="text-xl font-bold text-white mb-4">Score Distribution</h3>
                        <div className="space-y-3">
                            {analytics?.score_distribution?.map((item, idx) => (
                                <div key={idx} className="flex items-center gap-3">
                                    <span className="text-gray-400 text-sm w-20">{item.range}</span>
                                    <div className="flex-1 bg-gray-800 rounded-full h-4">
                                        <div
                                            className="bg-gradient-to-r from-purple-500 to-indigo-500 h-4 rounded-full"
                                            style={{ width: `${(item.count / (analytics?.completed_videos || 1)) * 100}%` }}
                                        />
                                    </div>
                                    <span className="text-white font-medium w-8">{item.count}</span>
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className="bg-gradient-to-br from-gray-900 to-black border border-purple-500/20 rounded-xl p-6">
                        <h3 className="text-xl font-bold text-white mb-4">Category Averages</h3>
                        <div className="space-y-4">
                            <div>
                                <div className="flex justify-between text-sm mb-1">
                                    <span className="text-gray-400">Content</span>
                                    <span className={getScoreColor(analytics?.avg_content || 0)}>
                                        {Math.round(analytics?.avg_content || 0)}
                                    </span>
                                </div>
                                <div className="w-full bg-gray-800 rounded-full h-2">
                                    <div className="bg-green-500 h-2 rounded-full" style={{ width: `${analytics?.avg_content || 0}%` }} />
                                </div>
                            </div>
                            <div>
                                <div className="flex justify-between text-sm mb-1">
                                    <span className="text-gray-400">Vocal Delivery</span>
                                    <span className={getScoreColor(analytics?.avg_vocal || 0)}>
                                        {Math.round(analytics?.avg_vocal || 0)}
                                    </span>
                                </div>
                                <div className="w-full bg-gray-800 rounded-full h-2">
                                    <div className="bg-green-500 h-2 rounded-full" style={{ width: `${analytics?.avg_vocal || 0}%` }} />
                                </div>
                            </div>
                            <div>
                                <div className="flex justify-between text-sm mb-1">
                                    <span className="text-gray-400">Visual Presentation</span>
                                    <span className={getScoreColor(analytics?.avg_visual || 0)}>
                                        {Math.round(analytics?.avg_visual || 0)}
                                    </span>
                                </div>
                                <div className="w-full bg-gray-800 rounded-full h-2">
                                    <div className="bg-green-500 h-2 rounded-full" style={{ width: `${analytics?.avg_visual || 0}%` }} />
                                </div>
                            </div>
                            <div>
                                <div className="flex justify-between text-sm mb-1">
                                    <span className="text-gray-400">Tone & Emotion</span>
                                    <span className={getScoreColor(analytics?.avg_emotion || 0)}>
                                        {Math.round(analytics?.avg_emotion || 0)}
                                    </span>
                                </div>
                                <div className="w-full bg-gray-800 rounded-full h-2">
                                    <div className="bg-green-500 h-2 rounded-full" style={{ width: `${analytics?.avg_emotion || 0}%` }} />
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Recent Users */}
                <div className="bg-gradient-to-br from-gray-900 to-black border border-purple-500/20 rounded-xl p-6">
                    <h3 className="text-xl font-bold text-white mb-4">Recent Users</h3>
                    <div className="space-y-3">
                        {analytics?.recent_users?.map((user, idx) => (
                            <div key={idx} className="flex items-center justify-between p-3 bg-gray-800/50 rounded-lg">
                                <div>
                                    <p className="text-white font-medium">{user.full_name || user.email}</p>
                                    <p className="text-gray-400 text-sm">{user.email}</p>
                                </div>
                                <div className="text-right">
                                    <p className="text-white">{user.video_count} videos</p>
                                    <p className="text-gray-400 text-sm">
                                        Joined {new Date(user.created_at).toLocaleDateString()}
                                    </p>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </main>
        </div>
    );
}