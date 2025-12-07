import { Link, useLocation } from "react-router-dom";
import React, { useState } from "react";
import { Button } from "../components/ui/button";
import { Card } from "../components/ui/card";
import { Search, Brain, Rocket, ArrowRight, Check, Loader2, Mic, MessageSquare, Target } from "lucide-react";

const features = [
  {
    icon: Mic,
    title: "Public Speaking",
    description: "Practice and perfect your public speaking skills with AI-powered real-time feedback",
    link: "/public-speaking"
  },
  {
    icon: MessageSquare,
    title: "Interview Analysis",
    description: "Get detailed feedback on your interview performance with personalized improvement tips",
    link: "/interview-analysis"
  },
  {
    icon: Target,
    title: "Pitching",
    description: "Master your pitch delivery with comprehensive analysis and actionable insights",
    link: "/pitching"
  },
];

export default function Home() {
  return (
    <div className="min-h-screen bg-black">
      {/* Hero Section */}
      <section className="relative bg-gradient-to-br from-emerald-600 via-green-600 to-teal-600 overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_30%_20%,rgba(255,255,255,0.1),transparent_50%)]" />
        <div className="relative max-w-7xl mx-auto px-6 py-24 md:py-32 text-center">
          <h1 className="text-5xl md:text-7xl font-bold text-white mb-6 tracking-tight">
            PitchQuest
          </h1>
          <p className="text-xl md:text-2xl text-white/95 font-medium mb-3">
            Get real time feedback on real world simulations for interviews, public speaking and pitching
          </p>
        </div>
      </section>

      {/* Features Section */}
      <section className="max-w-6xl mx-auto px-6 -mt-12 relative z-10">
        <div className="grid md:grid-cols-3 gap-6">
          {features.map((feature, index) => (
            <Link to={feature.link} key={index}>
              <Card className="p-8 bg-gradient-to-br from-gray-900 to-black shadow-xl hover:shadow-2xl transition-all duration-300 border border-green-500/20 hover:border-green-500/40 hover:-translate-y-1 cursor-pointer">
                <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-emerald-500 to-green-600 flex items-center justify-center mb-6 shadow-lg">
                  <feature.icon className="w-7 h-7 text-white" />
                </div>
                <h3 className="text-lg font-bold text-white mb-3">
                  {feature.title}
                </h3>
                <p className="text-gray-400 text-sm leading-relaxed">
                  {feature.description}
                </p>
              </Card>
            </Link>
          ))}
        </div>
      </section>

      {/* Footer Spacing */}
      <div className="h-20" />
    </div>
  );
}