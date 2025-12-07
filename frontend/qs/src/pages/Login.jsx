import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";

export default function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const navigate = useNavigate();

  const handleSubmit = (e) => {
    e.preventDefault();
    localStorage.setItem("isAuthenticated", "true");
    navigate("/");
  };

  return (
    <div className="min-h-screen flex">
      {/* Left side - Login Form (30%) */}
      <div className="w-full lg:w-[30%] bg-white flex items-center justify-center p-8">
        <div className="w-full max-w-md">
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-gray-900 mb-2">Welcome Back</h1>
            <p className="text-gray-600">Sign in to PitchQuest</p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Email
              </label>
              <Input
                type="email"
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Password
              </label>
              <Input
                type="password"
                placeholder="Enter your password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>

            <div className="flex items-center justify-between text-sm">
              <label className="flex items-center">
                <input type="checkbox" className="mr-2" />
                <span className="text-gray-600">Remember me</span>
              </label>
              <a href="#" className="text-emerald-600 hover:text-emerald-700">
                Forgot password?
              </a>
            </div>

            <Button type="submit" className="w-full bg-emerald-600 hover:bg-emerald-700" size="lg">
              Sign In
            </Button>
          </form>

          <div className="mt-6 text-center text-sm text-gray-600">
            Don't have an account?{" "}
            <Link to="/signup" className="text-emerald-600 hover:text-emerald-700 font-semibold">
              Sign up
            </Link>
          </div>
        </div>
      </div>

      {/* Right side - Hero (70%) */}
      <div className="hidden lg:flex lg:w-[70%] bg-gradient-to-br from-emerald-600 via-green-600 to-teal-600 items-center justify-center p-12 relative overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_30%_20%,rgba(255,255,255,0.1),transparent_50%)]" />
        <div className="relative text-center">
          <h1 className="text-7xl font-bold text-white mb-6 tracking-tight">
            PitchQuest
          </h1>
          <p className="text-2xl text-white/95 font-medium mb-4 max-w-3xl mx-auto leading-relaxed">
            Get real time feedback on real world simulations for interviews, public speaking and pitching
          </p>
        </div>
      </div>
    </div>
  );
}