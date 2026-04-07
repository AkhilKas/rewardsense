import { FormEvent, useState } from "react";
import { Link, Navigate, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import Button from "../components/Button";
import Card from "../components/Card";

export default function SignupPage() {
  const { signup, isAuthenticated } = useAuth();
  const navigate = useNavigate();

  const [displayName, setDisplayName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  if (isAuthenticated) {
    return <Navigate to="/recommend" replace />;
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      await signup(email, password, displayName);
      navigate("/recommend", { replace: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Signup failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-md mx-auto pt-12">
      <Card padding="lg">
        <h1 className="text-2xl font-bold text-secondary mb-6">
          Create account
        </h1>

        {error && (
          <div className="mb-4 rounded-md bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 px-4 py-3 text-sm text-red-700 dark:text-red-400">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label
              htmlFor="display-name"
              className="block text-sm font-medium text-secondary mb-1"
            >
              Display name
            </label>
            <input
              id="display-name"
              type="text"
              autoComplete="name"
              required
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              className="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-secondary placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-primary/50 transition-colors"
              placeholder="Jane Smith"
            />
          </div>

          <div>
            <label
              htmlFor="email"
              className="block text-sm font-medium text-secondary mb-1"
            >
              Email
            </label>
            <input
              id="email"
              type="email"
              autoComplete="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-secondary placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-primary/50 transition-colors"
              placeholder="you@example.com"
            />
          </div>

          <div>
            <label
              htmlFor="password"
              className="block text-sm font-medium text-secondary mb-1"
            >
              Password{" "}
              <span className="text-slate-400 font-normal">(8+ characters)</span>
            </label>
            <input
              id="password"
              type="password"
              autoComplete="new-password"
              required
              minLength={8}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-secondary placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-primary/50 transition-colors"
              placeholder="••••••••"
            />
          </div>

          <Button type="submit" className="w-full" disabled={loading}>
            {loading ? "Creating account…" : "Create account"}
          </Button>
        </form>

        <p className="mt-6 text-center text-sm text-slate-500 dark:text-slate-400">
          Already have an account?{" "}
          <Link
            to="/login"
            className="text-primary hover:underline font-medium"
          >
            Sign in
          </Link>
        </p>
      </Card>
    </div>
  );
}
