import { useState } from "react";
import type { FormEvent } from "react";
import { Link, Navigate, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import Button from "../components/Button";
import Card from "../components/Card";

const EMAIL_REGEX = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;

function getPasswordStrength(password: string): {
  label: string;
  color: string;
  width: string;
} {
  if (password.length === 0) return { label: "", color: "", width: "0%" };
  const hasUpper = /[A-Z]/.test(password);
  const hasLower = /[a-z]/.test(password);
  const hasNumber = /[0-9]/.test(password);
  const hasSpecial = /[^A-Za-z0-9]/.test(password);
  const score = [
    password.length >= 8,
    password.length >= 12,
    hasUpper && hasLower,
    hasNumber,
    hasSpecial,
  ].filter(Boolean).length;

  if (score <= 1) return { label: "Weak", color: "bg-red-500", width: "25%" };
  if (score === 2) return { label: "Fair", color: "bg-orange-400", width: "50%" };
  if (score === 3) return { label: "Good", color: "bg-yellow-400", width: "75%" };
  return { label: "Strong", color: "bg-green-500", width: "100%" };
}

export default function SignupPage() {
  const { signup, verifyEmail, resendOtp, isAuthenticated, isLoadingAuth } = useAuth();
  const navigate = useNavigate();

  const [displayName, setDisplayName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [emailError, setEmailError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // OTP step
  const [showOtp, setShowOtp] = useState(false);
  const [otp, setOtp] = useState("");
  const [otpError, setOtpError] = useState<string | null>(null);
  const [resendLoading, setResendLoading] = useState(false);
  const [resendSent, setResendSent] = useState(false);

  const strength = getPasswordStrength(password);

  if (!isLoadingAuth && isAuthenticated && !showOtp) {
    return <Navigate to="/recommend" replace />;
  }

  function validateEmail(value: string) {
    if (!value) { setEmailError(null); return; }
    setEmailError(EMAIL_REGEX.test(value) ? null : "Enter a valid email address.");
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!EMAIL_REGEX.test(email)) {
      setEmailError("Enter a valid email address.");
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const { is_verified } = await signup(email, password, displayName);
      if (!is_verified) {
        setShowOtp(true);
      } else {
        navigate("/recommend", { replace: true });
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Signup failed");
    } finally {
      setLoading(false);
    }
  }

  async function handleVerify(e: FormEvent) {
    e.preventDefault();
    setOtpError(null);
    setLoading(true);
    try {
      await verifyEmail(otp);
      navigate("/recommend", { replace: true });
    } catch (err) {
      setOtpError(err instanceof Error ? err.message : "Verification failed");
    } finally {
      setLoading(false);
    }
  }

  async function handleResend() {
    setResendLoading(true);
    setResendSent(false);
    try {
      await resendOtp();
      setResendSent(true);
    } catch {
      // silently fail
    } finally {
      setResendLoading(false);
    }
  }

  // ── OTP step ──────────────────────────────────────────────────────────────
  if (showOtp) {
    return (
      <div className="min-h-screen bg-surface px-4 py-10 flex items-start justify-center transition-colors duration-200">
        <Card padding="lg" className="w-full max-w-md">
          <h1 className="text-2xl font-bold text-secondary mb-2">Check your email</h1>
          <p className="text-sm text-slate-500 dark:text-slate-400 mb-6">
            We sent a 6-digit code to{" "}
            <span className="font-medium text-secondary">{email}</span>.
            Enter it below to verify your account.
          </p>

          {otpError && (
            <div className="mb-4 rounded-md bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 px-4 py-3 text-sm text-red-700 dark:text-red-400">
              {otpError}
            </div>
          )}

          <form onSubmit={handleVerify} className="space-y-4">
            <div>
              <label htmlFor="otp" className="block text-sm font-medium text-secondary mb-1">
                Verification code
              </label>
              <input
                id="otp"
                type="text"
                inputMode="numeric"
                pattern="[0-9]{6}"
                maxLength={6}
                required
                value={otp}
                onChange={(e) => setOtp(e.target.value.replace(/\D/g, ""))}
                className="w-full rounded-md border border-border bg-surface px-3 py-2 text-lg font-mono tracking-widest text-center text-secondary placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-primary/50 transition-colors"
                placeholder="000000"
                autoFocus
              />
            </div>

            <Button type="submit" className="w-full" loading={loading}>
              Verify email
            </Button>
          </form>

          <div className="mt-4 flex items-center justify-between text-sm">
            <button
              type="button"
              onClick={handleResend}
              disabled={resendLoading}
              className="text-primary hover:underline disabled:opacity-50"
            >
              {resendLoading ? "Sending…" : "Resend code"}
            </button>
            {resendSent && (
              <span className="text-green-600 dark:text-green-400 text-xs">Sent!</span>
            )}
            <button
              type="button"
              onClick={() => navigate("/recommend", { replace: true })}
              className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-300"
            >
              Skip for now
            </button>
          </div>
        </Card>
      </div>
    );
  }

  // ── Signup form ───────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-surface px-4 py-10 flex items-start justify-center transition-colors duration-200">
      <Card padding="lg" className="w-full max-w-md">
        <div className="flex justify-end mb-2">
          <Link
            to="/"
            aria-label="Close and return home"
            className="inline-flex items-center justify-center w-8 h-8 rounded-md text-slate-500 hover:text-secondary hover:bg-slate-100 dark:text-slate-400 dark:hover:bg-slate-700 transition-colors"
          >
            ×
          </Link>
        </div>
        <h1 className="text-2xl font-bold text-secondary mb-6">Create account</h1>

        {error && (
          <div className="mb-4 rounded-md bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 px-4 py-3 text-sm text-red-700 dark:text-red-400">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="display-name" className="block text-sm font-medium text-secondary mb-1">
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
            <label htmlFor="email" className="block text-sm font-medium text-secondary mb-1">
              Email
            </label>
            <input
              id="email"
              type="email"
              autoComplete="email"
              required
              value={email}
              onChange={(e) => { setEmail(e.target.value); validateEmail(e.target.value); }}
              onBlur={(e) => validateEmail(e.target.value)}
              className={`w-full rounded-md border px-3 py-2 text-sm text-secondary placeholder-slate-400 bg-surface focus:outline-none focus:ring-2 focus:ring-primary/50 transition-colors ${emailError ? "border-red-400" : "border-border"}`}
              placeholder="you@example.com"
            />
            {emailError && (
              <p className="mt-1 text-xs text-red-500">{emailError}</p>
            )}
          </div>

          <div>
            <label htmlFor="password" className="block text-sm font-medium text-secondary mb-1">
              Password
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
            {password.length > 0 && (
              <div className="mt-2">
                <div className="h-1.5 w-full bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-300 ${strength.color}`}
                    style={{ width: strength.width }}
                  />
                </div>
                <p className={`mt-1 text-xs font-medium ${
                  strength.label === "Strong" ? "text-green-600 dark:text-green-400" :
                  strength.label === "Good" ? "text-yellow-600 dark:text-yellow-400" :
                  strength.label === "Fair" ? "text-orange-500" : "text-red-500"
                }`}>
                  {strength.label}
                  {strength.label === "Weak" && " — add uppercase, numbers, or symbols"}
                  {strength.label === "Fair" && " — try mixing letters and numbers"}
                  {strength.label === "Good" && " — add symbols to make it stronger"}
                </p>
              </div>
            )}
          </div>

          <Button type="submit" className="w-full" loading={loading}>
            Create account
          </Button>
        </form>

        <p className="mt-6 text-center text-sm text-slate-500 dark:text-slate-400">
          Already have an account?{" "}
          <Link to="/login" className="text-primary hover:underline font-medium">
            Sign in
          </Link>
        </p>
      </Card>
    </div>
  );
}
