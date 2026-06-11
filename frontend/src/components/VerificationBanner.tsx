import { useState } from "react";
import { useAuth } from "../context/AuthContext";

export default function VerificationBanner() {
  const { user, resendOtp } = useAuth();
  const [sent, setSent] = useState(false);
  const [loading, setLoading] = useState(false);

  if (!user || user.is_verified) return null;

  async function handleResend() {
    setLoading(true);
    try {
      await resendOtp();
      setSent(true);
    } catch {
      // silently fail
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="w-full bg-amber-50 dark:bg-amber-900/20 border-b border-amber-200 dark:border-amber-800 px-4 py-2.5 flex items-center justify-between gap-4 text-sm">
      <p className="text-amber-800 dark:text-amber-300">
        Please verify your email address to unlock all features.
      </p>
      <div className="flex items-center gap-3 shrink-0">
        {sent ? (
          <span className="text-green-600 dark:text-green-400 font-medium">Code sent!</span>
        ) : (
          <button
            onClick={handleResend}
            disabled={loading}
            className="font-medium text-amber-700 dark:text-amber-300 hover:underline disabled:opacity-50"
          >
            {loading ? "Sending…" : "Resend code"}
          </button>
        )}
      </div>
    </div>
  );
}
