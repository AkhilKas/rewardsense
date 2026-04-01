import type { ReactNode } from "react";
import { useTheme } from "../hooks/useTheme";
import type { ThemePreference } from "../hooks/useTheme";

const icons: Record<ThemePreference, ReactNode> = {
  system: (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="h-5 w-5"
    >
      <rect x="2" y="3" width="20" height="14" rx="2" />
      <path d="M8 21h8M12 17v4" />
    </svg>
  ),
  light: (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="h-5 w-5"
    >
      <circle cx="12" cy="12" r="5" />
      <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
    </svg>
  ),
  dark: (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="h-5 w-5"
    >
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
    </svg>
  ),
};

const labels: Record<ThemePreference, string> = {
  system: "Using system theme",
  light: "Light mode",
  dark: "Dark mode",
};

export default function ThemeToggle() {
  const { preference, cycle } = useTheme();

  return (
    <button
      onClick={cycle}
      className="p-2 rounded-md text-slate-500 hover:text-secondary hover:bg-slate-100 dark:text-slate-400 dark:hover:bg-slate-700 dark:hover:text-slate-200 transition-colors duration-200 cursor-pointer"
      aria-label={labels[preference]}
      title={labels[preference]}
    >
      {icons[preference]}
    </button>
  );
}
