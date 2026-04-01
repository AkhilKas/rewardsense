import { useCallback, useEffect, useState } from "react";

export type ThemePreference = "system" | "light" | "dark";

const STORAGE_KEY = "rewardsense-theme";

function getSystemTheme(): "light" | "dark" {
  if (typeof window === "undefined") return "light";
  return window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

function applyTheme(preference: ThemePreference): void {
  const resolved = preference === "system" ? getSystemTheme() : preference;
  document.documentElement.classList.toggle("dark", resolved === "dark");
}

export function useTheme() {
  const [preference, setPreference] = useState<ThemePreference>(() => {
    try {
      return (
        (localStorage.getItem(STORAGE_KEY) as ThemePreference | null) ||
        "system"
      );
    } catch {
      return "system";
    }
  });

  useEffect(() => {
    applyTheme(preference);
    try {
      localStorage.setItem(STORAGE_KEY, preference);
    } catch {
      /* localStorage unavailable */
    }
  }, [preference]);

  useEffect(() => {
    if (preference !== "system") return;
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const handler = () => applyTheme("system");
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, [preference]);

  const cycle = useCallback(() => {
    setPreference((prev) => {
      if (prev === "system") return "light";
      if (prev === "light") return "dark";
      return "system";
    });
  }, []);

  const resolved: "light" | "dark" =
    preference === "system" ? getSystemTheme() : preference;

  return { preference, resolved, cycle } as const;
}
