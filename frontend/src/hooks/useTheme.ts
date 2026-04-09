import { useCallback, useEffect, useState } from "react";

export type ThemePreference = "system" | "light" | "dark";

const STORAGE_KEY = "rewardsense-theme";
const THEME_EVENT = "rewardsense-theme-change";

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

export function applyThemePreference(preference: ThemePreference): void {
  applyTheme(preference);
  try {
    localStorage.setItem(STORAGE_KEY, preference);
  } catch {
    /* localStorage unavailable */
  }
  window.dispatchEvent(new CustomEvent<ThemePreference>(THEME_EVENT, { detail: preference }));
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

  const commitPreference = useCallback((next: ThemePreference) => {
    applyThemePreference(next);
    setPreference(next);
  }, []);

  useEffect(() => {
    // If another part of the app updated the preference before this hook mounted,
    // synchronize from storage first so stale local state cannot override it.
    try {
      const stored = localStorage.getItem(STORAGE_KEY) as ThemePreference | null;
      if (stored && stored !== preference) {
        setPreference(stored);
        return;
      }
    } catch {
      /* localStorage unavailable */
    }
    applyTheme(preference);
  }, [preference]);

  useEffect(() => {
    if (preference !== "system") return;
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const handler = () => applyTheme("system");
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, [preference]);

  useEffect(() => {
    const onThemeEvent = (e: Event) => {
      const pref = (e as CustomEvent<ThemePreference>).detail;
      if (pref && pref !== preference) {
        setPreference(pref);
      }
    };
    const onStorage = (e: StorageEvent) => {
      if (e.key !== STORAGE_KEY || !e.newValue) return;
      const pref = e.newValue as ThemePreference;
      if (pref !== preference) {
        setPreference(pref);
      }
    };
    window.addEventListener(THEME_EVENT, onThemeEvent);
    window.addEventListener("storage", onStorage);
    return () => {
      window.removeEventListener(THEME_EVENT, onThemeEvent);
      window.removeEventListener("storage", onStorage);
    };
  }, [preference]);

  const cycle = useCallback(() => {
    const next =
      preference === "system" ? "light" : preference === "light" ? "dark" : "system";
    commitPreference(next);
  }, [preference, commitPreference]);

  const setThemePreference = useCallback((next: ThemePreference) => {
    commitPreference(next);
  }, [commitPreference]);

  const resolved: "light" | "dark" =
    preference === "system" ? getSystemTheme() : preference;

  return { preference, resolved, cycle, setPreference: setThemePreference } as const;
}
