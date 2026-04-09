import { useEffect, useRef, useState } from "react";
import { useAuth } from "../context/AuthContext";
import { getMe, updateProfile, updateSavedCards, getCardCatalog } from "../api/client";
import type { CardCatalogItem } from "../types";
import Button from "../components/Button";
import Card from "../components/Card";
import { applyThemePreference } from "../hooks/useTheme";

const PERSONA_OPTIONS = [
  {
    key: "student",
    label: "Student",
    description: "Minimize fees, favor no-fee simple cashback",
  },
  {
    key: "traveler",
    label: "Traveler",
    description: "Maximize travel multipliers and transferable points",
  },
  {
    key: "family",
    label: "Family",
    description: "Groceries, gas, and utilities focus",
  },
  {
    key: "cashback-focused",
    label: "Cashback-focused",
    description: "Flat-rate cashback simplicity",
  },
];

const REWARD_OPTIONS = [
  { value: "cashback", label: "Cash back" },
  { value: "points", label: "Points" },
  { value: "miles", label: "Miles" },
];

export default function ProfilePage() {
  const { user } = useAuth();
  const [catalog, setCatalog] = useState<CardCatalogItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Local editable state — kept in sync with profile on load/save
  const [displayName, setDisplayName] = useState("");
  const [personas, setPersonas] = useState<string[]>([]);
  const [rewardPref, setRewardPref] = useState("cashback");
  const [loggingEnabled, setLoggingEnabled] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  const [savedDarkMode, setSavedDarkMode] = useState(false);
  const [savedCardIds, setSavedCardIds] = useState<string[]>([]);

  // Per-section save states
  const [savingProfile, setSavingProfile] = useState(false);
  const [savingSettings, setSavingSettings] = useState(false);
  const [savingCards, setSavingCards] = useState(false);
  const [saveMsg, setSaveMsg] = useState<Record<string, string>>({});
  const darkModeRef = useRef(darkMode);
  const savedDarkModeRef = useRef(savedDarkMode);

  useEffect(() => {
    Promise.all([getMe(), getCardCatalog()])
      .then(([p, c]) => {
        setCatalog(c);
        setDisplayName(p.display_name);
        setPersonas(p.personas);
        setRewardPref(p.reward_preference);
        setLoggingEnabled(p.transaction_logging_enabled);
        setDarkMode(p.dark_mode);
        setSavedDarkMode(p.dark_mode);
        applyThemePreference(p.dark_mode ? "dark" : "light");
        setSavedCardIds(p.saved_card_ids);
      })
      .catch((e: unknown) =>
        setError(e instanceof Error ? e.message : "Failed to load profile"),
      )
      .finally(() => setLoading(false));
  }, []);

  function flash(section: string, msg: string) {
    setSaveMsg((prev) => ({ ...prev, [section]: msg }));
    setTimeout(
      () => setSaveMsg((prev) => ({ ...prev, [section]: "" })),
      2500,
    );
  }

  async function saveDisplayName() {
    setSavingProfile(true);
    try {
      await updateProfile({ display_name: displayName });
      flash("profile", "Saved");
    } catch (e: unknown) {
      flash("profile", e instanceof Error ? e.message : "Save failed");
    } finally {
      setSavingProfile(false);
    }
  }

  async function togglePersona(key: string) {
    const next = personas.includes(key)
      ? personas.filter((p) => p !== key)
      : [...personas, key];
    setPersonas(next);
    try {
      await updateProfile({ personas: next });
    } catch {
      // revert on failure
      setPersonas(personas);
    }
  }

  async function saveSettings() {
    setSavingSettings(true);
    const nextDarkMode = darkMode;
    const nextLoggingEnabled = loggingEnabled;
    const nextRewardPref = rewardPref;
    try {
      const updated = await updateProfile({
        reward_preference: nextRewardPref,
        transaction_logging_enabled: nextLoggingEnabled,
        dark_mode: nextDarkMode,
      });
      setRewardPref(updated.reward_preference);
      setLoggingEnabled(updated.transaction_logging_enabled);
      // Keep UI on the just-saved choice even if any stale response races in.
      setDarkMode(nextDarkMode);
      setSavedDarkMode(nextDarkMode);
      applyThemePreference(nextDarkMode ? "dark" : "light");
      flash("settings", "Saved");
    } catch (e: unknown) {
      flash("settings", e instanceof Error ? e.message : "Save failed");
    } finally {
      setSavingSettings(false);
    }
  }

  useEffect(() => {
    darkModeRef.current = darkMode;
  }, [darkMode]);

  useEffect(() => {
    savedDarkModeRef.current = savedDarkMode;
  }, [savedDarkMode]);

  // Dark mode should preview instantly on this page, but revert if user leaves without saving.
  useEffect(() => {
    return () => {
      if (darkModeRef.current !== savedDarkModeRef.current) {
        applyThemePreference(savedDarkModeRef.current ? "dark" : "light");
      }
    };
  }, []);

  async function saveWallet() {
    setSavingCards(true);
    try {
      await updateSavedCards(savedCardIds);
      flash("cards", "Wallet saved");
    } catch (e: unknown) {
      flash("cards", e instanceof Error ? e.message : "Save failed");
    } finally {
      setSavingCards(false);
    }
  }

  function toggleCard(cardId: string) {
    setSavedCardIds((prev) =>
      prev.includes(cardId)
        ? prev.filter((id) => id !== cardId)
        : [...prev, cardId],
    );
  }

  if (loading) {
    return (
      <div className="flex justify-center py-16">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-lg mx-auto pt-8">
        <div className="rounded-md bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 px-4 py-3 text-sm text-red-700 dark:text-red-400">
          {error}
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-2xl mx-auto space-y-8">
      <h1 className="text-2xl font-bold text-secondary">Profile &amp; Settings</h1>

      {/* ---- Display Name ---- */}
      <Card padding="lg">
        <h2 className="text-lg font-semibold text-secondary mb-4">Profile</h2>
        <p className="text-sm text-slate-500 dark:text-slate-400 mb-1">
          Email
        </p>
        <p className="text-sm text-secondary mb-4">{user?.email}</p>

        <label
          htmlFor="display-name"
          className="block text-sm font-medium text-secondary mb-1"
        >
          Display name
        </label>
        <div className="flex gap-3">
          <input
            id="display-name"
            type="text"
            value={displayName}
            onChange={(e) => setDisplayName(e.target.value)}
            className="flex-1 rounded-md border border-border bg-surface px-3 py-2 text-sm text-secondary focus:outline-none focus:ring-2 focus:ring-primary/50 transition-colors"
          />
          <Button onClick={saveDisplayName} disabled={savingProfile}>
            {savingProfile ? "Saving…" : "Save"}
          </Button>
        </div>
        {saveMsg.profile && (
          <p className="mt-2 text-sm text-green-600 dark:text-green-400">
            {saveMsg.profile}
          </p>
        )}
      </Card>

      {/* ---- Personas ---- */}
      <Card padding="lg">
        <h2 className="text-lg font-semibold text-secondary mb-1">Personas</h2>
        <p className="text-sm text-slate-500 dark:text-slate-400 mb-4">
          Select all that apply. Personas shape how recommendations are ranked for you.
        </p>
        <div className="space-y-3">
          {PERSONA_OPTIONS.map((p) => (
            <label
              key={p.key}
              className="flex items-start gap-3 cursor-pointer group"
            >
              <input
                type="checkbox"
                checked={personas.includes(p.key)}
                onChange={() => togglePersona(p.key)}
                className="mt-0.5 h-4 w-4 rounded border-border text-primary focus:ring-primary/50 cursor-pointer"
              />
              <div>
                <p className="text-sm font-medium text-secondary group-hover:text-primary transition-colors">
                  {p.label}
                </p>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  {p.description}
                </p>
              </div>
            </label>
          ))}
        </div>
      </Card>

      {/* ---- Settings ---- */}
      <Card padding="lg">
        <h2 className="text-lg font-semibold text-secondary mb-4">Settings</h2>

        <div className="space-y-5">
          <div>
            <label
              htmlFor="reward-pref"
              className="block text-sm font-medium text-secondary mb-1"
            >
              Reward preference
            </label>
            <select
              id="reward-pref"
              value={rewardPref}
              onChange={(e) => setRewardPref(e.target.value)}
              className="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-secondary focus:outline-none focus:ring-2 focus:ring-primary/50 transition-colors"
            >
              {REWARD_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          </div>

          <label className="flex items-center justify-between gap-4 cursor-pointer">
            <div>
              <p className="text-sm font-medium text-secondary">
                Transaction logging
              </p>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Enables personalised recommendations based on your history.
                Disabling makes recommendations more generic.
              </p>
            </div>
            <button
              role="switch"
              aria-checked={loggingEnabled}
              onClick={() => setLoggingEnabled((v) => !v)}
              className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-primary/50 ${
                loggingEnabled ? "bg-primary" : "bg-slate-300 dark:bg-slate-600"
              }`}
            >
              <span
                className={`inline-block h-5 w-5 rounded-full bg-white shadow transform transition-transform duration-200 ${
                  loggingEnabled ? "translate-x-5" : "translate-x-0"
                }`}
              />
            </button>
          </label>

          <label className="flex items-center justify-between gap-4 cursor-pointer">
            <div>
              <p className="text-sm font-medium text-secondary">Dark mode</p>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Persist dark mode preference to your account.
              </p>
            </div>
            <button
              role="switch"
              aria-checked={darkMode}
              onClick={() => {
                setDarkMode((v) => {
                  const next = !v;
                  applyThemePreference(next ? "dark" : "light");
                  return next;
                });
              }}
              className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-primary/50 ${
                darkMode ? "bg-primary" : "bg-slate-300 dark:bg-slate-600"
              }`}
            >
              <span
                className={`inline-block h-5 w-5 rounded-full bg-white shadow transform transition-transform duration-200 ${
                  darkMode ? "translate-x-5" : "translate-x-0"
                }`}
              />
            </button>
          </label>
        </div>

        <div className="mt-5 flex items-center gap-3">
          <Button onClick={saveSettings} disabled={savingSettings}>
            {savingSettings ? "Saving…" : "Save settings"}
          </Button>
          {saveMsg.settings && (
            <p className="text-sm text-green-600 dark:text-green-400">
              {saveMsg.settings}
            </p>
          )}
        </div>
      </Card>

      {/* ---- Wallet ---- */}
      <Card padding="lg">
        <h2 className="text-lg font-semibold text-secondary mb-1">
          Saved cards
        </h2>
        <p className="text-sm text-slate-500 dark:text-slate-400 mb-4">
          Select the cards in your wallet. Recommendations will rank these first.
        </p>
        <div className="space-y-2">
          {catalog.map((card) => (
            <label
              key={card.card_id}
              className="flex items-center gap-3 cursor-pointer group"
            >
              <input
                type="checkbox"
                checked={savedCardIds.includes(card.card_id)}
                onChange={() => toggleCard(card.card_id)}
                className="h-4 w-4 rounded border-border text-primary focus:ring-primary/50 cursor-pointer"
              />
              <div className="flex items-center justify-between flex-1">
                <p className="text-sm font-medium text-secondary group-hover:text-primary transition-colors">
                  {card.card_name}
                </p>
                <span className="text-xs text-slate-400">
                  {card.annual_fee === 0
                    ? "No annual fee"
                    : `$${card.annual_fee}/yr`}
                </span>
              </div>
            </label>
          ))}
        </div>
        <div className="mt-5 flex items-center gap-3">
          <Button onClick={saveWallet} disabled={savingCards}>
            {savingCards ? "Saving…" : "Save wallet"}
          </Button>
          {saveMsg.cards && (
            <p className="text-sm text-green-600 dark:text-green-400">
              {saveMsg.cards}
            </p>
          )}
        </div>
      </Card>
    </div>
  );
}
