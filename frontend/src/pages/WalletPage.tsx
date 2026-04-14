import { useEffect, useMemo, useState } from "react";
import { getCardCatalog, getMe, updateSavedCards } from "../api/client";
import type { CardCatalogItem } from "../types";
import Card from "../components/Card";
import Button from "../components/Button";
import CardImage from "../components/CardImage";

function sameCardSelection(a: string[], b: string[]): boolean {
  if (a.length !== b.length) return false;
  const sa = [...a].sort();
  const sb = [...b].sort();
  return sa.every((id, i) => id === sb[i]);
}

export default function WalletPage() {
  const [catalog, setCatalog] = useState<CardCatalogItem[]>([]);
  const [savedCardIds, setSavedCardIds] = useState<string[]>([]);
  const [initialSavedIds, setInitialSavedIds] = useState<string[]>([]);
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");

  useEffect(() => {
    async function load() {
      setLoading(true);
      setError("");
      try {
        const [me, cards] = await Promise.all([getMe(), getCardCatalog()]);
        const ids = me.saved_card_ids;
        setSavedCardIds(ids);
        setInitialSavedIds(ids);
        setCatalog(cards);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load wallet");
      } finally {
        setLoading(false);
      }
    }

    void load();
  }, []);

  const visibleCards = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return catalog;
    return catalog.filter(
      (c) =>
        c.card_name.toLowerCase().includes(q) ||
        c.issuer.toLowerCase().includes(q) ||
        c.reward_highlights.some((h) => h.toLowerCase().includes(q)),
    );
  }, [catalog, search]);

  const dirty = useMemo(
    () => !sameCardSelection(savedCardIds, initialSavedIds),
    [savedCardIds, initialSavedIds],
  );

  const catalogById = useMemo(() => {
    const m = new Map<string, CardCatalogItem>();
    for (const c of catalog) m.set(c.card_id, c);
    return m;
  }, [catalog]);

  function removeCardFromWallet(cardId: string) {
    setSavedCardIds((prev) => prev.filter((id) => id !== cardId));
  }

  function toggleCard(cardId: string) {
    setSavedCardIds((prev) =>
      prev.includes(cardId) ? prev.filter((id) => id !== cardId) : [...prev, cardId],
    );
  }

  async function handleSave() {
    setSaving(true);
    setError("");
    setMessage("");
    try {
      await updateSavedCards(savedCardIds);
      setInitialSavedIds([...savedCardIds]);
      setMessage("Wallet updated successfully.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save wallet");
    } finally {
      setSaving(false);
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
      </div>
    );
  }

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-secondary">Your Wallet</h1>
        <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
          Select the cards you currently hold. Recommendations will prioritize your wallet.
        </p>
      </div>

      <div
        className={`sticky top-16 z-40 -mx-4 px-4 py-3 sm:-mx-6 sm:px-6 rounded-lg border transition-colors duration-200 ${
          dirty
            ? "border-primary/40 bg-primary/5 dark:bg-primary/10 shadow-sm"
            : "border-border bg-card/90 dark:bg-card/90 backdrop-blur-sm"
        }`}
      >
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          {dirty ? (
            <p
              className="text-xs font-medium text-amber-800 dark:text-amber-300 animate-pulse"
              aria-live="polite"
            >
              Unsaved changes
            </p>
          ) : (
            <span className="min-h-[1.25rem] sm:flex-1" aria-hidden />
          )}
          <Button
            type="button"
            onClick={handleSave}
            loading={saving}
            className={`w-full sm:w-auto shrink-0 ${dirty ? "shadow-md shadow-primary/20" : ""}`}
          >
            Save Wallet
          </Button>
        </div>
      </div>

      <Card>
        <div className="flex flex-col sm:flex-row gap-3 sm:items-center sm:justify-between">
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search by card name, issuer, or benefit..."
            className="w-full sm:max-w-md rounded-md border border-border bg-surface px-3 py-2 text-sm text-secondary placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-primary/50 transition-colors"
          />
          <div className="text-sm text-slate-500 dark:text-slate-400">
            {savedCardIds.length} cards selected
          </div>
        </div>
        {savedCardIds.length > 0 && (
          <div className="mt-4 pt-3 border-t border-border">
            <p className="text-xs font-medium text-secondary mb-2">Selected cards</p>
            <div className="flex flex-wrap gap-2">
              {savedCardIds.map((cardId) => {
                const card = catalogById.get(cardId);
                const title = card?.card_name ?? cardId;
                return (
                  <span
                    key={cardId}
                    className="inline-flex items-center gap-1.5 rounded-full border border-primary/40 bg-primary/10 dark:bg-primary/15 px-2.5 py-1 text-xs font-medium text-secondary"
                  >
                    <span className="max-w-[200px] truncate sm:max-w-[260px]">
                      {title}
                    </span>
                    <button
                      type="button"
                      onClick={() => removeCardFromWallet(cardId)}
                      className="shrink-0 rounded-full p-0.5 text-slate-600 hover:bg-primary/20 hover:text-secondary dark:text-slate-300 dark:hover:bg-primary/25"
                      aria-label={`Remove ${title}`}
                    >
                      ×
                    </button>
                  </span>
                );
              })}
            </div>
          </div>
        )}
      </Card>

      {error && (
        <Card>
          <p className="text-danger text-sm">{error}</p>
        </Card>
      )}

      {message && (
        <Card>
          <p className="text-accent text-sm">{message}</p>
        </Card>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {visibleCards.map((card) => {
          const checked = savedCardIds.includes(card.card_id);
          return (
            <label
              key={card.card_id}
              className={`block rounded-xl border transition-colors cursor-pointer ${
                checked
                  ? "border-primary bg-primary/5"
                  : "border-border bg-card hover:border-primary/40"
              }`}
            >
              <div className="p-4 space-y-3">
                <div className="flex items-start gap-3">
                  <CardImage
                    cardId={card.card_id}
                    issuer={card.issuer}
                    alt={`${card.card_name} card`}
                    className="w-24 h-14 shrink-0"
                  />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between gap-2">
                      <h3 className="font-semibold text-secondary text-sm leading-tight">
                        {card.card_name}
                      </h3>
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={() => toggleCard(card.card_id)}
                        onClick={(e) => e.stopPropagation()}
                        className="h-4 w-4 rounded border-border text-primary focus:ring-primary/50"
                      />
                    </div>
                    <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                      {card.issuer} · {card.annual_fee === 0 ? "No annual fee" : `$${card.annual_fee}/yr`}
                    </p>
                  </div>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {card.reward_highlights.map((h) => (
                    <span
                      key={`${card.card_id}-${h}`}
                      className="text-[11px] px-2 py-1 rounded-full bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300"
                    >
                      {h}
                    </span>
                  ))}
                </div>
              </div>
            </label>
          );
        })}
      </div>
    </div>
  );
}
