import { useState } from "react";
import { recommendTransaction } from "../api/client";
import type { QuickRecommendationViewModel } from "../types/viewmodels";
import Card from "../components/Card";
import Button from "../components/Button";
import CardImage from "../components/CardImage";
import { mapQuickRecommendToVM } from "../viewmodels/viewMappers";

const CATEGORY_OPTIONS = [
  "dining",
  "travel",
  "groceries",
  "gas",
  "online_shopping",
  "entertainment",
  "utilities",
  "other",
];

export default function QuickRecommendPage() {
  const [merchant, setMerchant] = useState("");
  const [amount, setAmount] = useState("");
  const [category, setCategory] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<QuickRecommendationViewModel | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setResult(null);

    const numericAmount = Number(amount);
    if (!merchant.trim() || !Number.isFinite(numericAmount) || numericAmount <= 0) {
      setError("Enter a valid merchant and transaction amount.");
      return;
    }

    setLoading(true);
    try {
      const response = await recommendTransaction({
        merchant: merchant.trim(),
        amount: numericAmount,
        category: category || undefined,
      });
      setResult(mapQuickRecommendToVM(response));
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to get quick recommendation",
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-secondary">Quick Recommendation</h1>
        <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
          Enter one transaction to see which card in your wallet is the best fit.
        </p>
      </div>

      <Card>
        <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="md:col-span-2">
            <label className="block text-sm font-medium text-secondary mb-1">
              Merchant
            </label>
            <input
              type="text"
              value={merchant}
              onChange={(e) => setMerchant(e.target.value)}
              placeholder="e.g., McDonald's"
              className="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-secondary placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-primary/50"
              required
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-secondary mb-1">
              Amount
            </label>
            <input
              type="number"
              min="0.01"
              step="0.01"
              value={amount}
              onChange={(e) => setAmount(e.target.value)}
              placeholder="15.00"
              className="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-secondary placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-primary/50"
              required
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-secondary mb-1">
              Category (optional)
            </label>
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              className="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-secondary focus:outline-none focus:ring-2 focus:ring-primary/50"
            >
              <option value="">Auto-detect</option>
              {CATEGORY_OPTIONS.map((c) => (
                <option key={c} value={c}>
                  {c.replace(/_/g, " ")}
                </option>
              ))}
            </select>
          </div>
          <div className="md:col-span-4 flex justify-end">
            <Button type="submit" loading={loading}>
              Get Recommendation
            </Button>
          </div>
        </form>
      </Card>

      {error && (
        <Card>
          <p className="text-danger text-sm">{error}</p>
        </Card>
      )}

      {result && (
        <Card>
          <div className="mb-4">
            <h2 className="text-lg font-semibold text-secondary">Best Card for This Purchase</h2>
            <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
              {result.context}
            </p>
          </div>

          <div className="space-y-3">
            {result.cards.map((card) => (
              <div
                key={card.id}
                className="rounded-lg border border-border p-3 flex items-center gap-3"
              >
                <CardImage
                  cardId={card.id}
                  alt={`${card.name} card`}
                  className="w-24 h-14 shrink-0"
                />
                <div className="flex-1 min-w-0">
                  <p className="font-semibold text-secondary">
                    #{card.rank} {card.name}
                  </p>
                  <p className="text-sm text-slate-500 dark:text-slate-400">
                    Est. reward: ${card.rewardAmount.toFixed(2)} · Annual fee: $
                    {card.annualFee.toFixed(2)}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
