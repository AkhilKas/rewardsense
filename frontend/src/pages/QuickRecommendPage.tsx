import { useState } from "react";
import { createTransaction, recommendQuickTransaction } from "../api/client";
import type { TransactionCreateRequest } from "../types";
import type { QuickCardViewModel, QuickRecommendationViewModel } from "../types/viewmodels";
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

/** Matches `TransactionCreateRequest` in `src/app/transactions/schemas.py`. */
function buildTransactionPayload(
  merchantRaw: string,
  amount: number,
  categoryRaw: string,
  card: QuickCardViewModel,
): TransactionCreateRequest {
  const merchant = merchantRaw.trim().slice(0, 200);
  const category = (categoryRaw.trim().toLowerCase().slice(0, 50) || "other").slice(
    0,
    50,
  );
  const reward = Number(card.rewardAmount);
  const rewardEarned = Number.isFinite(reward) && reward >= 0 ? reward : 0;
  const savings = Number.isFinite(reward) ? reward : 0;

  const payload: TransactionCreateRequest = {
    merchant,
    amount,
    category,
    reward_earned: rewardEarned,
    estimated_savings: savings,
    source_flow: "transaction",
  };

  const cid = card.id?.trim();
  if (cid) payload.chosen_card_id = cid;
  const cname = card.name?.trim();
  if (cname) payload.chosen_card_name = cname;

  return payload;
}

export default function QuickRecommendPage() {
  const [merchant, setMerchant] = useState("");
  const [amount, setAmount] = useState("");
  const [category, setCategory] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<QuickRecommendationViewModel | null>(null);
  const [logBusyId, setLogBusyId] = useState<string | null>(null);
  const [logToast, setLogToast] = useState<string | null>(null);

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
      const response = await recommendQuickTransaction({
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

  async function handleLogTransaction(card: QuickCardViewModel) {
    const numericAmount = Number(amount);
    if (!merchant.trim() || !Number.isFinite(numericAmount) || numericAmount <= 0) {
      setError("Enter a valid merchant and transaction amount before logging.");
      return;
    }
    setError("");
    setLogBusyId(card.id);
    try {
      const fromResult = (result?.categoryUsed ?? "").trim().toLowerCase();
      const fromForm = category.trim().toLowerCase();
      const categoryForTxn = fromResult || fromForm || "other";

      await createTransaction(
        buildTransactionPayload(merchant, numericAmount, categoryForTxn, card),
      );
      const amt = numericAmount.toLocaleString(undefined, {
        style: "currency",
        currency: "USD",
      });
      setLogToast(
        `Transaction logged: ${amt} at ${merchant.trim()} on ${card.name}`,
      );
      window.setTimeout(() => setLogToast(null), 5000);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Could not log this transaction.",
      );
    } finally {
      setLogBusyId(null);
    }
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-secondary">Quick Check</h1>
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

      {logToast && (
        <div
          role="status"
          className="rounded-md border border-accent/40 bg-accent/10 dark:bg-accent/15 px-4 py-3 text-sm text-secondary"
        >
          {logToast}
        </div>
      )}

      {result && !result.hasSavedCards && (
        <Card>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            Add cards to your{" "}
            <a href="/wallet" className="text-primary hover:underline font-medium">wallet</a>{" "}
            to get personalized recommendations.
          </p>
        </Card>
      )}

      {result && result.hasSavedCards && (
        <Card>
          <div className="mb-4">
            <div className="flex items-center gap-2 mb-1">
              <h2 className="text-lg font-semibold text-secondary">Best Card for This Purchase</h2>
              <span className="text-xs px-2 py-0.5 rounded-full bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300 capitalize">
                {result.categoryUsed.replace(/_/g, " ")}
              </span>
            </div>
            <p className="text-sm text-slate-500 dark:text-slate-400">
              {result.context}
            </p>
            {result.moneySaved > 0 && (
              <p className="text-sm font-medium text-accent mt-1">
                Est. reward: ${result.moneySaved.toFixed(2)}
              </p>
            )}
          </div>

          <div className="space-y-3">
            {result.cards.map((card) => (
              <div
                key={card.id}
                className="rounded-lg border border-border p-3 flex flex-col gap-3 sm:flex-row sm:items-center sm:gap-4"
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
                <Button
                  type="button"
                  variant="secondary"
                  size="sm"
                  className="shrink-0 self-start sm:self-center whitespace-normal text-left"
                  loading={logBusyId === card.id}
                  onClick={() => void handleLogTransaction(card)}
                >
                  Use this card — log transaction
                </Button>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
