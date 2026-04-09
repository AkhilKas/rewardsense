import { useEffect, useMemo, useState } from "react";
import { getSummary } from "../api/client";
import type { SummaryResponse } from "../types";
import Card from "../components/Card";

function maxValue(values: number[]): number {
  return Math.max(1, ...values);
}

export default function ExpenseSummaryPage() {
  const [data, setData] = useState<SummaryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    async function load() {
      setLoading(true);
      setError("");
      try {
        const summary = await getSummary();
        setData(summary);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load summary");
      } finally {
        setLoading(false);
      }
    }
    void load();
  }, []);

  const maxSpend = useMemo(
    () => maxValue(data?.spend_by_category.map((x) => x.amount) ?? []),
    [data],
  );
  const maxRewards = useMemo(
    () => maxValue(data?.rewards_by_category.map((x) => x.reward_earned) ?? []),
    [data],
  );
  const maxSavings = useMemo(
    () => maxValue(data?.savings_by_card.map((x) => x.savings) ?? []),
    [data],
  );

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-secondary">Expense Summary</h1>
        <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
          Chart-ready summary of spending, rewards, and savings.
        </p>
      </div>

      {loading && (
        <Card>
          <div className="flex items-center justify-center py-10">
            <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
          </div>
        </Card>
      )}

      {error && (
        <Card>
          <p className="text-danger text-sm">{error}</p>
        </Card>
      )}

      {!loading && !error && data && (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <Card>
              <p className="text-xs text-slate-500 dark:text-slate-400">Categories tracked</p>
              <p className="text-2xl font-bold text-secondary mt-1">
                {data.spend_by_category.length}
              </p>
            </Card>
            <Card>
              <p className="text-xs text-slate-500 dark:text-slate-400">Fee-adjusted savings</p>
              <p className="text-2xl font-bold text-secondary mt-1">
                ${data.fee_adjusted_savings_total.toFixed(2)}
              </p>
            </Card>
            <Card>
              <p className="text-xs text-slate-500 dark:text-slate-400">Cards compared</p>
              <p className="text-2xl font-bold text-secondary mt-1">
                {data.savings_by_card.length}
              </p>
            </Card>
          </div>

          <Card>
            <h2 className="text-lg font-semibold text-secondary mb-4">Spend by Category</h2>
            <div className="space-y-3">
              {data.spend_by_category.map((item) => (
                <div key={item.category}>
                  <div className="flex items-center justify-between text-sm mb-1">
                    <span className="text-secondary capitalize">{item.category.replace(/_/g, " ")}</span>
                    <span className="text-slate-500 dark:text-slate-400">${item.amount.toFixed(2)}</span>
                  </div>
                  <div className="h-2 rounded-full bg-border overflow-hidden">
                    <div
                      className="h-full bg-primary"
                      style={{ width: `${(item.amount / maxSpend) * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </Card>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <h2 className="text-lg font-semibold text-secondary mb-4">Rewards by Category</h2>
              <div className="space-y-3">
                {data.rewards_by_category.map((item) => (
                  <div key={item.category}>
                    <div className="flex items-center justify-between text-sm mb-1">
                      <span className="text-secondary capitalize">{item.category.replace(/_/g, " ")}</span>
                      <span className="text-slate-500 dark:text-slate-400">${item.reward_earned.toFixed(2)}</span>
                    </div>
                    <div className="h-2 rounded-full bg-border overflow-hidden">
                      <div
                        className="h-full bg-accent"
                        style={{ width: `${(item.reward_earned / maxRewards) * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </Card>

            <Card>
              <h2 className="text-lg font-semibold text-secondary mb-4">Savings by Card</h2>
              <div className="space-y-3">
                {data.savings_by_card.map((item) => (
                  <div key={item.card_id}>
                    <div className="flex items-center justify-between text-sm mb-1">
                      <span className="text-secondary">{item.card_name}</span>
                      <span className="text-slate-500 dark:text-slate-400">${item.savings.toFixed(2)}</span>
                    </div>
                    <div className="h-2 rounded-full bg-border overflow-hidden">
                      <div
                        className="h-full bg-warning"
                        style={{ width: `${(item.savings / maxSavings) * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        </>
      )}
    </div>
  );
}
