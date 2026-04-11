import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { getSummary } from "../api/client";
import { getCardImage } from "../api/cardImages";
import type { SummaryResponse } from "../types";

const CATEGORY_COLORS: Record<string, string> = {
  dining: "bg-orange-500",
  groceries: "bg-green-500",
  travel: "bg-blue-500",
  gas: "bg-amber-500",
  entertainment: "bg-purple-500",
  online_shopping: "bg-pink-500",
  utilities: "bg-slate-500",
  streaming: "bg-indigo-500",
  other: "bg-gray-400",
};

const CATEGORY_DOT_COLORS: Record<string, string> = {
  dining: "bg-orange-500",
  groceries: "bg-green-500",
  travel: "bg-blue-500",
  gas: "bg-amber-500",
  entertainment: "bg-purple-500",
  online_shopping: "bg-pink-500",
  utilities: "bg-slate-500",
  streaming: "bg-indigo-500",
  other: "bg-gray-400",
};

function StatCard({
  label,
  value,
  detail,
  accent,
}: {
  label: string;
  value: string;
  detail?: string;
  accent?: "green" | "blue" | "amber" | "default";
}) {
  const accentCls =
    accent === "green"
      ? "text-green-600 dark:text-green-400"
      : accent === "blue"
        ? "text-blue-600 dark:text-blue-400"
        : accent === "amber"
          ? "text-amber-600 dark:text-amber-400"
          : "text-slate-900 dark:text-white";

  return (
    <div className="rounded-xl bg-card border border-border p-5">
      <p className="text-xs font-medium uppercase tracking-wide text-slate-500 dark:text-slate-400">
        {label}
      </p>
      <p className={`text-2xl font-bold mt-1 ${accentCls}`}>{value}</p>
      {detail && (
        <p className="text-xs text-slate-400 dark:text-slate-500 mt-1">{detail}</p>
      )}
    </div>
  );
}

function HorizontalBar({
  label,
  value,
  maxValue,
  colorClass,
  suffix,
}: {
  label: string;
  value: number;
  maxValue: number;
  colorClass: string;
  suffix?: string;
}) {
  const pct = maxValue > 0 ? Math.min((value / maxValue) * 100, 100) : 0;
  return (
    <div className="flex items-center gap-3">
      <span className="text-sm text-slate-600 dark:text-slate-300 w-28 shrink-0 capitalize truncate">
        {label.replace(/_/g, " ")}
      </span>
      <div className="flex-1 h-5 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${colorClass}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-sm font-mono font-medium text-slate-700 dark:text-slate-300 w-20 text-right shrink-0">
        ${value.toFixed(2)}{suffix ?? ""}
      </span>
    </div>
  );
}

export default function ExpenseSummaryPage() {
  const [data, setData] = useState<SummaryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      setLoading(true);
      try {
        const res = await getSummary();
        setData(res);
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : "Failed to load summary.");
      } finally {
        setLoading(false);
      }
    }
    void load();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 p-4 text-sm text-red-700 dark:text-red-300">
        {error}
      </div>
    );
  }

  // Empty state
  if (!data || data.transaction_count === 0) {
    return (
      <div className="space-y-6">
        <h1 className="text-2xl font-bold text-slate-900 dark:text-white">
          Expense Summary
        </h1>
        <div className="rounded-xl bg-card border border-border p-12 text-center">
          <div className="mx-auto w-16 h-16 rounded-full bg-slate-100 dark:bg-slate-800 flex items-center justify-center mb-4">
            <svg className="w-8 h-8 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-2">
            No spending data yet
          </h3>
          <p className="text-sm text-slate-500 dark:text-slate-400 mb-6 max-w-sm mx-auto">
            Enable transaction logging in your profile and start recording purchases to see spending insights.
          </p>
          <div className="flex items-center justify-center gap-3">
            <Link
              to="/profile"
              className="px-4 py-2 rounded-lg border border-border bg-card text-sm font-medium text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors"
            >
              Go to Profile
            </Link>
            <Link
              to="/transactions"
              className="px-4 py-2 rounded-lg bg-primary text-white text-sm font-medium hover:bg-primary/90 transition-colors"
            >
              View Transactions
            </Link>
          </div>
        </div>
      </div>
    );
  }

  const maxCatSpend = Math.max(...data.spend_by_category.map((c) => c.total_spend), 1);
  const maxCatReward = Math.max(...data.rewards_by_category.map((c) => c.total_reward), 1);
  const maxCardSavings = Math.max(...data.savings_by_card.map((c) => c.total_savings), 1);

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-2">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 dark:text-white">
            Expense Summary
          </h1>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
            {data.transaction_count} transaction{data.transaction_count !== 1 ? "s" : ""} analyzed
          </p>
        </div>
        <Link
          to="/transactions"
          className="text-sm text-primary hover:text-primary/80 font-medium transition-colors"
        >
          View all transactions &rarr;
        </Link>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="Total Spend"
          value={`$${data.total_spend.toFixed(2)}`}
        />
        <StatCard
          label="Total Rewards"
          value={`$${data.total_rewards.toFixed(2)}`}
          accent="green"
        />
        <StatCard
          label="Total Savings"
          value={`$${data.total_savings.toFixed(2)}`}
          accent="blue"
        />
        <StatCard
          label="Fee-Adjusted Savings"
          value={`$${data.fee_adjusted_savings.toFixed(2)}`}
          detail="After annual card fees"
          accent="amber"
        />
      </div>

      {/* Top insights */}
      {data.top_insights.length > 0 && (
        <div className="rounded-xl bg-gradient-to-r from-primary/5 to-blue-500/5 dark:from-primary/10 dark:to-blue-500/10 border border-primary/20 p-5">
          <h2 className="text-sm font-semibold text-primary mb-3">Top Insights</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            {data.top_insights.map((insight, i) => (
              <div key={i} className="flex flex-col">
                <span className="text-xs text-slate-500 dark:text-slate-400">{insight.label}</span>
                <span className="text-sm font-semibold text-slate-900 dark:text-white mt-0.5">
                  {insight.value}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Spend by category */}
      <div className="rounded-xl bg-card border border-border p-6">
        <h2 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
          Spend by Category
        </h2>
        <div className="space-y-3">
          {data.spend_by_category.map((cat) => (
            <HorizontalBar
              key={cat.category}
              label={cat.category}
              value={cat.total_spend}
              maxValue={maxCatSpend}
              colorClass={CATEGORY_COLORS[cat.category] ?? CATEGORY_COLORS.other}
            />
          ))}
        </div>
        {data.spend_by_category.length > 0 && (
          <div className="flex flex-wrap gap-x-4 gap-y-2 mt-5 pt-4 border-t border-border">
            {data.spend_by_category.map((cat) => (
              <div key={cat.category} className="flex items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400">
                <span className={`w-2.5 h-2.5 rounded-full shrink-0 ${CATEGORY_DOT_COLORS[cat.category] ?? CATEGORY_DOT_COLORS.other}`} />
                <span className="capitalize">{cat.category.replace(/_/g, " ")}</span>
                <span className="font-mono">({cat.transaction_count})</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Rewards by category */}
      <div className="rounded-xl bg-card border border-border p-6">
        <h2 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
          Rewards by Category
        </h2>
        <div className="space-y-3">
          {data.rewards_by_category.map((cat) => (
            <HorizontalBar
              key={cat.category}
              label={cat.category}
              value={cat.total_reward}
              maxValue={maxCatReward}
              colorClass={CATEGORY_COLORS[cat.category] ?? CATEGORY_COLORS.other}
            />
          ))}
        </div>
      </div>

      {/* Savings by card */}
      <div className="rounded-xl bg-card border border-border p-6">
        <h2 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
          Savings by Card
        </h2>
        <div className="space-y-4">
          {data.savings_by_card.map((card, i) => {
            const pct = maxCardSavings > 0 ? (card.total_savings / maxCardSavings) * 100 : 0;
            return (
              <div key={card.card_id ?? i} className="flex items-center gap-4">
                <img
                  src={getCardImage(card.card_id ?? undefined)}
                  alt=""
                  className="w-10 h-7 rounded object-cover shrink-0"
                />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm font-medium text-slate-900 dark:text-white truncate">
                      {card.card_name ?? "Unknown"}
                    </span>
                    <span className="text-sm font-mono font-semibold text-emerald-600 dark:text-emerald-400 shrink-0 ml-2">
                      ${card.total_savings.toFixed(2)}
                    </span>
                  </div>
                  <div className="h-2 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-emerald-500 rounded-full transition-all duration-500"
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <div className="flex items-center gap-3 mt-1 text-xs text-slate-400">
                    <span>{card.transaction_count} txn{card.transaction_count !== 1 ? "s" : ""}</span>
                    <span>${card.total_spend.toFixed(2)} spent</span>
                    <span>${card.total_reward.toFixed(2)} earned</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}