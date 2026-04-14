import { useCallback, useEffect, useState } from "react";
import { getTransactions, getSummary, exportTransactions, createTransaction } from "../api/client";
import { getCardImage } from "../api/cardImages";
import type { TransactionsResponse } from "../types";

const PAGE_SIZE = 10;

const CATEGORY_COLORS: Record<string, string> = {
  dining: "bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300",
  groceries: "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300",
  travel: "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300",
  gas: "bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300",
  entertainment: "bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300",
  online_shopping: "bg-pink-100 text-pink-800 dark:bg-pink-900/30 dark:text-pink-300",
  utilities: "bg-slate-100 text-slate-800 dark:bg-slate-700 dark:text-slate-300",
  streaming: "bg-indigo-100 text-indigo-800 dark:bg-indigo-900/30 dark:text-indigo-300",
  other: "bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300",
};

function categoryBadge(category: string) {
  const cls = CATEGORY_COLORS[category] ?? CATEGORY_COLORS.other;
  return (
    <span className={`inline-block px-2 py-0.5 rounded-full text-xs font-medium capitalize ${cls}`}>
      {category.replace(/_/g, " ")}
    </span>
  );
}

function formatDate(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
  } catch {
    return iso;
  }
}

function formatTime(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });
  } catch {
    return "";
  }
}

function formatMoney(n: number | undefined | null): string {
  const v = Number(n);
  return Number.isFinite(v) ? v.toFixed(2) : "0.00";
}

/** Ledger-wide totals (all transactions), from GET /summary — same math as Expense Summary. */
type LedgerTotals = {
  rewards: number;
  savings: number;
  transactionCount: number;
};

export default function TransactionHistoryPage() {
  const [data, setData] = useState<TransactionsResponse | null>(null);
  const [ledgerTotals, setLedgerTotals] = useState<LedgerTotals | null>(null);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);
  const [showAddForm, setShowAddForm] = useState(false);
  const [newMerchant, setNewMerchant] = useState("");
  const [newAmount, setNewAmount] = useState("");
  const [newCategory, setNewCategory] = useState("other");
  const [addLoading, setAddLoading] = useState(false);

  const fetchPage = useCallback(async (p: number) => {
    setLoading(true);
    setError(null);
    try {
      const [txRes, summaryRes] = await Promise.all([
        getTransactions(p, PAGE_SIZE),
        getSummary(),
      ]);
      setData(txRes);
      setLedgerTotals({
        rewards: summaryRes.total_rewards,
        savings: summaryRes.total_savings,
        transactionCount: summaryRes.transaction_count,
      });
      setPage(p);
    } catch (err: unknown) {
      setLedgerTotals(null);
      setError(err instanceof Error ? err.message : "Failed to load transactions.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchPage(1);
  }, [fetchPage]);

  async function handleExport(format: "csv" | "xlsx") {
    setExporting(true);
    try {
      await exportTransactions(format);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Export failed.");
    } finally {
      setExporting(false);
    }
  }

  async function handleAddTransaction(e: React.FormEvent) {
    e.preventDefault();
    const numericAmount = Number(newAmount);
    if (!newMerchant.trim() || !Number.isFinite(numericAmount) || numericAmount <= 0) {
      setError("Enter a valid merchant and amount.");
      return;
    }
    setAddLoading(true);
    setError(null);
    try {
      await createTransaction({
        merchant: newMerchant.trim(),
        amount: numericAmount,
        category: newCategory,
        source_flow: "manual",
      });
      setNewMerchant("");
      setNewAmount("");
      setNewCategory("other");
      setShowAddForm(false);
      void fetchPage(1);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to log transaction.");
    } finally {
      setAddLoading(false);
    }
  }

  const totalPages = data ? Math.ceil(data.total / PAGE_SIZE) : 0;

  const summaryRewards =
    ledgerTotals?.rewards ?? data?.total_rewards ?? 0;
  const summarySavings =
    ledgerTotals?.savings ?? data?.total_savings ?? 0;
  const summaryTxnCount =
    ledgerTotals?.transactionCount ?? data?.total ?? 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 dark:text-white">
            Transaction History
          </h1>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
            {data && data.total > 0
              ? `${data.total} transaction${data.total !== 1 ? "s" : ""} logged`
              : "Your transaction ledger"}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowAddForm((v) => !v)}
            className="px-3 py-2 text-sm font-medium rounded-lg bg-primary text-white hover:bg-primary/90 transition-colors cursor-pointer"
          >
            Log Transaction
          </button>
          {data && data.total > 0 && (
            <>
              <button
                onClick={() => handleExport("csv")}
                disabled={exporting}
                className="px-3 py-2 text-sm font-medium rounded-lg border border-border bg-card text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors disabled:opacity-50"
              >
                {exporting ? "Exporting..." : "Export CSV"}
              </button>
              <button
                onClick={() => handleExport("xlsx")}
                disabled={exporting}
                className="px-3 py-2 text-sm font-medium rounded-lg border border-border bg-card text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors disabled:opacity-50"
              >
                Export XLSX
              </button>
            </>
          )}
        </div>
      </div>

      {/* Add transaction form */}
      {showAddForm && (
        <div className="rounded-xl bg-card border border-border p-4">
          <form onSubmit={handleAddTransaction} className="grid grid-cols-1 sm:grid-cols-4 gap-3 items-end">
            <div>
              <label className="block text-xs font-medium text-slate-600 dark:text-slate-300 mb-1">Merchant</label>
              <input
                type="text"
                value={newMerchant}
                onChange={(e) => setNewMerchant(e.target.value)}
                placeholder="e.g., Whole Foods"
                className="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-slate-900 dark:text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-primary/50"
                required
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-600 dark:text-slate-300 mb-1">Amount</label>
              <input
                type="number"
                min="0.01"
                step="0.01"
                value={newAmount}
                onChange={(e) => setNewAmount(e.target.value)}
                placeholder="0.00"
                className="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-slate-900 dark:text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-primary/50"
                required
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-600 dark:text-slate-300 mb-1">Category</label>
              <select
                value={newCategory}
                onChange={(e) => setNewCategory(e.target.value)}
                className="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-primary/50"
              >
                {Object.keys(CATEGORY_COLORS).map((cat) => (
                  <option key={cat} value={cat}>{cat.replace(/_/g, " ")}</option>
                ))}
              </select>
            </div>
            <div className="flex gap-2">
              <button
                type="submit"
                disabled={addLoading}
                className="px-4 py-2 text-sm font-medium rounded-lg bg-primary text-white hover:bg-primary/90 transition-colors disabled:opacity-50 cursor-pointer"
              >
                {addLoading ? "Saving..." : "Save"}
              </button>
              <button
                type="button"
                onClick={() => setShowAddForm(false)}
                className="px-3 py-2 text-sm font-medium rounded-lg border border-border bg-card text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors cursor-pointer"
              >
                Cancel
              </button>
            </div>
          </form>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 p-4 text-sm text-red-700 dark:text-red-300">
          {error}
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="flex items-center justify-center py-16">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
        </div>
      )}

      {/* Empty state */}
      {!loading && data && data.total === 0 && (
        <div className="rounded-xl bg-card border border-border p-12 text-center">
          <div className="mx-auto w-16 h-16 rounded-full bg-slate-100 dark:bg-slate-800 flex items-center justify-center mb-4">
            <svg className="w-8 h-8 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h3.75M9 15h3.75M9 18h3.75m3 .75H18a2.25 2.25 0 002.25-2.25V6.108c0-1.135-.845-2.098-1.976-2.192a48.424 48.424 0 00-1.123-.08m-5.801 0c-.065.21-.1.433-.1.664 0 .414.336.75.75.75h4.5a.75.75 0 00.75-.75 2.25 2.25 0 00-.1-.664m-5.8 0A2.251 2.251 0 0113.5 2.25H15c1.012 0 1.867.668 2.15 1.586m-5.8 0c-.376.023-.75.05-1.124.08C9.095 4.01 8.25 4.973 8.25 6.108V8.25m0 0H4.875c-.621 0-1.125.504-1.125 1.125v11.25c0 .621.504 1.125 1.125 1.125h9.75c.621 0 1.125-.504 1.125-1.125V9.375c0-.621-.504-1.125-1.125-1.125H8.25z" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-2">
            No transactions yet
          </h3>
          <p className="text-sm text-slate-500 dark:text-slate-400 mb-6 max-w-sm mx-auto">
            Start logging transactions to track your spending and see which cards earn you the most rewards.
          </p>
          <button
            onClick={() => setShowAddForm(true)}
            className="inline-flex items-center px-4 py-2 rounded-lg bg-primary text-white text-sm font-medium hover:bg-primary/90 transition-colors cursor-pointer"
          >
            Log your first transaction
          </button>
        </div>
      )}

      {/* Transaction list */}
      {!loading && data && data.total > 0 && (
        <>
          <div className="rounded-xl border-2 border-primary/25 bg-gradient-to-br from-primary/10 via-primary/5 to-transparent dark:from-primary/15 dark:via-primary/10 dark:to-transparent dark:border-primary/35 px-4 py-4 sm:px-6 shadow-sm">
            <p className="text-sm sm:text-base font-medium text-slate-800 dark:text-slate-100 flex flex-col sm:flex-row sm:flex-wrap sm:items-center sm:justify-center gap-1 sm:gap-x-4 sm:gap-y-1 text-center">
              <span>
                Total Rewards:{" "}
                <span className="font-mono tabular-nums text-green-700 dark:text-green-400">
                  ${formatMoney(summaryRewards)}
                </span>
              </span>
              <span className="hidden sm:inline text-slate-400 dark:text-slate-500 font-normal" aria-hidden>
                |
              </span>
              <span>
                Total Savings:{" "}
                <span className="font-mono tabular-nums text-emerald-700 dark:text-emerald-400">
                  ${formatMoney(summarySavings)}
                </span>
              </span>
              <span className="hidden sm:inline text-slate-400 dark:text-slate-500 font-normal" aria-hidden>
                |
              </span>
              <span>
                Transactions:{" "}
                <span className="font-mono tabular-nums text-slate-900 dark:text-white">{summaryTxnCount}</span>
              </span>
            </p>
          </div>

          {/* Desktop table */}
          <div className="hidden md:block rounded-xl bg-card border border-border overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border bg-slate-50 dark:bg-slate-800/50">
                  <th className="text-left px-4 py-3 font-semibold text-slate-600 dark:text-slate-300">Merchant</th>
                  <th className="text-left px-4 py-3 font-semibold text-slate-600 dark:text-slate-300">Category</th>
                  <th className="text-right px-4 py-3 font-semibold text-slate-600 dark:text-slate-300">Amount</th>
                  <th className="text-left px-4 py-3 font-semibold text-slate-600 dark:text-slate-300">Card Used</th>
                  <th className="text-right px-4 py-3 font-semibold text-slate-600 dark:text-slate-300">Reward</th>
                  <th className="text-right px-4 py-3 font-semibold text-slate-600 dark:text-slate-300">Savings</th>
                  <th className="text-left px-4 py-3 font-semibold text-slate-600 dark:text-slate-300">Date</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {data.transactions.map((txn) => (
                  <tr key={txn.id} className="hover:bg-slate-50 dark:hover:bg-slate-800/30 transition-colors">
                    <td className="px-4 py-3">
                      <div className="font-medium text-slate-900 dark:text-white">{txn.merchant}</div>
                      <div className="text-xs text-slate-400 mt-0.5 capitalize">{txn.source_flow}</div>
                    </td>
                    <td className="px-4 py-3">{categoryBadge(txn.category)}</td>
                    <td className="px-4 py-3 text-right font-mono font-medium text-slate-900 dark:text-white">
                      ${txn.amount.toFixed(2)}
                    </td>
                    <td className="px-4 py-3">
                      {txn.chosen_card_name ? (
                        <div className="flex items-center gap-2">
                          <img
                            src={getCardImage(txn.chosen_card_id ?? undefined)}
                            alt=""
                            className="w-6 h-4 rounded-sm object-cover"
                          />
                          <span className="text-slate-700 dark:text-slate-300 text-xs">
                            {txn.chosen_card_name}
                          </span>
                        </div>
                      ) : (
                        <span className="text-slate-400 text-xs">-</span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-right font-mono text-green-600 dark:text-green-400">
                      +${txn.reward_earned.toFixed(2)}
                    </td>
                    <td className="px-4 py-3 text-right font-mono text-emerald-600 dark:text-emerald-400">
                      ${txn.estimated_savings.toFixed(2)}
                    </td>
                    <td className="px-4 py-3 text-slate-500 dark:text-slate-400 whitespace-nowrap">
                      <div>{formatDate(txn.timestamp)}</div>
                      <div className="text-xs text-slate-400">{formatTime(txn.timestamp)}</div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Mobile cards */}
          <div className="md:hidden space-y-3">
            {data.transactions.map((txn) => (
              <div key={txn.id} className="rounded-xl bg-card border border-border p-4">
                <div className="flex items-start justify-between mb-2">
                  <div>
                    <div className="font-medium text-slate-900 dark:text-white">{txn.merchant}</div>
                    <div className="text-xs text-slate-400 mt-0.5">{formatDate(txn.timestamp)}</div>
                  </div>
                  <div className="text-right">
                    <div className="font-mono font-semibold text-slate-900 dark:text-white">
                      ${txn.amount.toFixed(2)}
                    </div>
                    <div className="text-xs font-mono text-green-600 dark:text-green-400">
                      +${txn.reward_earned.toFixed(2)}
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-2 flex-wrap">
                  {categoryBadge(txn.category)}
                  {txn.chosen_card_name && (
                    <span className="text-xs text-slate-500 dark:text-slate-400">
                      via {txn.chosen_card_name}
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-between pt-2">
              <p className="text-sm text-slate-500 dark:text-slate-400">
                Page {page} of {totalPages}
              </p>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => fetchPage(page - 1)}
                  disabled={page <= 1}
                  className="px-3 py-1.5 text-sm rounded-lg border border-border bg-card text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  Previous
                </button>
                <button
                  onClick={() => fetchPage(page + 1)}
                  disabled={!data.has_next}
                  className="px-3 py-1.5 text-sm rounded-lg border border-border bg-card text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  Next
                </button>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}