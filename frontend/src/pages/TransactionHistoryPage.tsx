import { useEffect, useState } from "react";
import {
  exportTransactions,
  getTransactions,
} from "../api/client";
import type { TransactionsResponse } from "../types";
import Card from "../components/Card";
import Button from "../components/Button";

export default function TransactionHistoryPage() {
  const [page, setPage] = useState(1);
  const [pageSize] = useState(10);
  const [data, setData] = useState<TransactionsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [exportLink, setExportLink] = useState("");
  const [exporting, setExporting] = useState<"" | "csv" | "xlsx">("");
  const [exportMessage, setExportMessage] = useState("");

  useEffect(() => {
    async function load() {
      setLoading(true);
      setError("");
      try {
        const res = await getTransactions(page, pageSize);
        setData(res);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load transactions");
      } finally {
        setLoading(false);
      }
    }
    void load();
  }, [page, pageSize]);

  async function handleExport(format: "csv" | "xlsx") {
    setExporting(format);
    setError("");
    setExportMessage("");
    try {
      const res = await exportTransactions(format);
      setExportLink(res.download_url);
      if (
        res.download_url.startsWith("/") &&
        !res.download_url.startsWith("//")
      ) {
        setExportMessage(
          "Mock export generated. Download file endpoints are not wired yet in this phase.",
        );
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Export failed");
    } finally {
      setExporting("");
    }
  }

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h1 className="text-3xl font-bold text-secondary">Transaction History</h1>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
            Review logged transactions and export your history.
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            variant="secondary"
            onClick={() => void handleExport("csv")}
            loading={exporting === "csv"}
          >
            Export CSV
          </Button>
          <Button
            variant="secondary"
            onClick={() => void handleExport("xlsx")}
            loading={exporting === "xlsx"}
          >
            Export XLSX
          </Button>
        </div>
      </div>

      {exportLink && (
        <Card>
          <div className="space-y-1">
            {exportMessage ? (
              <>
                <p className="text-sm text-accent">{exportMessage}</p>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  Export path: <code>{exportLink}</code>
                </p>
              </>
            ) : (
              <p className="text-sm text-accent">
                Export generated:{" "}
                <a
                  href={exportLink}
                  className="underline"
                  target="_blank"
                  rel="noreferrer"
                >
                  {exportLink}
                </a>
              </p>
            )}
          </div>
        </Card>
      )}

      {error && (
        <Card>
          <p className="text-danger text-sm">{error}</p>
        </Card>
      )}

      <Card>
        {loading ? (
          <div className="flex items-center justify-center py-10">
            <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
          </div>
        ) : data && data.items.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-slate-500 dark:text-slate-400 border-b border-border">
                  <th className="py-2 pr-3">Timestamp</th>
                  <th className="py-2 pr-3">Merchant</th>
                  <th className="py-2 pr-3">Category</th>
                  <th className="py-2 pr-3">Amount</th>
                  <th className="py-2 pr-3">Reward</th>
                  <th className="py-2 pr-3">Savings</th>
                </tr>
              </thead>
              <tbody>
                {data.items.map((tx) => (
                  <tr key={tx.id} className="border-b border-border/60">
                    <td className="py-2 pr-3 text-secondary">
                      {new Date(tx.timestamp).toLocaleString()}
                    </td>
                    <td className="py-2 pr-3 text-secondary">{tx.merchant}</td>
                    <td className="py-2 pr-3 text-slate-500 dark:text-slate-400">
                      {tx.category}
                    </td>
                    <td className="py-2 pr-3 text-secondary">${tx.amount.toFixed(2)}</td>
                    <td className="py-2 pr-3 text-secondary">${tx.reward_earned.toFixed(2)}</td>
                    <td className="py-2 pr-3 text-secondary">${tx.estimated_savings.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-sm text-slate-500 dark:text-slate-400 py-6">
            No transactions found yet.
          </p>
        )}
      </Card>

      {data && data.total_pages > 1 && (
        <div className="flex items-center justify-between">
          <p className="text-sm text-slate-500 dark:text-slate-400">
            Page {data.page} of {data.total_pages}
          </p>
          <div className="flex gap-2">
            <Button
              variant="secondary"
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={data.page <= 1}
            >
              Previous
            </Button>
            <Button
              variant="secondary"
              onClick={() => setPage((p) => Math.min(data.total_pages, p + 1))}
              disabled={data.page >= data.total_pages}
            >
              Next
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
