import { useState, useEffect, useCallback } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import Card from "../components/Card";
import Badge from "../components/Badge";
import LoadingSpinner from "../components/LoadingSpinner";
import { getMonitoringData } from "../api/client";
import type { MonitoringData, RetrainEvent } from "../types";

const REFRESH_INTERVAL_MS = 60_000;

const DRIFT_THRESHOLD = 0.1;

function formatTimestamp(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function timeSince(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60_000);
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

const statusBadge: Record<RetrainEvent["status"], "success" | "danger" | "warning"> = {
  success: "success",
  failed: "danger",
  in_progress: "warning",
};

/* ------------------------------------------------------------------ */
/*  Sub-sections                                                       */
/* ------------------------------------------------------------------ */

function ModelInfoCard({ data }: { data: MonitoringData }) {
  return (
    <Card>
      <h2 className="text-lg font-semibold text-secondary mb-4">
        Deployment Status
      </h2>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <Stat label="Model Version" value={data.model_version} />
        <Stat
          label="Last Deployed"
          value={timeSince(data.last_deployment_time)}
        />
        <Stat
          label="Drift Detected"
          value={data.drift_check.detected ? "Yes" : "No"}
          variant={data.drift_check.detected ? "danger" : "success"}
        />
        <Stat
          label="Last Drift Check"
          value={timeSince(data.drift_check.timestamp)}
        />
      </div>
    </Card>
  );
}

function Stat({
  label,
  value,
  variant,
}: {
  label: string;
  value: string;
  variant?: "success" | "danger";
}) {
  let valueClass = "text-xl font-bold text-secondary";
  if (variant === "success") valueClass = "text-xl font-bold text-accent";
  if (variant === "danger") valueClass = "text-xl font-bold text-danger";

  return (
    <div>
      <p className="text-xs text-slate-500 dark:text-slate-400 mb-1">
        {label}
      </p>
      <p className={valueClass}>{value}</p>
    </div>
  );
}

function DriftHeatmap({ data }: { data: MonitoringData }) {
  const features = Object.entries(data.drift_check.feature_drift)
    .map(([name, score]) => ({
      name: name.replace(/_/g, " "),
      score,
      drifted: score > DRIFT_THRESHOLD,
    }))
    .sort((a, b) => b.score - a.score);

  return (
    <Card>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-secondary">
          Feature Drift
        </h2>
        <Badge variant={data.drift_check.detected ? "danger" : "success"}>
          {data.drift_check.detected ? "Drift Detected" : "No Drift"}
        </Badge>
      </div>

      <div className="h-64 outline-none" style={{ outline: "none" }}>
        <ResponsiveContainer width="100%" height="100%" className="outline-none">
          <BarChart data={features} layout="vertical" margin={{ left: 20 }} style={{ outline: "none" }}>
            <XAxis
              type="number"
              domain={[0, 0.2]}
              tickFormatter={(v: number) => v.toFixed(2)}
              tick={{ fontSize: 11, fill: "var(--color-secondary)" }}
            />
            <YAxis
              type="category"
              dataKey="name"
              width={110}
              tick={{ fontSize: 11, fill: "var(--color-secondary)" }}
            />
            <Tooltip
              formatter={(v) => [Number(v).toFixed(4), "Drift Score"]}
              cursor={{ fill: "var(--color-border)", opacity: 0.3 }}
              wrapperStyle={{ outline: "none" }}
              contentStyle={{
                backgroundColor: "var(--color-card)",
                border: "none",
                borderRadius: "0.5rem",
                boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
                fontSize: 12,
                color: "var(--color-secondary)",
              }}
              labelStyle={{ color: "var(--color-secondary)" }}
              itemStyle={{ color: "var(--color-secondary)" }}
            />
            <Bar dataKey="score" radius={[0, 4, 4, 0]} barSize={16}>
              {features.map((f, i) => (
                <Cell
                  key={i}
                  fill={f.drifted ? "var(--color-danger)" : "var(--color-primary)"}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <p className="mt-2 text-xs text-slate-400 dark:text-slate-500">
        Red bars exceed the {DRIFT_THRESHOLD} drift threshold.
      </p>
    </Card>
  );
}

function ServingMetricsCard({ data }: { data: MonitoringData }) {
  const m = data.serving_metrics;

  const metrics = [
    {
      label: "Requests (24h)",
      value: m.request_count.toLocaleString(),
      sub: null,
    },
    {
      label: "Avg Latency",
      value: `${m.avg_latency_ms.toLocaleString()}ms`,
      sub: `p95: ${m.p95_latency_ms.toLocaleString()}ms`,
    },
    {
      label: "Error Rate",
      value: `${(m.error_rate * 100).toFixed(2)}%`,
      sub: null,
    },
  ];

  const latencyData = [
    { name: "Avg", value: m.avg_latency_ms },
    { name: "p95", value: m.p95_latency_ms },
  ];

  return (
    <Card>
      <h2 className="text-lg font-semibold text-secondary mb-4">
        Serving Metrics
      </h2>

      <div className="grid grid-cols-3 gap-4 mb-6">
        {metrics.map((s) => (
          <div key={s.label}>
            <p className="text-xs text-slate-500 dark:text-slate-400 mb-1">
              {s.label}
            </p>
            <p className="text-xl font-bold text-secondary">{s.value}</p>
            {s.sub && (
              <p className="text-xs text-slate-400 dark:text-slate-500">
                {s.sub}
              </p>
            )}
          </div>
        ))}
      </div>

      <div className="h-32 outline-none" style={{ outline: "none" }}>
        <ResponsiveContainer width="100%" height="100%" className="outline-none">
          <BarChart data={latencyData} style={{ outline: "none" }}>
            <XAxis
              dataKey="name"
              tick={{ fontSize: 11, fill: "var(--color-secondary)" }}
            />
            <YAxis
              tick={{ fontSize: 11, fill: "var(--color-secondary)" }}
              tickFormatter={(v: number) => `${v}ms`}
            />
            <Tooltip
              formatter={(v) => [`${v}ms`, "Latency"]}
              cursor={{ fill: "var(--color-border)", opacity: 0.3 }}
              wrapperStyle={{ outline: "none" }}
              contentStyle={{
                backgroundColor: "var(--color-card)",
                border: "none",
                borderRadius: "0.5rem",
                boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
                fontSize: 12,
                color: "var(--color-secondary)",
              }}
              labelStyle={{ color: "var(--color-secondary)" }}
              itemStyle={{ color: "var(--color-secondary)" }}
            />
            <Bar dataKey="value" radius={[4, 4, 0, 0]} barSize={40}>
              <Cell fill="var(--color-primary)" />
              <Cell fill="var(--color-warning)" />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}

function RetrainHistoryCard({ data }: { data: MonitoringData }) {
  return (
    <Card>
      <h2 className="text-lg font-semibold text-secondary mb-4">
        Retrain History
      </h2>

      <div className="space-y-3">
        {data.retrain_history.map((event, i) => (
          <div
            key={i}
            className="flex items-start gap-3 relative pl-6"
          >
            {/* Timeline dot + line */}
            <div className="absolute left-0 top-1 flex flex-col items-center">
              <span
                className={`w-3 h-3 rounded-full shrink-0 ${
                  event.status === "success"
                    ? "bg-accent"
                    : event.status === "failed"
                      ? "bg-danger"
                      : "bg-warning"
                }`}
              />
              {i < data.retrain_history.length - 1 && (
                <span className="w-px flex-1 bg-border mt-1 min-h-[24px]" />
              )}
            </div>

            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-sm font-medium text-secondary">
                  {event.model_version}
                </span>
                <Badge variant={statusBadge[event.status]}>
                  {event.status}
                </Badge>
              </div>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                {event.trigger_reason}
              </p>
              <p className="text-xs text-slate-400 dark:text-slate-500">
                {formatTimestamp(event.timestamp)}
              </p>
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
}

/* ------------------------------------------------------------------ */
/*  Main Dashboard Page                                                */
/* ------------------------------------------------------------------ */

export default function DashboardPage() {
  const [data, setData] = useState<MonitoringData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const result = await getMonitoringData();
      setData(result);
      setError("");
      setLastRefresh(new Date());
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to load monitoring data.",
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, REFRESH_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [fetchData]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="text-center">
          <LoadingSpinner size="lg" className="mx-auto mb-4" />
          <p className="text-slate-500 dark:text-slate-400">
            Loading monitoring data...
          </p>
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <Card className="text-center">
        <p className="text-danger mb-2">{error || "No data available."}</p>
        <button
          onClick={fetchData}
          className="text-sm text-primary hover:text-primary-dark font-medium cursor-pointer"
        >
          Retry
        </button>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-secondary">
          Monitoring Dashboard
        </h1>
        {lastRefresh && (
          <p className="text-xs text-slate-400 dark:text-slate-500">
            Updated {lastRefresh.toLocaleTimeString()} &middot; refreshes every
            60s
          </p>
        )}
      </div>

      <ModelInfoCard data={data} />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <DriftHeatmap data={data} />
        <ServingMetricsCard data={data} />
      </div>

      <RetrainHistoryCard data={data} />
    </div>
  );
}
