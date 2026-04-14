import Card from "../Card";
import Button from "../Button";
import { INCOME_RANGES, REWARD_TYPES, SPENDING_CATEGORIES } from "./constants";
import type { WizardFormState, WizardStep } from "./wizardTypes";
import type { CardCatalogItem } from "../../types";

interface ReviewStepProps {
  state: WizardFormState;
  totalSpend: number;
  catalog: CardCatalogItem[];
  onEdit: (step: WizardStep) => void;
  onSubmit: () => void;
  loading: boolean;
  apiError: string;
}

function labelMap(
  options: { value: string; label: string }[],
  value: string,
): string {
  return options.find((o) => o.value === value)?.label ?? value;
}

export default function ReviewStep({
  state,
  totalSpend,
  catalog,
  onEdit,
  onSubmit,
  loading,
  apiError,
}: ReviewStepProps) {
  const spendingLines = SPENDING_CATEGORIES.filter(
    (c) => state.spending[c.key] > 0,
  ).map((c) => ({
    label: c.label,
    amount: state.spending[c.key],
  }));

  const rewardLabels = state.selectedRewards.map((v) =>
    labelMap(REWARD_TYPES, v),
  );

  const catalogById = new Map(catalog.map((c) => [c.card_id, c]));
  const cardLabels = state.currentCards.map(
    (id) => catalogById.get(id)?.card_name ?? id,
  );

  return (
    <div className="space-y-6">
      <Card>
        <h2 className="text-lg font-semibold text-secondary mb-1">
          Review your profile
        </h2>
        <p className="text-sm text-slate-500 dark:text-slate-400 mb-6">
          Double-check your details, then get your recommendations.
        </p>

        <div className="space-y-6">
          <section className="rounded-lg border border-border bg-surface/50 p-4">
            <div className="flex items-start justify-between gap-2 mb-3">
              <h3 className="text-sm font-semibold text-secondary">
                Monthly spending
              </h3>
              <button
                type="button"
                onClick={() => onEdit(1)}
                className="text-sm font-medium text-primary hover:underline shrink-0 cursor-pointer"
              >
                Edit
              </button>
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400 mb-2">
              Total:{" "}
              <span className="font-semibold text-secondary">
                ${totalSpend.toLocaleString()}
              </span>
              /mo
            </p>
            <ul className="space-y-1.5 text-sm text-secondary">
              {spendingLines.length === 0 ? (
                <li className="text-slate-500 dark:text-slate-400">—</li>
              ) : (
                spendingLines.map((row) => (
                  <li
                    key={row.label}
                    className="flex justify-between gap-4 border-b border-border/60 pb-1 last:border-0"
                  >
                    <span>{row.label}</span>
                    <span className="tabular-nums">
                      ${row.amount.toLocaleString()}
                    </span>
                  </li>
                ))
              )}
            </ul>
          </section>

          <section className="rounded-lg border border-border bg-surface/50 p-4">
            <div className="flex items-start justify-between gap-2 mb-3">
              <h3 className="text-sm font-semibold text-secondary">
                Preferences
              </h3>
              <button
                type="button"
                onClick={() => onEdit(2)}
                className="text-sm font-medium text-primary hover:underline shrink-0 cursor-pointer"
              >
                Edit
              </button>
            </div>
            <div className="space-y-2 text-sm text-secondary">
              <p>
                <span className="text-slate-500 dark:text-slate-400">
                  Reward types:{" "}
                </span>
                {rewardLabels.length ? rewardLabels.join(", ") : "—"}
              </p>
              <p>
                <span className="text-slate-500 dark:text-slate-400">
                  Income:{" "}
                </span>
                {state.incomeRange
                  ? labelMap(INCOME_RANGES, state.incomeRange)
                  : "—"}
              </p>
              <p>
                <span className="text-slate-500 dark:text-slate-400">
                  Cards you hold:{" "}
                </span>
                {cardLabels.length ? cardLabels.join(", ") : "None selected"}
              </p>
            </div>
          </section>
        </div>

        {apiError && (
          <div className="mt-6 rounded-md bg-red-50 dark:bg-red-900/20 border border-danger/30 p-4">
            <p className="text-sm text-danger">{apiError}</p>
          </div>
        )}

        <div className="mt-8">
          <Button
            type="button"
            size="lg"
            loading={loading}
            disabled={loading}
            className="w-full"
            onClick={onSubmit}
          >
            Get my recommendations
          </Button>
        </div>
      </Card>
    </div>
  );
}
