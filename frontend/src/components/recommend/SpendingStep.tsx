import Card from "../Card";
import SliderInput from "../SliderInput";
import { SPENDING_CATEGORIES, type CategoryKey } from "./constants";

interface SpendingStepProps {
  spending: Record<CategoryKey, number>;
  totalSpend: number;
  error?: string;
  onChange: (key: CategoryKey, value: number) => void;
}

export default function SpendingStep({
  spending,
  totalSpend,
  error,
  onChange,
}: SpendingStepProps) {
  return (
    <Card>
      <h2 className="text-lg font-semibold text-secondary mb-1">
        How do you spend?
      </h2>
      <p className="text-sm text-slate-500 dark:text-slate-400 mb-6">
        Adjust each category to match your typical monthly spend.
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-8 gap-y-5">
        {SPENDING_CATEGORIES.map((cat) => (
          <SliderInput
            key={cat.key}
            label={cat.label}
            value={spending[cat.key]}
            onChange={(v) => onChange(cat.key, v)}
            max={cat.max}
            step={50}
          />
        ))}
      </div>

      <div className="mt-6 pt-4 border-t border-border flex items-center justify-between">
        <span className="text-sm font-medium text-slate-600 dark:text-slate-400">
          Total monthly spend
        </span>
        <span className="text-lg font-bold text-primary">
          ${totalSpend.toLocaleString()}
        </span>
      </div>

      {error && (
        <p className="mt-2 text-xs text-danger" role="alert">
          {error}
        </p>
      )}
    </Card>
  );
}
