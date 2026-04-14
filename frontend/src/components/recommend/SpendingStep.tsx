import { useEffect, useState } from "react";
import Card from "../Card";
import SliderInput from "../SliderInput";
import {
  ARCHETYPE_META,
  SPENDING_CATEGORIES,
  type CategoryKey,
  type FrontendArchetype,
} from "./constants";

interface SpendingStepProps {
  spending: Record<CategoryKey, number>;
  totalSpend: number;
  detectedArchetype: FrontendArchetype;
  error?: string;
  onChange: (key: CategoryKey, value: number) => void;
}

export default function SpendingStep({
  spending,
  totalSpend,
  detectedArchetype,
  error,
  onChange,
}: SpendingStepProps) {
  const [visibleArchetype, setVisibleArchetype] =
    useState<FrontendArchetype>(detectedArchetype);
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    if (detectedArchetype === visibleArchetype) return;
    setIsVisible(false);
    const swapTimer = window.setTimeout(() => {
      setVisibleArchetype(detectedArchetype);
      setIsVisible(true);
    }, 140);
    return () => window.clearTimeout(swapTimer);
  }, [detectedArchetype, visibleArchetype]);

  const archetypeMeta = ARCHETYPE_META[visibleArchetype];

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
      <div className="mt-3">
        <div
          className={`inline-flex items-center gap-2 rounded-full border border-primary/25 bg-primary/10 dark:bg-primary/15 px-3 py-1.5 text-xs font-medium text-secondary transition-all duration-150 ${
            isVisible ? "opacity-100 translate-y-0" : "opacity-0 -translate-y-0.5"
          }`}
          aria-live="polite"
        >
          <span>{archetypeMeta.emoji}</span>
          <span>
            Your profile: <span className="font-semibold">{archetypeMeta.label}</span>
          </span>
        </div>
      </div>

      {error && (
        <p className="mt-2 text-xs text-danger" role="alert">
          {error}
        </p>
      )}
    </Card>
  );
}
