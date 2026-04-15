import Card from "../Card";
import CardMultiSelectCombobox from "../CardMultiSelectCombobox";
import Select from "../Select";
import type { CardCatalogItem } from "../../types";
import {
  ARCHETYPE_META,
  INCOME_RANGES,
  REWARD_TYPES,
  type FrontendArchetype,
} from "./constants";

interface PreferencesStepProps {
  catalog: CardCatalogItem[];
  catalogLoading?: boolean;
  selectedRewards: string[];
  incomeRange: string;
  currentCards: string[];
  selectedArchetype: FrontendArchetype;
  rewardsError?: string;
  incomeError?: string;
  onRewardsChange: (v: string[]) => void;
  onIncomeChange: (v: string) => void;
  onCardsChange: (v: string[]) => void;
  onArchetypeChange: (value: FrontendArchetype) => void;
}

export default function PreferencesStep({
  catalog,
  catalogLoading,
  selectedRewards,
  incomeRange,
  currentCards,
  selectedArchetype,
  rewardsError,
  incomeError,
  onRewardsChange,
  onIncomeChange,
  onCardsChange,
  onArchetypeChange,
}: PreferencesStepProps) {
  const archetypeOptions = Object.entries(ARCHETYPE_META) as Array<
    [FrontendArchetype, (typeof ARCHETYPE_META)[FrontendArchetype]]
  >;

  return (
    <div className="space-y-6">
      <Card>
        <h2 className="text-lg font-semibold text-secondary mb-1">
          What matters to you?
        </h2>
        <p className="text-sm text-slate-500 dark:text-slate-400 mb-6">
          Tell us how you like to earn and a bit about your finances.
        </p>

        <div className="space-y-6">
          <Select
            label="Preferred reward types"
            options={REWARD_TYPES}
            value={selectedRewards}
            onChange={onRewardsChange}
            multiple
            error={rewardsError}
          />

          <Select
            label="Annual income range"
            options={INCOME_RANGES}
            value={incomeRange}
            onChange={onIncomeChange}
            placeholder="Select your income range"
            error={incomeError}
          />

          <div>
            <h3 className="text-sm font-medium text-secondary mb-1">
              Your Spending Style
            </h3>
            <p className="text-sm text-slate-500 dark:text-slate-400 mb-3">
              We pre-selected a style from your sliders. You can keep it or
              choose a better fit.
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {archetypeOptions.map(([key, meta]) => {
                const selected = key === selectedArchetype;
                return (
                  <button
                    key={key}
                    type="button"
                    onClick={() => onArchetypeChange(key)}
                    className={`text-left rounded-lg border p-3 transition-colors cursor-pointer ${
                      selected
                        ? "border-primary bg-primary/10 dark:bg-primary/20 ring-1 ring-primary/30"
                        : "border-border bg-surface hover:border-primary/40"
                    }`}
                    aria-pressed={selected}
                  >
                    <p className="text-sm font-semibold text-secondary">
                      {meta.emoji} {meta.label}
                    </p>
                    <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
                      {meta.description}
                    </p>
                  </button>
                );
              })}
            </div>
          </div>

          <CardMultiSelectCombobox
            label="Current cards you hold"
            optional
            description="Search and add each card you already have. Selected cards appear as chips below."
            catalog={catalog}
            selectedIds={currentCards}
            onChange={onCardsChange}
            disabled={!!catalogLoading}
            dropdownStrategy="fixed"
          />
        </div>
      </Card>
    </div>
  );
}
