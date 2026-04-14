import Card from "../Card";
import CardMultiSelectCombobox from "../CardMultiSelectCombobox";
import Select from "../Select";
import type { CardCatalogItem } from "../../types";
import { INCOME_RANGES, REWARD_TYPES } from "./constants";

interface PreferencesStepProps {
  catalog: CardCatalogItem[];
  catalogLoading?: boolean;
  selectedRewards: string[];
  incomeRange: string;
  currentCards: string[];
  rewardsError?: string;
  incomeError?: string;
  onRewardsChange: (v: string[]) => void;
  onIncomeChange: (v: string) => void;
  onCardsChange: (v: string[]) => void;
}

export default function PreferencesStep({
  catalog,
  catalogLoading,
  selectedRewards,
  incomeRange,
  currentCards,
  rewardsError,
  incomeError,
  onRewardsChange,
  onIncomeChange,
  onCardsChange,
}: PreferencesStepProps) {
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
