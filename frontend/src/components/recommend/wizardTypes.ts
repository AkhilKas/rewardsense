import type { CategoryKey } from "./constants";
import { INITIAL_SPENDING } from "./constants";

export type WizardStep = 1 | 2 | 3;

export interface WizardFormState {
  spending: Record<CategoryKey, number>;
  selectedRewards: string[];
  incomeRange: string;
  currentCards: string[];
}

export function createInitialWizardState(): WizardFormState {
  return {
    spending: { ...INITIAL_SPENDING },
    selectedRewards: [],
    incomeRange: "",
    currentCards: [],
  };
}
