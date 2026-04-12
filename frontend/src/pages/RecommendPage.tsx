import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import Card from "../components/Card";
import Button from "../components/Button";
import SliderInput from "../components/SliderInput";
import Select from "../components/Select";
import LoadingSpinner from "../components/LoadingSpinner";
import { getCardCatalog, recommendPortfolio } from "../api/client";
import type { SpendingCategories } from "../types";
import { mapPortfolioToPredictionResponse } from "../viewmodels/viewMappers";
import { useAuth } from "../context/AuthContext";

const SPENDING_CATEGORIES = [
  { key: "groceries" as const, label: "Groceries", max: 3000 },
  { key: "dining" as const, label: "Dining", max: 3000 },
  { key: "travel" as const, label: "Travel", max: 3000 },
  { key: "gas" as const, label: "Gas", max: 1000 },
  { key: "online_shopping" as const, label: "Online Shopping", max: 3000 },
  { key: "entertainment" as const, label: "Entertainment", max: 1000 },
  { key: "utilities" as const, label: "Utilities", max: 1000 },
  { key: "other" as const, label: "Other", max: 2000 },
];

const REWARD_TYPES = [
  { value: "cashback", label: "Cashback" },
  { value: "travel_points", label: "Travel Points" },
  { value: "hotel_points", label: "Hotel Points" },
  { value: "airline_miles", label: "Airline Miles" },
];

const INCOME_RANGES = [
  { value: "under_30k", label: "Under $30,000" },
  { value: "30k_50k", label: "$30,000 – $50,000" },
  { value: "50k_75k", label: "$50,000 – $75,000" },
  { value: "75k_100k", label: "$75,000 – $100,000" },
  { value: "over_100k", label: "Over $100,000" },
];

const POPULAR_CARDS = [
  { value: "chase_sapphire_preferred", label: "Chase Sapphire Preferred" },
  { value: "amex_gold", label: "Amex Gold Card" },
  { value: "citi_double_cash", label: "Citi Double Cash" },
  { value: "capital_one_venture_x", label: "Capital One Venture X" },
  { value: "discover_it_cash_back", label: "Discover it Cash Back" },
  { value: "chase_freedom_unlimited", label: "Chase Freedom Unlimited" },
  { value: "capital_one_savor", label: "Capital One Savor" },
  { value: "wells_fargo_autograph", label: "Wells Fargo Autograph" },
  { value: "blue_cash_preferred", label: "Blue Cash Preferred" },
];

type CategoryKey = (typeof SPENDING_CATEGORIES)[number]["key"];

interface FormErrors {
  spending?: string;
  rewards?: string;
  income?: string;
}

const INITIAL_SPENDING: Record<CategoryKey, number> = {
  groceries: 0,
  dining: 0,
  travel: 0,
  gas: 0,
  online_shopping: 0,
  entertainment: 0,
  utilities: 0,
  other: 0,
};

export default function RecommendPage() {
  const navigate = useNavigate();
  const { user } = useAuth();

  const [spending, setSpending] =
    useState<Record<CategoryKey, number>>(INITIAL_SPENDING);
  const [selectedRewards, setSelectedRewards] = useState<string[]>([]);
  const [incomeRange, setIncomeRange] = useState("");
  const [currentCards, setCurrentCards] = useState<string[]>([]);
  const [errors, setErrors] = useState<FormErrors>({});
  const [loading, setLoading] = useState(false);
  const [apiError, setApiError] = useState("");

  const totalSpend = Object.values(spending).reduce((a, b) => a + b, 0);

  useEffect(() => {
    if (!user?.reward_preference) return;
    setSelectedRewards((prev) =>
      prev.length > 0 ? prev : [user.reward_preference],
    );
  }, [user]);

  function updateSpending(key: CategoryKey, value: number) {
    setSpending((prev) => ({ ...prev, [key]: value }));
    if (errors.spending) setErrors((prev) => ({ ...prev, spending: undefined }));
  }

  function validate(): FormErrors {
    const errs: FormErrors = {};
    if (totalSpend === 0) {
      errs.spending = "Set at least one spending category above $0.";
    }
    if (selectedRewards.length === 0) {
      errs.rewards = "Select at least one preferred reward type.";
    }
    if (!incomeRange) {
      errs.income = "Select your annual income range.";
    }
    return errs;
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setApiError("");

    const errs = validate();
    if (Object.keys(errs).length > 0) {
      setErrors(errs);
      return;
    }
    setErrors({});

    setLoading(true);
    try {
      // Always send every category key so the API receives a full breakdown (not {}).
      const spendingCategories: SpendingCategories = {
        ...INITIAL_SPENDING,
        ...spending,
      };

      const start = performance.now();
      const [portfolioResult, catalog] = await Promise.all([
        recommendPortfolio({
          spending_categories: spendingCategories as Record<string, number>,
          monthly_spend: totalSpend,
          use_full_catalog: true,
        }),
        getCardCatalog(),
      ]);

      const response = mapPortfolioToPredictionResponse({
        portfolio: portfolioResult,
        catalog,
        totalSpend,
        latencyMs: Math.round(performance.now() - start),
      });
      navigate("/results", { state: response });
    } catch (err) {
      setApiError(
        err instanceof Error ? err.message : "Something went wrong. Please try again.",
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
    <div className="max-w-3xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-secondary">
          Find Your Best Cards
        </h1>
        <p className="mt-2 text-slate-600 dark:text-slate-400">
          Tell us about your spending and we'll find the best credit cards for
          you.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-8">
        {/* Spending Categories */}
        <Card>
          <h2 className="text-lg font-semibold text-secondary mb-1">
            Monthly Spending by Category
          </h2>
          <p className="text-sm text-slate-500 dark:text-slate-400 mb-6">
            Drag the sliders to match your typical monthly spend.
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-8 gap-y-5">
            {SPENDING_CATEGORIES.map((cat) => (
              <SliderInput
                key={cat.key}
                label={cat.label}
                value={spending[cat.key]}
                onChange={(v) => updateSpending(cat.key, v)}
                max={cat.max}
                step={50}
              />
            ))}
          </div>

          <div className="mt-6 pt-4 border-t border-border flex items-center justify-between">
            <span className="text-sm font-medium text-slate-600 dark:text-slate-400">
              Total Monthly Spend
            </span>
            <span className="text-lg font-bold text-primary">
              ${totalSpend.toLocaleString()}
            </span>
          </div>

          {errors.spending && (
            <p className="mt-2 text-xs text-danger">{errors.spending}</p>
          )}
        </Card>

        {/* Reward Preferences */}
        <Card>
          <Select
            label="Preferred Reward Types"
            options={REWARD_TYPES}
            value={selectedRewards}
            onChange={(v) => {
              setSelectedRewards(v);
              if (errors.rewards) setErrors((prev) => ({ ...prev, rewards: undefined }));
            }}
            multiple
            error={errors.rewards}
          />
        </Card>

        {/* Income Range */}
        <Card>
          <Select
            label="Annual Income Range"
            options={INCOME_RANGES}
            value={incomeRange}
            onChange={(v) => {
              setIncomeRange(v);
              if (errors.income) setErrors((prev) => ({ ...prev, income: undefined }));
            }}
            placeholder="Select your income range"
            error={errors.income}
          />
        </Card>

        {/* Current Cards (optional) */}
        <Card>
          <Select
            label="Current Cards You Hold"
            options={POPULAR_CARDS}
            value={currentCards}
            onChange={setCurrentCards}
            multiple
            optional
          />
        </Card>

        {/* Error display */}
        {apiError && (
          <div className="rounded-md bg-red-50 dark:bg-red-900/20 border border-danger/30 p-4">
            <p className="text-sm text-danger">{apiError}</p>
          </div>
        )}

        {/* Submit */}
        <Button
          type="submit"
          size="lg"
          loading={loading}
          disabled={loading}
          className="w-full"
        >
          Get Recommendations
        </Button>
      </form>
    </div>

    {/* Loading overlay — outside max-w container so fixed positioning is clean */}
    {loading && (
      <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/40 backdrop-blur-sm">
        <Card className="text-center max-w-xs mx-4">
          <LoadingSpinner size="lg" className="mx-auto mb-4" />
          <p className="font-medium text-secondary">
            Analyzing your profile...
          </p>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
            This usually takes a few seconds
          </p>
        </Card>
      </div>
    )}
    </>
  );
}
