import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import Button from "../components/Button";
import LoadingSpinner from "../components/LoadingSpinner";
import Card from "../components/Card";
import PreferencesStep from "../components/recommend/PreferencesStep";
import ReviewStep from "../components/recommend/ReviewStep";
import SpendingStep from "../components/recommend/SpendingStep";
import StepIndicator from "../components/recommend/StepIndicator";
import {
  ARCHETYPE_META,
  detectArchetype,
  INITIAL_SPENDING,
  type CategoryKey,
  type FrontendArchetype,
} from "../components/recommend/constants";
import {
  createInitialWizardState,
  type WizardFormState,
  type WizardStep,
} from "../components/recommend/wizardTypes";
import { getCardCatalog, recommendPortfolio, updateSavedCards } from "../api/client";
import type { CardCatalogItem, SpendingCategories } from "../types";
import { mapPortfolioToPredictionResponse } from "../viewmodels/viewMappers";
import { useAuth } from "../context/AuthContext";

interface StepErrors {
  spending?: string;
  rewards?: string;
  income?: string;
}

function validateStep1(form: WizardFormState): string | undefined {
  const total = Object.values(form.spending).reduce((a, b) => a + b, 0);
  if (total <= 0) return "Set at least one spending category above $0.";
  return undefined;
}

function validateStep2(form: WizardFormState): StepErrors {
  const errs: StepErrors = {};
  if (form.selectedRewards.length === 0) {
    errs.rewards = "Select at least one preferred reward type.";
  }
  if (!form.incomeRange) {
    errs.income = "Select your annual income range.";
  }
  return errs;
}

export default function RecommendPage() {
  const navigate = useNavigate();
  const { user } = useAuth();

  const [step, setStep] = useState<WizardStep>(1);
  const [form, setForm] = useState<WizardFormState>(() =>
    createInitialWizardState(),
  );
  const [errors, setErrors] = useState<StepErrors>({});
  const [loading, setLoading] = useState(false);
  const [apiError, setApiError] = useState("");
  const [cardCatalog, setCardCatalog] = useState<CardCatalogItem[]>([]);
  const [catalogLoading, setCatalogLoading] = useState(true);

  const totalSpend = useMemo(
    () => Object.values(form.spending).reduce((a, b) => a + b, 0),
    [form.spending],
  );
  const detectedArchetype = useMemo(
    () => detectArchetype(form.spending),
    [form.spending],
  );
  const selectedArchetype: FrontendArchetype =
    form.selectedArchetype ?? detectedArchetype;

  useEffect(() => {
    if (!user) return;
    setForm((prev) => ({
      ...prev,
      // Pre-populate cards the user already has saved in their wallet
      currentCards:
        prev.currentCards.length > 0
          ? prev.currentCards
          : (user.saved_card_ids ?? []),
      selectedRewards:
        prev.selectedRewards.length > 0
          ? prev.selectedRewards
          : user.reward_preference
            ? [user.reward_preference]
            : [],
    }));
  }, [user]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setCatalogLoading(true);
      try {
        const cards = await getCardCatalog();
        if (!cancelled) setCardCatalog(cards);
      } catch {
        if (!cancelled) setCardCatalog([]);
      } finally {
        if (!cancelled) setCatalogLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const updateSpending = useCallback((key: CategoryKey, value: number) => {
    setForm((prev) => ({
      ...prev,
      spending: { ...prev.spending, [key]: value },
    }));
    setErrors((e) => ({ ...e, spending: undefined }));
  }, []);

  const step1Valid = validateStep1(form) === undefined;
  const step2Errors = validateStep2(form);
  const step2Valid =
    step2Errors.rewards === undefined && step2Errors.income === undefined;

  const goNext = useCallback(() => {
    if (step === 1) {
      const msg = validateStep1(form);
      if (msg) {
        setErrors((e) => ({ ...e, spending: msg }));
        return;
      }
      setErrors((e) => ({ ...e, spending: undefined }));
      setStep(2);
      return;
    }
    if (step === 2) {
      const e2 = validateStep2(form);
      if (Object.keys(e2).length > 0) {
        setErrors((prev) => ({ ...prev, ...e2 }));
        return;
      }
      setErrors((prev) => ({ ...prev, rewards: undefined, income: undefined }));
      setStep(3);
    }
  }, [form, step]);

  const goBack = useCallback(() => {
    if (step === 2) setStep(1);
    else if (step === 3) setStep(2);
  }, [step]);

  const handleSubmit = useCallback(async () => {
    setApiError("");
    const e1 = validateStep1(form);
    const e2 = validateStep2(form);
    if (e1 || Object.keys(e2).length > 0) {
      setErrors({ spending: e1, ...e2 });
      setStep(e1 ? 1 : 2);
      return;
    }
    setErrors({});

    setLoading(true);
    try {
      const spendingCategories: SpendingCategories = {
        ...INITIAL_SPENDING,
        ...form.spending,
      };

      // Persist card selections to the wallet so they're reflected in future
      // sessions and excluded from new-card recommendations automatically.
      if (form.currentCards.length > 0) {
        await updateSavedCards(form.currentCards);
      }

      const start = performance.now();
      const [portfolioResult, catalog] = await Promise.all([
        recommendPortfolio({
          spending_categories: spendingCategories as Record<string, number>,
          monthly_spend: totalSpend,
          use_full_catalog: true,
          personas: [ARCHETYPE_META[selectedArchetype].backendPersona],
        }),
        getCardCatalog(),
      ]);

      const response = mapPortfolioToPredictionResponse({
        portfolio: portfolioResult,
        catalog,
        totalSpend,
        latencyMs: Math.round(performance.now() - start),
        selectedArchetype,
      });
      navigate("/results", { state: response });
    } catch (err) {
      setApiError(
        err instanceof Error ? err.message : "Something went wrong. Please try again.",
      );
    } finally {
      setLoading(false);
    }
  }, [form, navigate, selectedArchetype, totalSpend]);

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key !== "Enter" || loading) return;
      const t = e.target as HTMLElement | null;
      if (t?.closest("textarea")) return;

      if (step === 1 && step1Valid) {
        e.preventDefault();
        goNext();
      } else if (step === 2 && step2Valid) {
        e.preventDefault();
        goNext();
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [goNext, loading, step, step1Valid, step2Valid]);

  const slidePct = ((step - 1) * 100) / 3;

  return (
    <>
      <div className="max-w-3xl mx-auto">
        <div className="mb-6 sm:mb-8">
          <h1 className="text-2xl sm:text-3xl font-bold text-secondary">
            Find your best cards
          </h1>
          <p className="mt-2 text-sm sm:text-base text-slate-600 dark:text-slate-400">
            A quick three-step guide to match you with the right cards.
          </p>
        </div>

        <StepIndicator currentStep={step} />

        <div className="overflow-hidden w-full rounded-xl">
          <div
            className="flex w-[300%] transition-transform duration-300 ease-out motion-reduce:transition-none"
            style={{ transform: `translateX(-${slidePct}%)` }}
          >
            <div className="w-1/3 shrink-0 min-w-0 px-0.5 sm:px-0">
              <SpendingStep
                spending={form.spending}
                totalSpend={totalSpend}
                detectedArchetype={detectedArchetype}
                error={errors.spending}
                onChange={updateSpending}
              />
            </div>
            <div className="w-1/3 shrink-0 min-w-0 px-0.5 sm:px-0">
              <PreferencesStep
                catalog={cardCatalog}
                catalogLoading={catalogLoading}
                selectedRewards={form.selectedRewards}
                incomeRange={form.incomeRange}
                currentCards={form.currentCards}
                selectedArchetype={selectedArchetype}
                rewardsError={errors.rewards}
                incomeError={errors.income}
                onRewardsChange={(v) => {
                  setForm((p) => ({ ...p, selectedRewards: v }));
                  setErrors((e) => ({ ...e, rewards: undefined }));
                }}
                onIncomeChange={(v) => {
                  setForm((p) => ({ ...p, incomeRange: v }));
                  setErrors((e) => ({ ...e, income: undefined }));
                }}
                onCardsChange={(v) =>
                  setForm((p) => ({ ...p, currentCards: v }))
                }
                onArchetypeChange={(value) =>
                  setForm((p) => ({ ...p, selectedArchetype: value }))
                }
              />
            </div>
            <div className="w-1/3 shrink-0 min-w-0 px-0.5 sm:px-0">
              <ReviewStep
                state={form}
                totalSpend={totalSpend}
                catalog={cardCatalog}
                onEdit={(s) => setStep(s)}
                onSubmit={handleSubmit}
                loading={loading}
                apiError={apiError}
              />
            </div>
          </div>
        </div>

        {step < 3 && (
          <div className="mt-6 flex flex-col-reverse gap-3 sm:flex-row sm:items-center sm:justify-between">
            {step >= 2 ? (
              <Button
                type="button"
                variant="secondary"
                onClick={goBack}
                disabled={loading}
              >
                Back
              </Button>
            ) : (
              <span className="hidden sm:block" aria-hidden />
            )}
            <Button
              type="button"
              onClick={goNext}
              disabled={
                loading || (step === 1 ? !step1Valid : step === 2 && !step2Valid)
              }
              className="w-full sm:w-auto sm:min-w-[140px] sm:ml-auto"
            >
              Next
            </Button>
          </div>
        )}

        {step === 3 && (
          <div className="mt-6 flex justify-start">
            <Button
              type="button"
              variant="secondary"
              onClick={goBack}
              disabled={loading}
            >
              Back
            </Button>
          </div>
        )}
      </div>

      {loading && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/40 backdrop-blur-sm pointer-events-auto">
          <Card className="text-center max-w-xs mx-4">
            <LoadingSpinner size="lg" className="mx-auto mb-4" />
            <p className="font-medium text-secondary">Analyzing your profile...</p>
            <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
              This usually takes a few seconds
            </p>
          </Card>
        </div>
      )}
    </>
  );
}
