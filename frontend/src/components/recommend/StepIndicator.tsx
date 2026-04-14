import type { WizardStep } from "./wizardTypes";

const STEPS: { id: WizardStep; short: string }[] = [
  { id: 1, short: "Spending" },
  { id: 2, short: "Preferences" },
  { id: 3, short: "Review" },
];

function CheckIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d="M20 6L9 17l-5-5" />
    </svg>
  );
}

interface StepIndicatorProps {
  currentStep: WizardStep;
}

function StepCircle({
  step,
  currentStep,
}: {
  step: (typeof STEPS)[number];
  currentStep: WizardStep;
}) {
  const done = currentStep > step.id;
  const active = currentStep === step.id;

  return (
    <div className="flex flex-col items-center shrink-0 w-[72px] sm:w-24">
      <div
        className={`flex h-9 w-9 sm:h-10 sm:w-10 items-center justify-center rounded-full border-2 text-xs sm:text-sm font-semibold transition-colors duration-300 ${
          done
            ? "border-primary bg-primary text-white"
            : active
              ? "border-primary bg-primary-light text-primary ring-2 ring-primary/30"
              : "border-border bg-card text-slate-400 dark:text-slate-500"
        }`}
        aria-current={active ? "step" : undefined}
      >
        {done ? (
          <CheckIcon className="h-4 w-4 sm:h-5 sm:w-5" />
        ) : (
          <span>{step.id}</span>
        )}
      </div>
      <span
        className={`mt-1.5 text-[10px] sm:text-xs font-medium text-center leading-tight px-0.5 ${
          active
            ? "text-primary"
            : done
              ? "text-secondary"
              : "text-slate-400 dark:text-slate-500"
        }`}
      >
        {step.short}
      </span>
    </div>
  );
}

function Connector({
  leftStepId,
  currentStep,
}: {
  leftStepId: WizardStep;
  currentStep: WizardStep;
}) {
  const filled = currentStep > leftStepId;
  return (
    <div
      className={`flex-1 h-0.5 mt-[18px] sm:mt-5 mx-1 sm:mx-2 rounded min-w-[8px] transition-colors duration-300 ${
        filled ? "bg-primary" : "bg-border"
      }`}
      aria-hidden
    />
  );
}

export default function StepIndicator({ currentStep }: StepIndicatorProps) {
  return (
    <nav className="mb-8" aria-label="Recommendation steps">
      <div className="flex items-start w-full">
        <StepCircle step={STEPS[0]} currentStep={currentStep} />
        <Connector leftStepId={1} currentStep={currentStep} />
        <StepCircle step={STEPS[1]} currentStep={currentStep} />
        <Connector leftStepId={2} currentStep={currentStep} />
        <StepCircle step={STEPS[2]} currentStep={currentStep} />
      </div>
    </nav>
  );
}
