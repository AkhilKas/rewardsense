import { memo, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import Badge from "../components/Badge";
import Button from "../components/Button";
import Card from "../components/Card";
import ScoreGauge from "../components/ScoreGauge";
import { useAuth } from "../context/AuthContext";
import { useScrollAnimation } from "../hooks/useScrollAnimation";

const HOW_IT_WORKS_STEPS = [
  {
    step: "01",
    title: "Tell us how you spend",
    description:
      "Use simple sliders to map your monthly spending across dining, travel, groceries, and more.",
    icon: (
      <svg
        viewBox="0 0 24 24"
        className="h-9 w-9 text-primary"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden
      >
        <line x1="4" y1="7" x2="20" y2="7" />
        <circle cx="9" cy="7" r="2.2" />
        <line x1="4" y1="12" x2="20" y2="12" />
        <circle cx="15" cy="12" r="2.2" />
        <line x1="4" y1="17" x2="20" y2="17" />
        <circle cx="11.5" cy="17" r="2.2" />
      </svg>
    ),
  },
  {
    step: "02",
    title: "Pick your style",
    description:
      "Choose your spending archetype to match recommendations to your reward priorities and fee comfort.",
    icon: (
      <svg
        viewBox="0 0 24 24"
        className="h-9 w-9 text-primary"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden
      >
        <circle cx="12" cy="8" r="3.5" />
        <path d="M5 19c1.8-3 4.4-4.5 7-4.5S17.2 16 19 19" />
        <path d="M4 12l2-2 2 2M20 12l-2-2-2 2" />
      </svg>
    ),
  },
  {
    step: "03",
    title: "Get matched instantly",
    description:
      "See ranked card recommendations with clear projected reward outcomes in seconds.",
    icon: (
      <svg
        viewBox="0 0 24 24"
        className="h-9 w-9 text-primary"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden
      >
        <rect x="3" y="5" width="18" height="14" rx="2.5" />
        <line x1="3" y1="10" x2="21" y2="10" />
        <path d="M8 15h3M14 15h2" />
      </svg>
    ),
  },
] as const;

const HERO_WORDS = ["dining", "travel", "groceries", "everything"] as const;

const HERO_STATS = [
  { label: "Cards Analyzed", value: 9, suffix: "+" },
  { label: "Step Process", value: 3, suffix: "" },
  { label: "Seconds", value: 30, prefix: "< " },
];

function usePrefersReducedMotion() {
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(false);

  useEffect(() => {
    const mediaQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setPrefersReducedMotion(mediaQuery.matches);
    update();
    mediaQuery.addEventListener("change", update);
    return () => mediaQuery.removeEventListener("change", update);
  }, []);

  return prefersReducedMotion;
}

function AnimatedStat({
  value,
  label,
  prefix = "",
  suffix = "",
  shouldAnimate,
  durationMs = 1100,
}: {
  value: number;
  label: string;
  prefix?: string;
  suffix?: string;
  shouldAnimate: boolean;
  durationMs?: number;
}) {
  const [displayValue, setDisplayValue] = useState(0);

  useEffect(() => {
    if (!shouldAnimate) {
      setDisplayValue(value);
      return;
    }

    let raf = 0;
    const start = performance.now();

    const tick = (now: number) => {
      const progress = Math.min((now - start) / durationMs, 1);
      setDisplayValue(Math.round(value * progress));
      if (progress < 1) {
        raf = window.requestAnimationFrame(tick);
      }
    };

    raf = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(raf);
  }, [durationMs, shouldAnimate, value]);

  return (
    <div className="rounded-xl border border-border/70 bg-card/70 px-4 py-3 shadow-sm backdrop-blur-sm dark:border-border/70 dark:bg-card/60">
      <p className="font-mono text-xl sm:text-2xl font-semibold text-secondary">
        {prefix}
        {displayValue}
        {suffix}
      </p>
      <p className="mt-1 text-xs sm:text-sm text-slate-600 dark:text-zinc-400">
        {label}
      </p>
    </div>
  );
}

function HeroSection({ ctaPath }: { ctaPath: string }) {
  const prefersReducedMotion = usePrefersReducedMotion();
  const [activeWordIndex, setActiveWordIndex] = useState(0);
  const { ref: statsRef, isVisible: statsVisible } =
    useScrollAnimation<HTMLDivElement>({ threshold: 0.35 });

  useEffect(() => {
    if (prefersReducedMotion) return;
    const interval = window.setInterval(() => {
      setActiveWordIndex((prev) => (prev + 1) % HERO_WORDS.length);
    }, 2000);
    return () => window.clearInterval(interval);
  }, [prefersReducedMotion]);

  const activeWord = useMemo(
    () => HERO_WORDS[activeWordIndex],
    [activeWordIndex],
  );

  return (
    <section className="relative overflow-hidden rounded-3xl border border-border/70 bg-linear-to-br from-surface via-card to-primary-light/35 px-6 py-14 text-center shadow-sm sm:px-10 sm:py-18 dark:border-border/80 dark:from-[#0a0a0a] dark:via-[#101010] dark:to-[#181818]">
      <div className="relative z-10 mx-auto max-w-4xl">
        <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-secondary tracking-tight leading-[1.1] sm:leading-tight">
          <span className="block">Find your perfect</span>
          <span className="mt-1 block sm:mt-0.5 sm:whitespace-nowrap">
            <span className="text-secondary">card for </span>
            <span
              key={activeWord}
              className={`inline-block text-primary ${
                prefersReducedMotion ? "" : "premium-text-swap"
              }`}
            >
              {activeWord}
            </span>
          </span>
        </h1>
        <p className="mt-5 text-base sm:text-lg text-slate-700 dark:text-zinc-300 max-w-2xl mx-auto">
          Smart recommendations based on how you actually spend.
        </p>
      </div>

      <div className="relative z-10 mt-8 flex items-center justify-center gap-4">
        <Link to={ctaPath}>
          <Button size="lg" className="premium-pulse-glow px-8">
            Get Started. It&apos;s Free
          </Button>
        </Link>
      </div>

      <div
        ref={statsRef}
        className="relative z-10 mx-auto mt-10 grid max-w-3xl grid-cols-1 gap-3 sm:grid-cols-3"
      >
        {HERO_STATS.map((stat) => (
          <AnimatedStat
            key={stat.label}
            value={stat.value}
            label={stat.label}
            prefix={stat.prefix}
            suffix={stat.suffix}
            shouldAnimate={statsVisible && !prefersReducedMotion}
          />
        ))}
      </div>
    </section>
  );
}

const PREVIEW_MOCK_CARDS = [
  {
    id: "1",
    rankClass: "preview-mock-card-1",
    score: 93.6,
    name: "Chase Sapphire Preferred",
    issuer: "Chase",
    annualFee: 95,
    rewardRate: 2.8,
    scoreBreakdown: { base: 71.4, boosted: 22.2 },
    keyBenefits: [
      "2x travel rewards",
      "3x dining rewards",
      "60k intro bonus",
    ],
    why: "Strong dining and travel value with a low annual fee for frequent spenders.",
  },
  {
    id: "2",
    rankClass: "preview-mock-card-2",
    score: 89.1,
    name: "Amex Gold Card",
    issuer: "American Express",
    annualFee: 250,
    rewardRate: 3.1,
    scoreBreakdown: { base: 67.3, boosted: 21.8 },
    keyBenefits: [
      "4x dining rewards",
      "4x U.S. supermarkets",
      "Dining credits",
    ],
    why: "Top food and grocery earning rates offset the fee for high monthly spend.",
  },
  {
    id: "3",
    rankClass: "preview-mock-card-3",
    score: 84.4,
    name: "Capital One Venture X",
    issuer: "Capital One",
    annualFee: 395,
    rewardRate: 2.4,
    scoreBreakdown: { base: 63.1, boosted: 21.3 },
    keyBenefits: [
      "10k anniversary miles",
      "$300 travel credit",
      "Lounge access",
    ],
    why: "Premium travel perks and simple 2x miles provide high long-term value.",
  },
] as const;

function ScoreBreakdownBarMini({
  base,
  boosted,
}: {
  base: number;
  boosted: number;
}) {
  const total = base + boosted;
  const basePct = total > 0 ? (base / total) * 100 : 50;
  const boostPct = 100 - basePct;

  return (
    <div className="mt-2">
      <div className="mb-1 flex items-center justify-between text-[10px] text-slate-500 dark:text-zinc-500">
        <span>Base {base.toFixed(1)}</span>
        <span>Boost {boosted.toFixed(1)}</span>
      </div>
      <div className="flex h-1.5 overflow-hidden rounded-full bg-border">
        <div className="h-full bg-primary/75" style={{ width: `${basePct}%` }} />
        <div className="h-full bg-accent/75" style={{ width: `${boostPct}%` }} />
      </div>
    </div>
  );
}

const LivePreviewSection = memo(function LivePreviewSection() {
  const prefersReducedMotion = usePrefersReducedMotion();
  const { ref: sectionRef, isVisible: inViewport } = useScrollAnimation<HTMLElement>(
    { threshold: 0.2, once: false },
  );
  const [revealActive, setRevealActive] = useState(false);
  const [loopPulse, setLoopPulse] = useState(false);

  useEffect(() => {
    if (inViewport) setRevealActive(true);
  }, [inViewport]);

  useEffect(() => {
    if (!inViewport || prefersReducedMotion) {
      setLoopPulse(false);
      return;
    }

    let intervalId = 0;
    let pulseClearId = 0;

    const triggerPulse = () => {
      setLoopPulse(true);
      pulseClearId = window.setTimeout(() => {
        setLoopPulse(false);
      }, 1280);
    };

    intervalId = window.setInterval(triggerPulse, 5000);

    return () => {
      window.clearInterval(intervalId);
      window.clearTimeout(pulseClearId);
      setLoopPulse(false);
    };
  }, [inViewport, prefersReducedMotion]);

  return (
    <section
      ref={sectionRef}
      className="overflow-hidden rounded-3xl border border-border/70 bg-linear-to-br from-card via-surface to-primary-light/25 px-6 py-10 shadow-sm sm:px-8 sm:py-12 dark:border-border/80 dark:from-[#0a0a0a] dark:via-[#101010] dark:to-[#181818]"
    >
      <div className="mx-auto max-w-6xl">
        <p className="text-center text-xs font-semibold uppercase tracking-widest text-primary lg:text-left">
          Live preview
        </p>
        <div className="mt-2 grid gap-8 lg:grid-cols-2 lg:items-start lg:gap-12">
          <div className="min-w-0 text-center lg:text-left">
            <h2 className="text-2xl font-bold text-secondary sm:text-3xl">
              Ranked cards, real math in seconds
            </h2>
            <p className="mt-3 text-sm text-slate-600 dark:text-zinc-400 sm:text-base">
              Your spending profile turns into a clear stack of matches with
              projected reward strength. The same flow you&apos;ll see after you
              sign up.
            </p>
          </div>

          <div className="min-w-0 w-full -mt-2 sm:-mt-3 lg:-mt-4">
            <div
              className={`preview-mock-stack relative mx-auto w-full overflow-hidden rounded-2xl pb-10 pl-0 pr-1 pt-4 sm:pb-12 sm:pr-2 sm:pt-5 md:pb-14 md:pr-3 md:pt-6 ${
                revealActive ? "preview-mock-reveal-active" : ""
              } ${loopPulse ? "preview-mock-loop-active" : ""}`}
              aria-hidden
            >
              <div className="flex min-h-[196px] min-w-max items-start justify-start -ml-2 pl-0 sm:min-h-[212px] sm:-ml-3 md:-ml-4">
              {PREVIEW_MOCK_CARDS.map((card, index) => (
                <div
                  key={card.id}
                  className={`preview-mock-card ${card.rankClass} relative w-[210px] shrink-0 rounded-2xl border border-border bg-card p-3 text-secondary shadow-lg sm:w-[228px] sm:p-3.5 md:w-[240px] dark:border-primary/30 dark:bg-linear-to-br dark:from-[#1a1a1a] dark:via-[#141414] dark:to-[#0f0f0f] dark:shadow-lg ${
                    index > 0 ? "-ml-6 sm:-ml-7 md:-ml-8" : ""
                  } ${index === 0 ? "z-30" : index === 1 ? "z-20" : "z-10"}`}
                >
                  {index === 0 ? (
                    <Badge
                      variant="success"
                      className="absolute right-2 top-2 z-20 px-1.5 py-0 text-[9px]"
                    >
                      Top Pick
                    </Badge>
                  ) : null}
                  <div className="preview-mock-card-loop-inner h-full w-full rounded-[inherit]">
                    <div className="mb-2 flex items-start gap-2.5">
                      <ScoreGauge score={card.score} size={42} strokeWidth={4} />
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-1.5 pr-14">
                          <p className="truncate text-xs font-semibold text-secondary">
                            {card.name}
                          </p>
                        </div>
                        <p className="mt-0.5 text-[10px] text-slate-500 dark:text-zinc-500">
                          {card.issuer} • ${card.annualFee}/yr • {card.rewardRate}% avg
                        </p>
                      </div>
                    </div>

                    <ScoreBreakdownBarMini
                      base={card.scoreBreakdown.base}
                      boosted={card.scoreBreakdown.boosted}
                    />

                    <div className="mt-2 flex flex-wrap gap-1">
                      {card.keyBenefits.map((benefit) => (
                        <Badge key={benefit} variant="info" className="px-1.5 py-0 text-[9px]">
                          {benefit}
                        </Badge>
                      ))}
                    </div>

                    <div className="mt-2 rounded-lg border border-border/80 bg-surface/80 p-2 dark:bg-surface/60">
                      <p className="text-[10px] font-semibold uppercase tracking-wide text-secondary/80">
                        Why this card?
                      </p>
                      <p className="mt-1 line-clamp-2 text-[10px] leading-relaxed text-slate-600 dark:text-zinc-400">
                        {card.why}
                      </p>
                    </div>

                    <div className="mt-2 flex items-center justify-between border-t border-border/80 pt-1.5">
                      <span className="text-[10px] text-slate-500 dark:text-zinc-500">
                        Match score
                      </span>
                      <p className="text-xs font-bold tabular-nums text-primary">
                        {card.score.toFixed(1)}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
});

const HowItWorksSection = memo(function HowItWorksSection() {
  const prefersReducedMotion = usePrefersReducedMotion();
  const { ref: stepsRef, isVisible: stepsVisible } =
    useScrollAnimation<HTMLElement>({ threshold: 0.2 });

  return (
    <section
      ref={stepsRef}
      className="rounded-3xl border border-border/70 bg-card/70 px-6 py-10 shadow-sm sm:px-8 sm:py-12 dark:border-border/80 dark:bg-card/60"
    >
      <h2 className="text-center text-2xl sm:text-3xl font-bold text-secondary">
        How It Works
      </h2>
      <p className="mx-auto mt-3 max-w-2xl text-center text-sm sm:text-base text-slate-600 dark:text-zinc-400">
        Three focused steps from your spending profile to your best card match.
      </p>

      <div className="relative mt-10">
        <div className="pointer-events-none absolute left-[16.66%] right-[16.66%] top-[3.1rem] hidden lg:block">
          <svg viewBox="0 0 100 8" className="h-8 w-full" aria-hidden>
            <path
              d="M2 4 H98"
              className={`fill-none stroke-primary/50 stroke-[1.5] ${
                prefersReducedMotion ? "" : "premium-dash-flow"
              }`}
            />
          </svg>
        </div>

        <div
          className={`grid grid-cols-1 gap-4 sm:gap-5 lg:grid-cols-3 ${
            stepsVisible ? "steps-reveal-active" : ""
          }`}
        >
          {HOW_IT_WORKS_STEPS.map((step, index) => (
            <Card
              key={step.step}
              className={`step-card relative z-10 border border-border/80 bg-card/85 text-left shadow-sm backdrop-blur-sm dark:border-border/80 dark:bg-card/80 ${
                index === 0
                  ? "step-card-1"
                  : index === 1
                  ? "step-card-2"
                  : "step-card-3"
              }`}
            >
              <div className="mb-4 flex items-center gap-3">
                <div className="flex h-12 w-12 items-center justify-center rounded-full border-2 border-primary/60 bg-primary/10 text-sm font-bold tracking-wide text-primary">
                  {step.step}
                </div>
                <div className="flex h-11 w-11 items-center justify-center rounded-lg bg-primary/10">
                  {step.icon}
                </div>
              </div>
              <h3 className="text-lg font-semibold text-secondary">{step.title}</h3>
              <p className="mt-2 text-sm leading-6 text-slate-600 dark:text-zinc-400">
                {step.description}
              </p>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
});

const TRUST_BADGES = [
  {
    id: "enc",
    label: "256-bit Encrypted",
    icon: (
      <svg
        viewBox="0 0 24 24"
        className="h-6 w-6 text-primary"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.75"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden
      >
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
        <path d="M9 12l2 2 4-4" />
      </svg>
    ),
  },
  {
    id: "data",
    label: "No Data Stored",
    icon: (
      <svg
        viewBox="0 0 24 24"
        className="h-6 w-6 text-primary"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.75"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden
      >
        <ellipse cx="12" cy="5" rx="9" ry="3" />
        <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" />
        <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
      </svg>
    ),
  },
  {
    id: "ind",
    label: "Independent Recommendations",
    icon: (
      <svg
        viewBox="0 0 24 24"
        className="h-6 w-6 text-primary"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.75"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden
      >
        <path d="M12 3l1.9 5.8h6.1l-5 3.6 1.9 5.8-5-3.6-5 3.6 1.9-5.8-5-3.6h6.1z" />
      </svg>
    ),
  },
] as const;

const TESTIMONIALS = [
  {
    quote: "I switched cards and saved $340 in my first year.",
    attribution: "Jordan M.",
  },
  {
    quote: "The 3-step wizard made it so easy. Got matched in under a minute.",
    attribution: "Priya S.",
  },
  {
    quote: "Finally understand why my old card was costing me money.",
    attribution: "Marcus T.",
  },
  {
    quote: "The travel card recommendation alone paid for two lounge visits.",
    attribution: "Sarah K.",
  },
  {
    quote: "Way better than scrolling through comparison sites for hours.",
    attribution: "Alex R.",
  },
] as const;

const TrustSection = memo(function TrustSection() {
  const prefersReducedMotion = usePrefersReducedMotion();
  const { ref: sectionRef, isVisible } = useScrollAnimation<HTMLElement>({
    threshold: 0.2,
  });
  const [testimonialIndex, setTestimonialIndex] = useState(0);
  const [quoteOpacity, setQuoteOpacity] = useState(1);

  useEffect(() => {
    if (prefersReducedMotion) return;
    const fadeMs = 300;
    const id = window.setInterval(() => {
      setQuoteOpacity(0);
      window.setTimeout(() => {
        setTestimonialIndex((prev) => (prev + 1) % TESTIMONIALS.length);
        setQuoteOpacity(1);
      }, fadeMs);
    }, 4000);
    return () => window.clearInterval(id);
  }, [prefersReducedMotion]);

  const active = TESTIMONIALS[testimonialIndex];

  return (
    <section
      ref={sectionRef}
      className={`rounded-3xl border border-border/70 bg-card/80 px-6 py-10 shadow-sm sm:px-8 sm:py-12 dark:border-border/80 dark:bg-card/55 ${
        isVisible ? "trust-section-visible" : ""
      }`}
    >
      <div className="mx-auto max-w-4xl">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3 sm:gap-5">
          {TRUST_BADGES.map((badge, index) => (
            <div
              key={badge.id}
              className={`trust-badge flex items-center gap-3 rounded-xl border border-border/80 bg-surface/80 px-4 py-3 dark:border-border/80 dark:bg-surface/70 ${
                index === 0
                  ? "trust-badge-1"
                  : index === 1
                  ? "trust-badge-2"
                  : "trust-badge-3"
              }`}
            >
              <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg bg-primary/10 dark:bg-primary/15">
                {badge.icon}
              </div>
              <p className="text-sm font-medium leading-snug text-secondary">
                {badge.label}
              </p>
            </div>
          ))}
        </div>

        <div className="trust-testimonial-block mx-auto mt-10 max-w-2xl sm:mt-12">
          <div
            className="min-h-[9rem] px-2 sm:min-h-[8rem] sm:px-6"
            aria-live="polite"
            aria-atomic="true"
          >
            <div
              className="transition-opacity duration-300 ease-in-out"
              style={{ opacity: quoteOpacity }}
            >
              <blockquote className="mx-auto max-w-xl">
                <div className="relative text-lg leading-relaxed text-secondary sm:text-xl sm:leading-relaxed">
                  <span
                    className="pointer-events-none absolute left-0 top-0 z-0 font-serif text-[2.35rem] font-normal leading-[0.85] text-primary not-italic sm:text-[2.85rem]"
                    aria-hidden
                  >
                    &ldquo;
                  </span>
                  <p className="relative z-[1] px-11 pb-12 pt-10 text-center font-medium sm:px-14 sm:pb-14 sm:pt-11">
                    {active.quote}
                  </p>
                  <span
                    className="pointer-events-none absolute bottom-0 right-0 z-0 font-serif text-[2.35rem] font-normal leading-[0.85] text-primary not-italic sm:text-[2.85rem]"
                    aria-hidden
                  >
                    &rdquo;
                  </span>
                </div>
              </blockquote>
              <p className="mt-4 text-center text-sm text-slate-600 dark:text-zinc-400">
                {active.attribution}
              </p>
            </div>
          </div>

          <div
            className="mt-6 flex justify-center gap-2"
            role="tablist"
            aria-label="Testimonial slides"
          >
            {TESTIMONIALS.map((_, i) => (
              <button
                key={`testimonial-dot-${i}`}
                type="button"
                role="tab"
                aria-selected={i === testimonialIndex}
                aria-label={`Testimonial ${i + 1} of ${TESTIMONIALS.length}`}
                className={`h-2 rounded-full transition-all duration-300 ease-out focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 dark:focus:ring-offset-[#0f0f0f] ${
                  i === testimonialIndex
                    ? "w-6 bg-primary"
                    : "w-2 bg-slate-400/55 hover:bg-slate-400/80 dark:bg-white/25 dark:hover:bg-white/35"
                }`}
                onClick={() => {
                  setTestimonialIndex(i);
                  setQuoteOpacity(1);
                }}
              />
            ))}
          </div>
        </div>
      </div>
    </section>
  );
});

const FinalCtaSection = memo(function FinalCtaSection({
  ctaPath,
}: {
  ctaPath: string;
}) {
  const { ref: sectionRef, isVisible } = useScrollAnimation<HTMLElement>({
    threshold: 0.22,
  });

  return (
    <section
      ref={sectionRef}
      aria-labelledby="final-cta-heading"
      className={`relative overflow-hidden rounded-3xl border border-primary/35 bg-linear-to-br from-[#faf9f7] via-[#f5f0eb] to-[#faf8f5] px-6 py-14 text-center shadow-xl shadow-primary/10 sm:px-10 sm:py-16 dark:border-primary/45 dark:from-[#0a0a0a] dark:via-[#101010] dark:to-[#181818] dark:shadow-primary/15 ${
        isVisible ? "final-cta-visible" : ""
      }`}
    >
      <div className="pointer-events-none absolute inset-0 -z-0">
        <div className="absolute -right-24 -top-28 h-72 w-72 rounded-full bg-primary/18 blur-3xl dark:bg-primary/18" />
        <div className="absolute -bottom-32 -left-20 h-64 w-64 rounded-full bg-primary/10 blur-3xl dark:bg-primary/12" />
        <div
          className="premium-float absolute bottom-14 right-[18%] hidden h-14 w-20 rounded-xl border border-primary/25 bg-primary/[0.08] opacity-70 sm:block dark:border-white/10 dark:bg-white/[0.06] dark:opacity-50"
          style={{ animationDelay: "1.2s" }}
          aria-hidden
        />
      </div>

      <div className="final-cta-block relative z-10 mx-auto max-w-2xl">
        <p className="text-xs font-semibold uppercase tracking-widest text-primary">
          Start free
        </p>
        <h2
          id="final-cta-heading"
          className="mt-3 text-2xl font-bold tracking-tight text-secondary sm:text-3xl dark:text-white"
        >
          Ready to Find Your Best Card?
        </h2>
        <p className="mt-4 text-base leading-relaxed text-secondary sm:text-lg dark:text-zinc-300">
          Enter your spending profile and get personalized recommendations in
          seconds. Clear math, ranked matches, no endless comparison tabs.
        </p>
        <div className="mt-8 flex justify-center">
          <Link to={ctaPath}>
            <Button
              size="lg"
              className="premium-pulse-glow px-8 shadow-lg shadow-primary/25"
            >
              Get Started. It&apos;s Free
            </Button>
          </Link>
        </div>
      </div>
    </section>
  );
});

export default function HomePage() {
  const { isAuthenticated } = useAuth();
  const ctaPath = isAuthenticated ? "/recommend" : "/signup";

  return (
    <div className="space-y-16">
      <HeroSection ctaPath={ctaPath} />
      <HowItWorksSection />
      <LivePreviewSection />
      <TrustSection />
      <FinalCtaSection ctaPath={ctaPath} />
    </div>
  );
}
