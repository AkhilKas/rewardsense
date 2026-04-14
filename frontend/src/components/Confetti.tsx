import { useMemo } from "react";

/** Option C palette: amber / gold, white, emerald */
const CONFETTI_COLORS = [
  "#f59e0b",
  "#d97706",
  "#fcd34d",
  "#fef3c7",
  "#ffffff",
  "#22c55e",
  "#4ade80",
];

const PARTICLE_MIN = 50;
const PARTICLE_MAX = 80;

interface Particle {
  leftPct: number;
  width: number;
  height: number;
  color: string;
  drift: string;
  spin: string;
  delayMs: number;
  durationMs: number;
}

function randomBetween(min: number, max: number): number {
  return min + Math.random() * (max - min);
}

function buildParticles(count: number): Particle[] {
  return Array.from({ length: count }, () => ({
    leftPct: Math.random() * 100,
    width: randomBetween(4, 9),
    height: randomBetween(5, 12),
    color:
      CONFETTI_COLORS[
        Math.floor(Math.random() * CONFETTI_COLORS.length)
      ] as string,
    drift: `${randomBetween(-100, 100)}px`,
    spin: `${randomBetween(360, 1080)}deg`,
    delayMs: Math.random() * 180,
    durationMs: randomBetween(2000, 2500),
  }));
}

/**
 * Full-viewport decorative confetti. Parent controls mount duration.
 * `pointer-events: none` — does not block clicks.
 */
export default function Confetti() {
  const particles = useMemo(() => {
    const count =
      PARTICLE_MIN +
      Math.floor(Math.random() * (PARTICLE_MAX - PARTICLE_MIN + 1));
    return buildParticles(count);
  }, []);

  return (
    <div
      className="fixed inset-0 z-[45] overflow-hidden pointer-events-none motion-reduce:hidden"
      aria-hidden
    >
      {particles.map((p, i) => (
        <div
          key={i}
          className="absolute rounded-[1px] will-change-transform"
          style={{
            left: `${p.leftPct}%`,
            top: "-14px",
            width: p.width,
            height: p.height,
            backgroundColor: p.color,
            opacity: 0.95,
            boxShadow: "0 1px 2px rgba(0,0,0,0.12)",
            ["--confetti-drift" as string]: p.drift,
            ["--confetti-spin" as string]: p.spin,
            animation: `confetti-fall ${p.durationMs}ms ease-out ${p.delayMs}ms forwards`,
          }}
        />
      ))}
    </div>
  );
}
