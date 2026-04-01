interface SliderInputProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  format?: (value: number) => string;
  className?: string;
}

const dollarFormat = (v: number) => `$${v.toLocaleString()}`;

export default function SliderInput({
  label,
  value,
  onChange,
  min = 0,
  max = 3000,
  step = 50,
  format = dollarFormat,
  className = "",
}: SliderInputProps) {
  const pct = max > min ? ((value - min) / (max - min)) * 100 : 0;

  const trackStyle = {
    background: `linear-gradient(to right, var(--color-primary) ${pct}%, var(--color-border) ${pct}%)`,
  };

  return (
    <div className={className}>
      <div className="flex items-center justify-between mb-1.5">
        <label className="text-sm font-medium text-secondary">{label}</label>
        <span className="text-sm font-semibold text-primary tabular-nums">
          {format(value)}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        style={trackStyle}
        className="w-full"
      />
    </div>
  );
}
