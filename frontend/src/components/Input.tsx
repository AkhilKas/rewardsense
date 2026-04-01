import type { InputHTMLAttributes } from "react";

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label: string;
  error?: string;
}

export default function Input({
  label,
  error,
  id,
  className = "",
  ...props
}: InputProps) {
  const inputId =
    id || props.name || label.toLowerCase().replace(/\s+/g, "-");

  return (
    <div className={className}>
      <label
        htmlFor={inputId}
        className="block text-sm font-medium text-secondary mb-1"
      >
        {label}
      </label>
      <input
        id={inputId}
        className={`block w-full rounded-md border bg-card px-3 py-2 text-sm text-secondary placeholder:text-slate-400 dark:placeholder:text-slate-500 transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-primary focus:border-primary ${
          error ? "border-danger" : "border-border"
        }`}
        {...props}
      />
      {error && <p className="mt-1 text-xs text-danger">{error}</p>}
    </div>
  );
}
