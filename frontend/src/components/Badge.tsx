import type { ReactNode } from "react";

interface BadgeProps {
  children: ReactNode;
  variant?: "info" | "success" | "warning" | "danger";
  className?: string;
}

const variantClasses: Record<string, string> = {
  info: "bg-primary-light text-primary dark:text-blue-300",
  success: "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400",
  warning: "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400",
  danger: "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400",
};

export default function Badge({
  children,
  variant = "info",
  className = "",
}: BadgeProps) {
  return (
    <span
      className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium transition-colors duration-200 ${variantClasses[variant]} ${className}`}
    >
      {children}
    </span>
  );
}
