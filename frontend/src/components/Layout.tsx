import { useState } from "react";
import { Link, NavLink, Outlet } from "react-router-dom";
import ThemeToggle from "./ThemeToggle";

const navLinks = [
  { to: "/", label: "Home" },
  { to: "/recommend", label: "Recommend" },
  { to: "/dashboard", label: "Dashboard" },
];

export default function Layout() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <div className="min-h-screen flex flex-col bg-surface font-sans transition-colors duration-200">
      <header className="sticky top-0 z-50 bg-card border-b border-border transition-colors duration-200">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <Link to="/" className="text-xl font-bold text-primary shrink-0">
            RewardSense
          </Link>

          {/* Desktop nav */}
          <div className="hidden sm:flex items-center gap-1">
            <ThemeToggle />
            <nav className="flex items-center gap-1">
              {navLinks.map((link) => (
                <NavLink
                  key={link.to}
                  to={link.to}
                  end={link.to === "/"}
                  className={({ isActive }) =>
                    `px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200 ${
                      isActive
                        ? "bg-primary-light text-primary dark:text-blue-300"
                        : "text-slate-600 hover:text-secondary hover:bg-slate-100 dark:text-slate-400 dark:hover:text-slate-200 dark:hover:bg-slate-700"
                    }`
                  }
                >
                  {link.label}
                </NavLink>
              ))}
            </nav>
          </div>

          {/* Mobile controls */}
          <div className="flex sm:hidden items-center gap-1">
            <ThemeToggle />
            <button
              onClick={() => setMobileMenuOpen((o) => !o)}
              className="p-2 rounded-md text-slate-500 hover:text-secondary hover:bg-slate-100 dark:text-slate-400 dark:hover:bg-slate-700 dark:hover:text-slate-200 transition-colors cursor-pointer"
              aria-label="Toggle menu"
            >
              {mobileMenuOpen ? (
                <svg
                  className="h-5 w-5"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M18 6 6 18M6 6l12 12" />
                </svg>
              ) : (
                <svg
                  className="h-5 w-5"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M3 12h18M3 6h18M3 18h18" />
                </svg>
              )}
            </button>
          </div>
        </div>

        {/* Mobile dropdown menu */}
        {mobileMenuOpen && (
          <nav className="sm:hidden border-t border-border bg-card px-4 pb-3 pt-2 space-y-1 transition-colors duration-200">
            {navLinks.map((link) => (
              <NavLink
                key={link.to}
                to={link.to}
                end={link.to === "/"}
                onClick={() => setMobileMenuOpen(false)}
                className={({ isActive }) =>
                  `block px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200 ${
                    isActive
                      ? "bg-primary-light text-primary dark:text-blue-300"
                      : "text-slate-600 hover:text-secondary hover:bg-slate-100 dark:text-slate-400 dark:hover:text-slate-200 dark:hover:bg-slate-700"
                  }`
                }
              >
                {link.label}
              </NavLink>
            ))}
          </nav>
        )}
      </header>

      <main className="flex-1">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <Outlet />
        </div>
      </main>

      <footer className="border-t border-border bg-card transition-colors duration-200">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-6 text-center text-sm text-slate-500 dark:text-slate-400">
          RewardSense &copy; 2026. Built with MLOps.
        </div>
      </footer>
    </div>
  );
}
