import {
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { createPortal } from "react-dom";
import type { CardCatalogItem } from "../types";

function matchesQuery(card: CardCatalogItem, q: string): boolean {
  if (!q) return true;
  const n = card.card_name.toLowerCase();
  const iss = card.issuer.toLowerCase();
  return (
    n.includes(q) ||
    iss.includes(q) ||
    card.reward_highlights.some((h) => h.toLowerCase().includes(q))
  );
}

export interface CardMultiSelectComboboxProps {
  label: string;
  optional?: boolean;
  description?: string;
  catalog: CardCatalogItem[];
  selectedIds: string[];
  onChange: (ids: string[]) => void;
  placeholder?: string;
  disabled?: boolean;
  error?: string;
  /**
   * `fixed` + portal: use inside parents with `overflow: hidden` or `transform`
   * (e.g. wizard slider). Plain `fixed` inside a transformed ancestor is broken
   * in CSS — the menu is clipped or positioned wrong; portaling to `document.body`
   * fixes that.
   */
  dropdownStrategy?: "absolute" | "fixed";
}

export default function CardMultiSelectCombobox({
  label,
  optional,
  description,
  catalog,
  selectedIds,
  onChange,
  placeholder = "Search cards by name, issuer, or benefit…",
  disabled = false,
  error,
  dropdownStrategy = "absolute",
}: CardMultiSelectComboboxProps) {
  const id = useId();
  const inputId = `${id}-input`;
  const containerRef = useRef<HTMLDivElement>(null);
  const portalListRef = useRef<HTMLUListElement>(null);
  const anchorRef = useRef<HTMLDivElement>(null);
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [fixedBox, setFixedBox] = useState<{
    top: number;
    left: number;
    width: number;
  } | null>(null);

  const byId = useMemo(() => {
    const m = new Map<string, CardCatalogItem>();
    for (const c of catalog) m.set(c.card_id, c);
    return m;
  }, [catalog]);

  const available = useMemo(
    () => catalog.filter((c) => !selectedIds.includes(c.card_id)),
    [catalog, selectedIds],
  );

  const q = query.trim().toLowerCase();
  const filtered = useMemo(
    () => available.filter((c) => matchesQuery(c, q)),
    [available, q],
  );

  /** Panel open whenever the field is active — avoids “nothing happens” on focus. */
  const menuOpen = open && !disabled;

  useEffect(() => {
    function onDocMouseDown(e: MouseEvent) {
      const t = e.target as Node | null;
      if (!t) return;
      if (containerRef.current?.contains(t)) return;
      if (portalListRef.current?.contains(t)) return;
      setOpen(false);
    }
    document.addEventListener("mousedown", onDocMouseDown);
    return () => document.removeEventListener("mousedown", onDocMouseDown);
  }, []);

  function addCard(cardId: string) {
    if (selectedIds.includes(cardId)) return;
    onChange([...selectedIds, cardId]);
    setQuery("");
    setOpen(true);
  }

  function removeCard(cardId: string) {
    onChange(selectedIds.filter((id) => id !== cardId));
  }

  const updateFixedPosition = useCallback(() => {
    const el = anchorRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    setFixedBox({
      top: r.bottom + 4,
      left: r.left,
      width: Math.max(r.width, 200),
    });
  }, []);

  useLayoutEffect(() => {
    if (!menuOpen || dropdownStrategy !== "fixed") {
      setFixedBox(null);
      return;
    }
    updateFixedPosition();
    window.addEventListener("scroll", updateFixedPosition, true);
    window.addEventListener("resize", updateFixedPosition);
    return () => {
      window.removeEventListener("scroll", updateFixedPosition, true);
      window.removeEventListener("resize", updateFixedPosition);
    };
  }, [menuOpen, dropdownStrategy, updateFixedPosition, query, filtered.length, selectedIds.length]);

  const dropdownClassName =
    "z-[100] max-h-60 overflow-y-auto rounded-md border border-border bg-card py-1 shadow-lg dark:shadow-black/40";

  function renderMenuBody() {
    if (catalog.length === 0) {
      return (
        <li className="px-3 py-2 text-sm text-slate-500 dark:text-slate-400">
          No cards loaded. Check your connection or try again later.
        </li>
      );
    }
    if (available.length === 0 && !q) {
      return (
        <li className="px-3 py-2 text-sm text-slate-500 dark:text-slate-400">
          All catalog cards are already selected.
        </li>
      );
    }
    if (filtered.length === 0) {
      return (
        <li className="px-3 py-2 text-sm text-slate-500 dark:text-slate-400">
          No matching cards
        </li>
      );
    }
    return filtered.map((card) => (
      <li key={card.card_id} role="option">
        <button
          type="button"
          className="w-full px-3 py-2 text-left text-sm text-secondary hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
          onMouseDown={(e) => e.preventDefault()}
          onClick={() => addCard(card.card_id)}
        >
          <span className="font-medium">{card.card_name}</span>
          <span className="block text-xs text-slate-500 dark:text-slate-400">
            {card.issuer}
          </span>
        </button>
      </li>
    ));
  }

  const listbox = menuOpen && (
    <>
      {dropdownStrategy === "absolute" && (
        <ul
          id={`${id}-listbox`}
          role="listbox"
          className={`absolute left-0 right-0 mt-1 w-full ${dropdownClassName}`}
        >
          {renderMenuBody()}
        </ul>
      )}
      {dropdownStrategy === "fixed" &&
        fixedBox &&
        typeof document !== "undefined" &&
        createPortal(
          <ul
            ref={portalListRef}
            id={`${id}-listbox`}
            role="listbox"
            className={dropdownClassName}
            style={{
              position: "fixed",
              top: fixedBox.top,
              left: fixedBox.left,
              width: fixedBox.width,
            }}
          >
            {renderMenuBody()}
          </ul>,
          document.body,
        )}
    </>
  );

  return (
    <div ref={containerRef} className="relative">
      <label
        htmlFor={inputId}
        className="block text-sm font-medium text-secondary mb-2"
      >
        {label}
        {optional && (
          <span className="ml-1 text-slate-400 dark:text-slate-500 font-normal">
            (optional)
          </span>
        )}
      </label>
      {description && (
        <p className="text-sm text-slate-500 dark:text-slate-400 mb-2">
          {description}
        </p>
      )}

      <div ref={anchorRef} className="relative w-full">
        <input
          id={inputId}
          type="text"
          value={query}
          disabled={disabled}
          onChange={(e) => {
            setQuery(e.target.value);
            setOpen(true);
          }}
          onFocus={() => {
            if (!disabled) setOpen(true);
          }}
          placeholder={disabled ? "Loading…" : placeholder}
          autoComplete="off"
          className="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-secondary placeholder:text-slate-400 dark:placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-primary/50 transition-colors disabled:opacity-60"
          aria-expanded={menuOpen}
          aria-controls={`${id}-listbox`}
          aria-autocomplete="list"
          role="combobox"
        />
      </div>

      {listbox}

      {selectedIds.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-2">
          {selectedIds.map((cardId) => {
            const card = byId.get(cardId);
            const title = card?.card_name ?? cardId;
            return (
              <span
                key={cardId}
                className="inline-flex items-center gap-1.5 rounded-full border border-primary/40 bg-primary/10 dark:bg-primary/15 px-2.5 py-1 text-xs font-medium text-secondary"
              >
                <span className="max-w-[200px] truncate sm:max-w-[260px]">
                  {title}
                </span>
                <button
                  type="button"
                  onClick={() => removeCard(cardId)}
                  className="shrink-0 rounded-full p-0.5 text-slate-600 hover:bg-primary/20 hover:text-secondary dark:text-slate-300 dark:hover:bg-primary/25"
                  aria-label={`Remove ${title}`}
                >
                  ×
                </button>
              </span>
            );
          })}
        </div>
      )}

      {error && <p className="mt-1.5 text-xs text-danger">{error}</p>}
    </div>
  );
}
