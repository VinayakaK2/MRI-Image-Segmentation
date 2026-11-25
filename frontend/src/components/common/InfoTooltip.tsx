import { Info } from "lucide-react";
import { useId, useState } from "react";

interface InfoTooltipProps {
  label: string;
  content: string;
}

/** Accessible hover/focus/click tooltip for technical metrics and statistical definitions. */
export function InfoTooltip({ label, content }: InfoTooltipProps) {
  const [open, setOpen] = useState(false);
  const tooltipId = useId();

  return (
    <span
      className="relative inline-flex items-center"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      <button
        type="button"
        aria-label={label}
        aria-describedby={open ? tooltipId : undefined}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onClick={() => setOpen((prev) => !prev)}
        className="inline-flex h-5 w-5 items-center justify-center rounded-full text-ink-secondary transition-colors hover:bg-surface-secondary hover:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
      >
        <Info className="h-3.5 w-3.5" aria-hidden="true" />
      </button>
      {open && (
        <span
          id={tooltipId}
          role="tooltip"
          className="absolute bottom-full right-0 z-40 mb-2 w-64 rounded-xl border border-border-subtle bg-surface p-3 text-xs leading-relaxed text-ink shadow-lg"
        >
          {content}
        </span>
      )}
    </span>
  );
}
