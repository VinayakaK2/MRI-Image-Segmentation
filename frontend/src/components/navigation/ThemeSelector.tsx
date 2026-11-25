import { Monitor, Moon, Sun } from "lucide-react";
import { useTheme, type ThemeMode } from "@/theme/theme-context";

const THEME_OPTIONS: { value: ThemeMode; label: string; icon: typeof Sun }[] = [
  { value: "light", label: "Light", icon: Sun },
  { value: "dark", label: "Dark", icon: Moon },
  { value: "system", label: "System", icon: Monitor },
];

/** Accessible 3-way theme mode selector (Light / Dark / System) persisted in localStorage. */
export function ThemeSelector() {
  const { mode, setMode } = useTheme();

  return (
    <div
      role="group"
      aria-label="Theme selector"
      className="inline-flex items-center rounded-xl border border-border-subtle bg-surface-secondary p-0.5"
    >
      {THEME_OPTIONS.map((option) => {
        const Icon = option.icon;
        const active = mode === option.value;
        return (
          <button
            key={option.value}
            type="button"
            aria-pressed={active}
            onClick={() => setMode(option.value)}
            title={`${option.label} theme`}
            className={[
              "inline-flex items-center gap-1.5 rounded-[10px] px-2 py-1 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent sm:px-2.5 sm:py-1.5",
              active
                ? "bg-surface text-ink shadow-sm"
                : "text-ink-secondary hover:text-ink",
            ].join(" ")}
          >
            <Icon className="h-3.5 w-3.5" aria-hidden="true" />
            <span className="hidden sm:inline">{option.label}</span>
          </button>
        );
      })}
    </div>
  );
}
