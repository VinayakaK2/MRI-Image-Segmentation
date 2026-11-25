import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

export type ThemeMode = "light" | "dark" | "system";
export type ResolvedTheme = "light" | "dark";

const THEME_STORAGE_KEY = "cardioai-theme-mode";

/** Theme-invariant anatomical segmentation colors required across all pages. */
export const ANATOMICAL_CLASS_SPEC = [
  { id: 0, code: "BG", label: "Background", colorHex: "#111827", cssVar: "var(--class-bg-color)" },
  { id: 1, code: "RV", label: "RV (Right Ventricle)", colorHex: "#ef4444", cssVar: "var(--class-rv-color)" },
  { id: 2, code: "MYO", label: "MYO (Myocardium)", colorHex: "#22c55e", cssVar: "var(--class-myo-color)" },
  { id: 3, code: "LV", label: "LV (Left Ventricle)", colorHex: "#3b82f6", cssVar: "var(--class-lv-color)" },
] as const;

interface ThemeContextValue {
  mode: ThemeMode;
  resolvedTheme: ResolvedTheme;
  setMode: (nextMode: ThemeMode) => void;
}

const ThemeContext = createContext<ThemeContextValue | undefined>(undefined);

/** Reads persisted theme mode safely from localStorage, defaulting to 'system'. */
function readStoredThemeMode(): ThemeMode {
  try {
    const raw = window.localStorage.getItem(THEME_STORAGE_KEY);
    if (raw === "light" || raw === "dark" || raw === "system") {
      return raw;
    }
  } catch {
    // Fallback to system when storage is blocked
  }
  return "system";
}

/** Detects current OS color scheme preference safely. */
function getSystemTheme(): ResolvedTheme {
  if (typeof window !== "undefined" && typeof window.matchMedia === "function") {
    const mql = window.matchMedia("(prefers-color-scheme: dark)");
    return mql?.matches ? "dark" : "light";
  }
  return "light";
}

/** Applies the active theme class to the root HTML element. */
function applyRootThemeClass(resolved: ResolvedTheme): void {
  const root = document.documentElement;
  if (resolved === "dark") {
    root.classList.add("dark");
  } else {
    root.classList.remove("dark");
  }
}

/** Context provider managing Light, Dark, and System theme modes with persistence. */
export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [mode, setModeState] = useState<ThemeMode>(() => readStoredThemeMode());
  const [systemTheme, setSystemTheme] = useState<ResolvedTheme>(() => getSystemTheme());

  useEffect(() => {
    if (typeof window.matchMedia !== "function") {
      return;
    }
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    if (!mediaQuery || typeof mediaQuery.addEventListener !== "function") {
      return;
    }
    const handleChange = (event: MediaQueryListEvent) => {
      setSystemTheme(event.matches ? "dark" : "light");
    };
    mediaQuery.addEventListener("change", handleChange);
    return () => mediaQuery.removeEventListener("change", handleChange);
  }, []);

  const resolvedTheme: ResolvedTheme = mode === "system" ? systemTheme : mode;

  useEffect(() => {
    applyRootThemeClass(resolvedTheme);
  }, [resolvedTheme]);

  const setMode = useCallback((nextMode: ThemeMode) => {
    setModeState(nextMode);
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, nextMode);
    } catch {
      // Ignore storage persistence failure in restricted contexts
    }
  }, []);

  const value = useMemo<ThemeContextValue>(
    () => ({ mode, resolvedTheme, setMode }),
    [mode, resolvedTheme, setMode]
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

/** Hook returning current theme mode, resolved theme, and setter. */
export function useTheme(): ThemeContextValue {
  const ctx = useContext(ThemeContext);
  if (!ctx) {
    throw new Error("useTheme must be used inside ThemeProvider");
  }
  return ctx;
}
