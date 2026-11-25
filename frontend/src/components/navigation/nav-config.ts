import {
  Activity,
  BarChart3,
  FileText,
  LayoutDashboard,
  Settings,
  ShieldAlert,
  Wrench,
  type LucideIcon,
} from "lucide-react";

export interface NavItemConfig {
  id: string;
  label: string;
  path: string;
  icon: LucideIcon;
  implemented: boolean;
  description: string;
}

/** Primary WORKSPACE navigation routes matching the CardioAI Research architecture. */
export const WORKSPACE_NAV_ITEMS: NavItemConfig[] = [
  {
    id: "overview",
    label: "Overview",
    path: "/overview",
    icon: LayoutDashboard,
    implemented: true,
    description: "Research summary, baseline benchmarks, and verified Phase 4/5 findings.",
  },
  {
    id: "segmentation",
    label: "Segmentation",
    path: "/segmentation",
    icon: Activity,
    implemented: false,
    description: "Interactive 2D slice inference and mask overlay studio (planned next phase).",
  },
  {
    id: "robustness",
    label: "Robustness",
    path: "/robustness",
    icon: ShieldAlert,
    implemented: false,
    description: "Detailed Phase 4 corruption exploration and slice comparison (planned next phase).",
  },
  {
    id: "mitigation",
    label: "Mitigation",
    path: "/mitigation",
    icon: Wrench,
    implemented: false,
    description: "Detailed Phase 5 intervention ablation workbench (planned next phase).",
  },
  {
    id: "analytics",
    label: "Analytics",
    path: "/analytics",
    icon: BarChart3,
    implemented: false,
    description: "Statistical deep-dive across cohorts and anatomical classes (planned next phase).",
  },
  {
    id: "reports",
    label: "Reports",
    path: "/reports",
    icon: FileText,
    implemented: false,
    description: "Research report viewer (no backend export endpoint currently exists).",
  },
];

/** Secondary utility route pinned to the sidebar footer. */
export const SETTINGS_NAV_ITEM: NavItemConfig = {
  id: "settings",
  label: "Settings",
  path: "/settings",
  icon: Settings,
  implemented: false,
  description: "Workstation preferences and environment diagnostics (planned next phase).",
};

/** Resolves route metadata for header breadcrumbs and document title updates. */
export function findNavItemByPath(pathname: string): NavItemConfig {
  const allItems = [...WORKSPACE_NAV_ITEMS, SETTINGS_NAV_ITEM];
  return (
    allItems.find((item) => pathname === item.path || pathname.startsWith(`${item.path}/`)) ??
    WORKSPACE_NAV_ITEMS[0]
  );
}
