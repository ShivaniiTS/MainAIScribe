"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  FileText,
  Users,
  Mic,
  Activity,
} from "lucide-react";

interface NavItem {
  href: string;
  label: string;
  icon: React.ComponentType<{ size?: number }>;
}

const NAV: NavItem[] = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/samples", label: "Encounters", icon: FileText },
  { href: "/providers", label: "Providers", icon: Users },
  { href: "/capture", label: "Capture", icon: Mic },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside
      className="flex flex-col w-56 min-h-screen flex-shrink-0"
      style={{ background: "var(--sidebar-bg)" }}
    >
      {/* Practice branding — customizable per deployment */}
      <div className="flex items-center gap-3 px-5 py-6 border-b border-white/10">
        <div
          className="w-9 h-9 rounded-lg flex items-center justify-center text-white font-bold text-sm flex-shrink-0"
          style={{ background: "var(--brand-primary)" }}
        >
          AI
        </div>
        <div>
          <div className="text-white font-semibold text-sm leading-tight">
            AI Scribe
          </div>
          <div className="text-white/40 text-xs">Provider Portal</div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-4 space-y-1">
        {NAV.map(({ href, label, icon: Icon }) => {
          const active =
            href === "/" ? pathname === "/" : pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors"
              style={{
                background: active ? "var(--sidebar-active)" : "transparent",
                color: active ? "#FFFFFF" : "rgba(255,255,255,0.6)",
              }}
              onMouseEnter={(e) => {
                if (!active)
                  (e.currentTarget as HTMLElement).style.background =
                    "var(--sidebar-hover)";
              }}
              onMouseLeave={(e) => {
                if (!active)
                  (e.currentTarget as HTMLElement).style.background =
                    "transparent";
              }}
            >
              <Icon size={16} />
              {label}
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="px-5 py-4 border-t border-white/10">
        <div className="flex items-center gap-2">
          <Activity size={12} className="text-green-400" />
          <span className="text-white/40 text-xs">AI Scribe v1.0</span>
        </div>
      </div>
    </aside>
  );
}
