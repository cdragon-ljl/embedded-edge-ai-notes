"use client";

import Link from "next/link";
import { useTheme } from "@/components/ThemeProvider";
import { SERIES } from "@/lib/series";

export default function Header({
  showBack = false,
  seriesId,
  articleNumber,
  total,
}: {
  showBack?: boolean;
  seriesId?: string;
  articleNumber?: number;
  total?: number;
}) {
  const { theme, toggle } = useTheme();

  const currentSeries = seriesId ? SERIES.find((s) => s.id === seriesId) : null;
  const totalCount = SERIES.reduce((sum, s) => sum + s.totalCount, 0);

  return (
    <header className="border-b border-line bg-surface sticky top-0 z-50 dark:border-slate-800 dark:bg-[#0a0e1a]">
      <div className="mx-auto max-w-5xl px-6 h-14 flex items-center justify-between">
        {showBack ? (
          <Link
            href="/"
            className="text-sm text-ink-muted hover:text-accent transition-colors flex items-center gap-1.5 dark:text-slate-400 dark:hover:text-blue-400"
          >
            <span className="text-base">&larr;</span>
            <span>{currentSeries ? currentSeries.shortName + " Series" : "全部系列"}</span>
          </Link>
        ) : (
          <Link href="/" className="flex items-center gap-2">
            <span className="w-1 h-4 bg-accent rounded-sm" />
            <span className="text-sm font-semibold text-ink tracking-tight dark:text-slate-200">
              嵌入式全栈笔记
            </span>
          </Link>
        )}

        <div className="flex items-center gap-3">
          {articleNumber !== undefined && total !== undefined && (
            <span className="text-xs text-ink-faint font-mono dark:text-slate-500">
              {String(articleNumber).padStart(2, "0")} / {String(total).padStart(2, "0")}
            </span>
          )}
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-accent-light text-accent dark:bg-blue-900/30 dark:text-blue-400">
            {totalCount} articles
          </span>

          {/* Theme toggle */}
          <button
            onClick={toggle}
            className="theme-toggle"
            aria-label="Toggle dark mode"
            title={theme === "light" ? "切换到暗色模式" : "切换到亮色模式"}
          />
        </div>
      </div>
    </header>
  );
}
