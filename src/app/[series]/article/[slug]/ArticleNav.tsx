"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import type { ArticleMeta } from "@/lib/articles";

export default function ArticleNav({
  prev,
  next,
  seriesId,
}: {
  prev: ArticleMeta | null;
  next: ArticleMeta | null;
  seriesId: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true);
          observer.unobserve(el);
        }
      },
      { threshold: 0.2 }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return (
    <div
      ref={ref}
      className="mt-16 pt-6 border-t border-line dark:border-slate-800 grid grid-cols-2 gap-4"
      style={{
        opacity: visible ? 1 : 0,
        transform: visible ? "translateY(0)" : "translateY(12px)",
        transition: "all 0.5s cubic-bezier(0.16, 1, 0.3, 1)",
      }}
    >
      {prev ? (
        <Link
          href={`/${seriesId}/article/${prev.slug}`}
          className="group rounded-xl p-4 hover:bg-surface-alt transition-all duration-200 hover:-translate-x-1 dark:hover:bg-slate-800/60"
        >
          <p className="text-xs text-ink-faint mb-1.5 dark:text-slate-500">&larr; 上一篇</p>
          <p className="text-sm font-medium text-ink group-hover:text-accent transition-colors line-clamp-1 dark:text-slate-300 dark:group-hover:text-blue-400">
            {prev.title}
          </p>
        </Link>
      ) : (
        <div />
      )}
      {next ? (
        <Link
          href={`/${seriesId}/article/${next.slug}`}
          className="group rounded-xl p-4 hover:bg-surface-alt transition-all duration-200 hover:translate-x-1 text-right dark:hover:bg-slate-800/60"
        >
          <p className="text-xs text-ink-faint mb-1.5 dark:text-slate-500">下一篇 &rarr;</p>
          <p className="text-sm font-medium text-ink group-hover:text-accent transition-colors line-clamp-1 dark:text-slate-300 dark:group-hover:text-blue-400">
            {next.title}
          </p>
        </Link>
      ) : (
        <div />
      )}

      {/* Back to list */}
      <div className="mt-4 text-center col-span-2">
        <Link
          href="/"
          className="inline-flex items-center gap-1.5 text-sm text-ink-muted hover:text-accent transition-colors dark:text-slate-400 dark:hover:text-blue-400"
        >
          <span>&larr;</span>
          <span>返回文章列表</span>
        </Link>
      </div>
    </div>
  );
}
