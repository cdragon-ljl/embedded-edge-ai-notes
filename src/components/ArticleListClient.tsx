"use client";

import { useState, useMemo, useEffect, useRef } from "react";
import Link from "next/link";
import type { ArticleMeta } from "@/lib/articles";
import type { SeriesConfig } from "@/lib/series";

function ArticleCard({ article, index }: { article: ArticleMeta; index: number }) {
  const ref = useRef<HTMLAnchorElement>(null);
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
      { threshold: 0.1, rootMargin: "0px 0px -20px 0px" }
    );

    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return (
    <Link
      ref={ref}
      href={`/${article.seriesId}/article/${article.slug}`}
      className="article-card dark:hover:bg-slate-800/60 dark:hover:border-slate-700/50 group"
      style={{
        opacity: visible ? 1 : 0,
        transform: visible ? "translateY(0)" : "translateY(12px)",
        transition: `all 0.4s cubic-bezier(0.16, 1, 0.3, 1) ${index * 0.05}s`,
      }}
    >
      <div className="flex items-start gap-3">
        <span className="text-xs font-mono text-ink-faint mt-0.5 tabular-nums dark:text-slate-500">
          {String(article.number).padStart(2, "0")}
        </span>
        <div className="flex-1 min-w-0">
          <h3 className="text-base font-semibold text-ink leading-snug mb-1.5 group-hover:text-accent transition-colors duration-200 dark:text-slate-200 dark:group-hover:text-blue-400">
            {article.title}
          </h3>
          <p className="text-sm text-ink-muted leading-relaxed mb-2.5 line-clamp-2 dark:text-slate-400">
            {article.excerpt}
          </p>
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-xs text-ink-faint dark:text-slate-500">{article.readTime} min</span>
            <span className="text-xs text-ink-faint dark:text-slate-600">·</span>
            {article.tags.slice(0, 3).map((tag) => (
              <span key={tag} className="tag-pill">
                {tag}
              </span>
            ))}
          </div>
        </div>
      </div>
    </Link>
  );
}

export default function ArticleListClient({
  seriesList,
  allData,
}: {
  seriesList: SeriesConfig[];
  allData: { seriesId: string; articles: ArticleMeta[] }[];
}) {
  const [activeSeries, setActiveSeries] = useState(seriesList[0]?.id || "");
  const [query, setQuery] = useState("");

  const currentArticles = useMemo(() => {
    const found = allData.find((d) => d.seriesId === activeSeries);
    return found ? found.articles : [];
  }, [allData, activeSeries]);

  const currentSeries = useMemo(
    () => seriesList.find((s) => s.id === activeSeries),
    [seriesList, activeSeries]
  );

  const filtered = useMemo(() => {
    if (!query.trim()) return currentArticles;
    const q = query.toLowerCase();
    return currentArticles.filter(
      (a) =>
        a.title.toLowerCase().includes(q) ||
        a.excerpt.toLowerCase().includes(q) ||
        a.tags.some((t) => t.toLowerCase().includes(q)) ||
        a.category.toLowerCase().includes(q)
    );
  }, [currentArticles, query]);

  const grouped = useMemo(() => {
    const groups: Record<string, ArticleMeta[]> = {};
    for (const a of filtered) {
      if (!groups[a.category]) groups[a.category] = [];
      groups[a.category].push(a);
    }
    return groups;
  }, [filtered]);

  const categoryOrder = currentSeries?.categories.map((c) => c.name) || [];

  return (
    <div>
      {/* Series navigation tabs */}
      <div className="flex items-center gap-1 mb-6 p-1 bg-surface-alt rounded-xl dark:bg-slate-800/50">
        {seriesList.map((s) => (
          <button
            key={s.id}
            onClick={() => {
              setActiveSeries(s.id);
              setQuery("");
            }}
            className={`flex-1 px-4 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 ${
              activeSeries === s.id
                ? "bg-surface text-ink shadow-sm dark:bg-slate-700 dark:text-slate-100"
                : "text-ink-muted hover:text-ink hover:bg-surface/60 dark:text-slate-400 dark:hover:text-slate-200 dark:hover:bg-slate-700/40"
            }`}
          >
            <span className="flex items-center justify-center gap-2">
              <span className={`w-1.5 h-1.5 rounded-full transition-colors ${activeSeries === s.id ? "bg-accent" : "bg-ink-faint dark:bg-slate-600"}`} />
              {s.shortName}
              <span className="text-xs opacity-60">
                {allData.find((d) => d.seriesId === s.id)?.articles.length || 0}
              </span>
            </span>
          </button>
        ))}
      </div>

      {/* Search */}
      <div className="mb-8">
        <div className="relative group/search">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="搜索文章..."
            className="w-full bg-surface-alt rounded-xl px-4 py-3 text-sm text-ink placeholder:text-ink-faint border border-transparent focus:border-accent focus:outline-none transition-all duration-200 dark:bg-slate-800/50 dark:text-slate-200 dark:placeholder:text-slate-500 dark:focus:border-blue-500"
          />
          <span className="absolute right-3.5 top-1/2 -translate-y-1/2 text-ink-faint text-sm dark:text-slate-500">
            {query ? `${filtered.length} 篇` : "⌕"}
          </span>
        </div>
      </div>

      {/* Article list grouped by category */}
      <div className="space-y-10">
        {categoryOrder.map((cat) => {
          const items = grouped[cat];
          if (!items || items.length === 0) return null;
          return (
            <section key={cat}>
              <div className="flex items-center gap-2 mb-4 px-1">
                <span className="w-1 h-4 bg-accent rounded-sm" />
                <h2 className="text-xs font-semibold text-ink-muted tracking-wide uppercase dark:text-slate-400">
                  {cat}
                </h2>
                <span className="text-xs text-ink-faint dark:text-slate-600">{items.length}</span>
              </div>
              <div className="space-y-1">
                {items.map((article, i) => (
                  <ArticleCard
                    key={article.slug}
                    article={article}
                    index={i}
                  />
                ))}
              </div>
            </section>
          );
        })}

        {filtered.length === 0 && (
          <div className="text-center py-16 animate-fade-in">
            <p className="text-sm text-ink-faint dark:text-slate-500">没有找到匹配的文章</p>
          </div>
        )}
      </div>
    </div>
  );
}
