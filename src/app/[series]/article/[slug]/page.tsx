import { notFound } from "next/navigation";
import Link from "next/link";
import type { Metadata } from "next";
import Header from "@/components/Header";
import ArticleContent from "@/components/ArticleContent";
import {
  SERIES,
  getSeriesById,
  getAllArticles,
  getArticleBySlug,
  getAdjacentArticles,
  getAllSlugsAllSeries,
} from "@/lib/articles";
import ArticleNav from "./ArticleNav";

export function generateStaticParams() {
  return getAllSlugsAllSeries().map(({ slug, seriesId }) => ({
    series: seriesId,
    slug,
  }));
}

export function generateMetadata({
  params,
}: {
  params: { series: string; slug: string };
}): Metadata {
  const article = getArticleBySlug(params.series, params.slug);
  if (!article) return { title: "Not Found" };
  const series = getSeriesById(params.series);
  return {
    title: `${article.title} — ${series?.shortName || ""} Series`,
    description: article.excerpt,
  };
}

export default function ArticlePage({
  params,
}: {
  params: { series: string; slug: string };
}) {
  const series = getSeriesById(params.series);
  if (!series) notFound();

  const article = getArticleBySlug(params.series, params.slug);
  if (!article) notFound();

  const all = getAllArticles(params.series);
  const { prev, next } = getAdjacentArticles(params.series, params.slug);

  return (
    <div className="min-h-screen dark:bg-[#0a0e1a] transition-colors duration-300">
      <Header showBack seriesId={params.series} articleNumber={article.number} total={all.length} />

      <main className="mx-auto max-w-prose px-6 py-10">
        {/* Series breadcrumb */}
        <div className="flex items-center gap-2 mb-6 animate-fade-in">
          <Link
            href="/"
            className="text-xs text-ink-faint hover:text-accent transition-colors dark:text-slate-500 dark:hover:text-blue-400"
          >
            全部系列
          </Link>
          <span className="text-xs text-ink-faint dark:text-slate-600">/</span>
          <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-accent dark:text-blue-400">
            <span className="w-1 h-3 bg-accent rounded-sm dark:bg-blue-400" />
            {series.name}
          </span>
        </div>

        {/* Title card with rounded border */}
        <div className="rounded-2xl border border-line p-6 sm:p-8 mb-8 animate-fade-in bg-surface dark:border-slate-700/60 dark:bg-slate-800/30">
          {/* Category + Tags */}
          <div className="flex items-center gap-2 mb-4 flex-wrap">
            <span className="inline-flex items-center gap-1.5">
              <span className="w-1 h-3.5 bg-accent rounded-sm dark:bg-blue-400" />
              <span className="text-xs font-semibold text-accent tracking-wide uppercase dark:text-blue-400">
                {article.category}
              </span>
            </span>
            {article.tags.map((tag) => (
              <span key={tag} className="tag-pill">
                {tag}
              </span>
            ))}
          </div>

          {/* Title */}
          <h1 className="text-3xl sm:text-4xl font-bold text-ink leading-tight mb-4 tracking-tight dark:text-slate-100">
            {article.title}
          </h1>

          {/* Meta info */}
          <div className="flex items-center gap-2 text-sm text-ink-muted mb-3 flex-wrap dark:text-slate-400">
            <span className="font-mono text-accent dark:text-blue-400">
              {series.shortName}-{String(article.number).padStart(2, "0")}
            </span>
            <span className="text-ink-faint dark:text-slate-600">·</span>
            <span>{article.readTime} min read</span>
            <span className="text-ink-faint dark:text-slate-600">·</span>
            <span>{article.wordCount} 字</span>
          </div>

          {/* Blue separator */}
          <div className="w-16 h-0.5 bg-accent dark:bg-blue-500" />

          {/* Metadata box */}
          {article.metaLines.length > 0 && (
            <div className="mt-4 space-y-1.5">
              {article.metaLines.map((line, i) => (
                <p key={i} className="text-xs text-ink-muted leading-relaxed dark:text-slate-400">
                  {line}
                </p>
              ))}
            </div>
          )}
        </div>

        {/* Article content */}
        <ArticleContent content={article.content} />

        {/* Prev / Next navigation */}
        <ArticleNav prev={prev} next={next} seriesId={params.series} />
      </main>
    </div>
  );
}
