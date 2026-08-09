import Header from "@/components/Header";
import ArticleListClient from "@/components/ArticleListClient";
import { SERIES, getAllArticlesAllSeries } from "@/lib/articles";

export default function HomePage() {
  const allData = getAllArticlesAllSeries();
  const totalArticles = allData.reduce((sum, d) => sum + d.articles.length, 0);

  return (
    <div className="min-h-screen dark:bg-[#0a0e1a] transition-colors duration-300">
      <Header />

      <main className="mx-auto max-w-3xl px-6 py-12">
        {/* Hero */}
        <div className="mb-10 animate-fade-in">
          <div className="flex items-center gap-2 mb-4">
            <span className="w-1 h-5 bg-accent rounded-sm" />
            <span className="text-xs font-semibold text-accent tracking-wider uppercase dark:text-blue-400">
              技术专栏
            </span>
          </div>
          <h1 className="text-4xl sm:text-5xl font-bold text-ink mb-3 tracking-tight leading-tight dark:text-slate-100">
            嵌入式 AI 全栈笔记
          </h1>
          <p className="text-base text-ink-muted leading-relaxed max-w-xl dark:text-slate-400">
            从 GPU 并行编程到 NPU 端侧部署——两条技术路径，覆盖 CUDA 算子开发、RKNN 模型转换量化、板端推理与性能优化。
            共 {totalArticles} 篇实战文章。
          </p>
        </div>

        {/* Article list with series navigation */}
        <ArticleListClient seriesList={SERIES} allData={allData} />
      </main>

      <footer className="border-t border-line mt-20 dark:border-slate-800">
        <div className="mx-auto max-w-3xl px-6 py-6 flex items-center justify-between">
          <p className="text-xs text-ink-faint dark:text-slate-500">
            嵌入式 AI 全栈笔记
          </p>
          <p className="text-xs text-ink-faint font-mono dark:text-slate-600">
            {totalArticles} articles · {SERIES.length} series
          </p>
        </div>
      </footer>
    </div>
  );
}
