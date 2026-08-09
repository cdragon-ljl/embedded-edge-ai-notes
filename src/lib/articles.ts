import fs from "fs";
import path from "path";
import { SERIES, getSeriesById } from "./series";
import type { SeriesConfig } from "./series";

/* Re-export for convenience */
export { SERIES, getSeriesById } from "./series";
export type { SeriesConfig } from "./series";

/* ---------- Article types ---------- */

export interface ArticleMeta {
  slug: string;
  series: string;
  seriesId: string;
  number: number;
  title: string;
  seriesName: string;
  excerpt: string;
  prerequisites: string | null;
  env: string | null;
  readerProfile: string | null;
  category: string;
  tags: string[];
  wordCount: number;
  readTime: number;
}

export interface ArticleFull extends ArticleMeta {
  content: string;
  metaLines: string[];
}

/* ---------- Parsing helpers ---------- */

function extractNumber(fileName: string, prefix: string): number {
  const escaped = prefix.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const m = fileName.match(new RegExp(escaped + "(\\d+)", "i"));
  return m ? parseInt(m[1], 10) : 0;
}

function cleanSlug(fileName: string): string {
  return fileName.replace(/\(\d+\)/, "").replace(/\.md$/i, "").trim();
}

function generateExcerpt(content: string, maxLen = 120): string {
  const text = content
    .replace(/^#+\s+.*$/gm, "")
    .replace(/^>.*$/gm, "")
    .replace(/```[\s\S]*?```/g, "")
    .replace(/\$\$[\s\S]*?\$\$/g, "")
    .replace(/\$[^$]+\$/g, "")
    .replace(/[*_`~|]/g, "")
    .replace(/\n+/g, " ")
    .trim();
  return text.length > maxLen ? text.slice(0, maxLen) + "..." : text;
}

function deriveCategoryAndTags(
  series: SeriesConfig,
  number: number,
  title: string
): { category: string; tags: string[] } {
  let category = "其他";
  for (const c of series.categories) {
    if (number >= c.range[0] && number <= c.range[1]) {
      category = c.name;
      break;
    }
  }

  const tags = new Set<string>();
  const lower = title.toLowerCase();

  for (const [kw, tag] of Object.entries(series.keywords)) {
    if (lower.includes(kw)) tags.add(tag);
  }

  if (tags.size === 0) tags.add(series.shortName);

  return { category, tags: Array.from(tags) };
}

function parseArticle(
  fileName: string,
  raw: string,
  series: SeriesConfig
): { meta: ArticleMeta; content: string; metaLines: string[] } {
  const lines = raw.split("\n");

  let title = "";
  let titleIndex = -1;
  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(/^#\s+(.+)$/);
    if (m) {
      title = m[1].trim();
      titleIndex = i;
      break;
    }
  }

  const metaLines: string[] = [];
  let contentStart = titleIndex + 1;
  for (let i = titleIndex + 1; i < lines.length; i++) {
    const trimmed = lines[i].trim();
    if (trimmed.startsWith(">")) {
      metaLines.push(trimmed.replace(/^>\s?/, ""));
      contentStart = i + 1;
    } else if (trimmed === "" && metaLines.length > 0) {
      contentStart = i + 1;
    } else if (trimmed !== "") {
      break;
    }
  }

  const content = lines.slice(contentStart).join("\n").trim();

  const seriesLine = metaLines.find((l) => l.includes("系列")) || "";
  const prereqLine = metaLines.find((l) => l.includes("前置")) || null;
  const envLine = metaLines.find((l) => l.includes("配套")) || null;
  const readerLine = metaLines.find((l) => l.includes("读者画像") || l.includes("定位")) || null;

  const number = extractNumber(fileName, series.filePrefix);
  const slug = cleanSlug(fileName);
  const { category, tags } = deriveCategoryAndTags(series, number, title);

  const wordCount = content.replace(/\s/g, "").length;
  const readTime = Math.max(1, Math.ceil(wordCount / 500));

  const meta: ArticleMeta = {
    slug,
    series: seriesLine,
    seriesId: series.id,
    number,
    title,
    seriesName: series.name,
    excerpt: generateExcerpt(content),
    prerequisites: prereqLine,
    env: envLine,
    readerProfile: readerLine,
    category,
    tags,
    wordCount,
    readTime,
  };

  return { meta, content, metaLines };
}

/* ---------- Public API ---------- */

export function getAllArticles(seriesId: string): ArticleMeta[] {
  const series = getSeriesById(seriesId);
  if (!series) return [];

  const dir = path.join(process.cwd(), series.dir);
  if (!fs.existsSync(dir)) return [];

  const files = fs.readdirSync(dir).filter((f) => /\.md$/i.test(f));
  const articles: ArticleMeta[] = [];

  for (const file of files) {
    const raw = fs.readFileSync(path.join(dir, file), "utf8");
    const { meta } = parseArticle(file, raw, series);
    articles.push(meta);
  }

  return articles.sort((a, b) => a.number - b.number);
}

export function getAllArticlesAllSeries(): { seriesId: string; articles: ArticleMeta[] }[] {
  return SERIES.map((s) => ({
    seriesId: s.id,
    articles: getAllArticles(s.id),
  }));
}

export function getArticleBySlug(seriesId: string, slug: string): ArticleFull | null {
  const series = getSeriesById(seriesId);
  if (!series) return null;

  const dir = path.join(process.cwd(), series.dir);
  if (!fs.existsSync(dir)) return null;

  const files = fs.readdirSync(dir).filter((f) => /\.md$/i.test(f));
  const targetFile = files.find((f) => cleanSlug(f) === slug);
  if (!targetFile) return null;

  const raw = fs.readFileSync(path.join(dir, targetFile), "utf8");
  const { meta, content, metaLines } = parseArticle(targetFile, raw, series);

  return { ...meta, content, metaLines };
}

export function getAdjacentArticles(
  seriesId: string,
  slug: string
): { prev: ArticleMeta | null; next: ArticleMeta | null } {
  const all = getAllArticles(seriesId);
  const idx = all.findIndex((a) => a.slug === slug);
  if (idx === -1) return { prev: null, next: null };
  return {
    prev: idx > 0 ? all[idx - 1] : null,
    next: idx < all.length - 1 ? all[idx + 1] : null,
  };
}

export function getAllSlugs(seriesId: string): { slug: string; seriesId: string }[] {
  return getAllArticles(seriesId).map((a) => ({ slug: a.slug, seriesId }));
}

export function getAllSlugsAllSeries(): { slug: string; seriesId: string }[] {
  const result: { slug: string; seriesId: string }[] = [];
  for (const s of SERIES) {
    result.push(...getAllSlugs(s.id));
  }
  return result;
}
