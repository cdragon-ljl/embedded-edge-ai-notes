import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";
import type { Components } from "react-markdown";

const BASE_PATH = "/embedded-edge-ai-notes";

function makeImgComponent(seriesId: string) {
  return function Img({ src, alt, ...props }: React.ImgHTMLAttributes<HTMLImageElement>) {
    let resolvedSrc = src;
    if (src && src.startsWith("./images/")) {
      const filename = src.replace("./images/", "");
      resolvedSrc = `${BASE_PATH}/images/${seriesId}/${filename}`;
    }
    return <img src={resolvedSrc} alt={alt || ""} {...props} />;
  };
}

function createComponents(seriesId: string): Components {
  return {
    pre({ children }) {
      const child = Array.isArray(children) ? children[0] : children;
      const childProps = (child as React.ReactElement)?.props || {};
      const className: string = childProps.className || "";
      const langMatch = className.match(/language-(\w+)/);
      const lang = langMatch ? langMatch[1] : "";

      return (
        <div className="relative">
          {lang && (
            <div className="text-xs text-ink-faint font-mono px-1 mb-1.5 uppercase tracking-wide dark:text-slate-500">
              {lang}
            </div>
          )}
          <pre>{children}</pre>
        </div>
      );
    },
    table({ children }) {
      return (
        <div className="overflow-x-auto my-5 rounded-lg border border-line">
          <table>{children}</table>
        </div>
      );
    },
    a({ children, href }) {
      const external = href?.startsWith("http");
      return (
        <a
          href={href}
          {...(external ? { target: "_blank", rel: "noopener noreferrer" } : {})}
        >
          {children}
        </a>
      );
    },
    img: makeImgComponent(seriesId),
  };
}

export default function ArticleContent({
  content,
  seriesId,
}: {
  content: string;
  seriesId: string;
}) {
  return (
    <div className="article-content">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex, rehypeHighlight]}
        components={createComponents(seriesId)}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
