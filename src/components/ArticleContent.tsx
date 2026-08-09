import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";
import type { Components } from "react-markdown";

const components: Components = {
  pre({ children, ...props }) {
    const child = Array.isArray(children) ? children[0] : children;
    const childProps = (child as React.ReactElement)?.props || {};
    const className: string = childProps.className || "";
    const langMatch = className.match(/language-(\w+)/);
    const lang = langMatch ? langMatch[1] : "";

    return (
      <div className="relative">
        {lang && (
          <div className="text-xs text-ink-faint font-mono px-1 mb-1.5 uppercase tracking-wide">
            {lang}
          </div>
        )}
        <pre {...props}>{children}</pre>
      </div>
    );
  },
  table({ children, ...props }) {
    return (
      <div className="overflow-x-auto my-5 rounded-lg border border-line">
        <table {...props}>{children}</table>
      </div>
    );
  },
  a({ children, href, ...props }) {
    const external = href?.startsWith("http");
    return (
      <a
        href={href}
        {...(external ? { target: "_blank", rel: "noopener noreferrer" } : {})}
        {...props}
      >
        {children}
      </a>
    );
  },
};

export default function ArticleContent({ content }: { content: string }) {
  return (
    <div className="article-content">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex, rehypeHighlight]}
        components={components}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
