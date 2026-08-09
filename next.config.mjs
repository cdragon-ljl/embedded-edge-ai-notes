/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export",
  basePath: "/embedded-edge-ai-notes",
  trailingSlash: true,
  images: {
    unoptimized: true,
  },
  reactStrictMode: true,
  // Skip file tracing to avoid sandbox safe-delete hangs during static export
  experimental: {
    outputFileTracingExcludes: {
      "*": ["**/*"],
    },
  },
};

export default nextConfig;
