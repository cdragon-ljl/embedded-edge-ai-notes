/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export",
  basePath: "/embedded-edge-ai-notes",
  trailingSlash: true,
  images: {
    unoptimized: true,
  },
  reactStrictMode: true,
};

export default nextConfig;
