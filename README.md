# 嵌入式 AI 全栈笔记

> 从 GPU 并行编程到 NPU 端侧部署 —— 两条技术路径，27 篇实战文章

线上地址：**https://cdragon-ljl.github.io/embedded-edge-ai-notes/**

## 项目简介

一个纯静态技术文章展示网站，包含两个系列：

- **CUDA · NPU 系列**（16 篇）—— 从 CUDA 并行编程基础出发，经算子优化、深度学习框架实战，最终落地 NPU 算子开发与端侧 AI 部署
- **RKNN 端侧部署**（11 篇）—— 基于 RV1126 平台的模型转换、INT8 量化、板端推理到性能调优全流程实战

## 功能特性

- **极简极客风设计** —— 白底蓝色点缀，无衬线字体，克制留白
- **暗色模式** —— 一键切换，偏好持久化到 localStorage，防闪烁加载
- **卡片入场动画** —— IntersectionObserver 驱动的错峰淡入 + hover 微交互
- **多系列导航** —— 首页 Tab 切换 CUDA / RKNN，面包屑导航回溯
- **Markdown 全量渲染** —— GFM 表格、代码语法高亮（Highlight.js）、LaTeX 数学公式（KaTeX）
- **自动元数据提取** —— 从 `#` 标题和 `>` 引用块解析文章信息，无需 frontmatter
- **全文搜索** —— 标题 + 摘要实时过滤
- **静态导出** —— 纯 HTML/CSS/JS，GitHub Pages 一键部署

## 技术栈

| 类别 | 技术 |
|------|------|
| 框架 | Next.js 14 (App Router, SSG) |
| 语言 | TypeScript |
| 样式 | Tailwind CSS + @tailwindcss/typography |
| Markdown | react-markdown + remark-gfm + remark-math |
| 代码高亮 | rehype-highlight (Highlight.js) |
| 数学公式 | rehype-katex + KaTeX |
| 部署 | GitHub Pages + GitHub Actions |

## 项目结构

```
├── cuda/                      # CUDA · NPU 系列文章 (16 篇)
├── rknn/                      # RKNN 端侧部署系列文章 (11 篇)
├── src/
│   ├── app/
│   │   ├── layout.tsx         # 全局布局 + 字体 + 主题
│   │   ├── page.tsx           # 首页（系列导航 + 文章列表）
│   │   ├── globals.css        # 设计系统样式
│   │   └── [series]/article/[slug]/
│   │       ├── page.tsx       # 文章详情页
│   │       └── ArticleNav.tsx # 上下篇导航
│   ├── components/
│   │   ├── Header.tsx         # 顶部导航 + 主题切换
│   │   ├── ThemeProvider.tsx  # 暗色模式 Provider
│   │   ├── ArticleListClient.tsx  # 搜索 + 系列切换 + 卡片列表
│   │   └── ArticleContent.tsx     # Markdown 渲染组件
│   └── lib/
│       ├── series.ts          # 系列配置（纯数据，客户端安全）
│       └── articles.ts        # 文章解析（读 fs，仅服务端）
├── .github/workflows/deploy.yml  # GitHub Actions 自动部署
├── next.config.mjs            # 静态导出 + basePath 配置
├── tailwind.config.ts         # 设计系统配色定义
└── package.json
```

## 本地开发

```bash
# 安装依赖
npm install

# 开发模式
npm run dev

# 生产构建
npm run build

# 启动生产服务器
npm start
```

## 文章目录

### CUDA · NPU 系列（16 篇）

| 阶段 | 篇目 | 内容 |
|------|------|------|
| CUDA 基础 | 01-06 | 环境搭建、内存模型、线程组织、矩阵乘法、并行归约 |
| 深度学习与部署 | 07-11 | Nsight 分析、PyTorch/TensorFlow 实战、ONNX/TensorRT 部署 |
| 进阶优化 | 12-16 | Roofline 模型、算子融合、AI 编译器、综合项目、性能报告 |

### RKNN 端侧部署（11 篇）

| 阶段 | 篇目 | 内容 |
|------|------|------|
| RKNN 工具链 | 01-04 | 平台总览、环境搭建、模型转换、INT8 量化 |
| 板端推理部署 | 05-07 | C API 推理、Python-Lite、摄像头集成 |
| 进阶实战 | 08-11 | YOLO 检测、后处理优化、性能调优、完整项目 |

## 部署

网站通过 GitHub Actions 自动部署到 GitHub Pages：

1. 推送代码到 `master` 分支
2. Actions 自动执行：安装依赖 → 构建 → 静态导出 → 部署
3. 1-2 分钟后线上更新

新增文章只需将 `.md` 文件放入 `cuda/` 或 `rknn/` 目录，推送即可。

## License

MIT
