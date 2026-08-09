export interface SeriesConfig {
  id: string;
  name: string;
  shortName: string;
  dir: string;
  description: string;
  categories: { name: string; range: [number, number] }[];
  keywords: Record<string, string>;
  filePrefix: string;
  totalCount: number;
}

export const SERIES: SeriesConfig[] = [
  {
    id: "cuda",
    name: "CUDA · NPU 系列",
    shortName: "CUDA",
    dir: "cuda",
    description: "从 CUDA 出发，通往 NPU 算子开发与端侧 AI 部署的完整学习路径",
    categories: [
      { name: "CUDA 基础", range: [1, 6] },
      { name: "深度学习与部署", range: [7, 11] },
      { name: "进阶优化", range: [12, 99] },
    ],
    keywords: {
      cuda: "CUDA",
      gpu: "GPU",
      memory: "内存",
      thread: "线程",
      matmul: "矩阵乘法",
      matrix: "矩阵乘法",
      reduction: "归约",
      nsight: "性能分析",
      pytorch: "PyTorch",
      cnn: "CNN",
      onnx: "ONNX",
      tensorflow: "TensorFlow",
      tensorrt: "TensorRT",
      roofline: "Roofline",
      fusion: "算子融合",
      compiler: "编译器",
      tvm: "TVM",
      triton: "Triton",
      inference: "推理",
      benchmark: "实践",
      "deep learning": "深度学习",
      conv: "卷积",
    },
    filePrefix: "npu-",
    totalCount: 16,
  },
  {
    id: "rknn",
    name: "RKNN 端侧部署",
    shortName: "RKNN",
    dir: "rknn",
    description: "基于 RV1126 的模型转换、量化、板端推理到性能调优实战",
    categories: [
      { name: "RKNN 工具链", range: [1, 4] },
      { name: "板端推理部署", range: [5, 7] },
      { name: "进阶实战", range: [8, 99] },
    ],
    keywords: {
      rknn: "RKNN",
      rv1126: "RV1126",
      npu: "NPU",
      quantization: "量化",
      int8: "INT8",
      conversion: "模型转换",
      yolo: "YOLO",
      detection: "目标检测",
      camera: "摄像头",
      inference: "推理",
      "c api": "C API",
      python: "Python",
      lite: "Python-Lite",
      postprocess: "后处理",
      performance: "性能调优",
      ipc: "IPC",
      ispp: "ISP",
      quantize: "量化",
      "rknn-toolkit": "Toolkit",
      "rknn-toolkit2": "Toolkit2",
      "rknn-api": "RKNN API",
      board: "板端",
      deployment: "部署",
    },
    filePrefix: "rknn-",
    totalCount: 10,
  },
];

export function getSeriesById(id: string): SeriesConfig | undefined {
  return SERIES.find((s) => s.id === id);
}
