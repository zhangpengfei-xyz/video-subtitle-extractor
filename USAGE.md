# 命令行使用说明

在代码仓目录执行以下命令。

## 安装依赖

```bash
pip3 install --break-system-packages uv
uv python install 3.12
uv venv --python 3.12 vse_venv
uv pip install --python vse_venv pip
./vse_venv/bin/pip3 install -r requirements.txt
./vse_venv/bin/pip3 install paddle2onnx==2.1.0 openvino==2026.4.0 onnxruntime==1.30.0

# convert onnx models
for MODEL_DIR in backend/models/V5/PP-OCRv5_*_infer/; do
  ./vse_venv/bin/paddle2onnx --model_dir "$MODEL_DIR" \
    --model_filename inference.json --params_filename inference.pdiparams \
    --save_file "$MODEL_DIR/inference.onnx"
done
```

使用 OpenVINO 或 ONNX Runtime 前导出模型一次；更换模型权重后重新导出。

## 提取字幕

```bash
# 单个视频：OpenVINO CPU FP32
./vse_venv/bin/python3 cli.py "/path/to/video.mp4" --backend openvino

# 批量视频
./vse_venv/bin/python3 cli.py /path/to/videos/*.mp4 --backend openvino

# 自定义选区：左上角、右下角坐标，均为 0～1 比例
./vse_venv/bin/python3 cli.py "/path/to/video.mp4" \
  --crop 0.05 0.88 0.95 0.98 --threads 8

# 全部参数
./vse_venv/bin/python3 cli.py --help
```

- `--backend paddle|openvino|onnxruntime`：默认 `paddle`。
- `--crop-bottom 0.3`：默认搜索底部 30%，与 `--crop` 二选一。
- `--min-text-height-ratio 0.037037`：最小字高占完整视频高度的比例，默认约 40/1080；0 关闭过滤。
- `--threads 8`：OCR 和每个 VideoSubFinder 进程的线程数，默认 8。
- OpenCV 图像处理默认 1 线程，可通过环境变量 `OPENCV_FOR_THREADS_NUM` 覆盖。
- `--vsf-workers 6`：Linux/macOS 默认最多 6 段并行扫描，短视频自动减少分段；设为 1 使用单进程。
- `--vsf-overlap 5`：分段边界初始重叠秒数，遇到截断字幕自动扩大扫描范围。
- `--language ch --mode fast`：默认中文、fast 模型；模式可选 `fast|auto|accurate`。
- `--report result.json`：保存单个视频的处理报告。
- `--keep-work`：成功后保留中间文件。

字幕保存为视频旁的同名 `.srt`，成功时覆盖已有字幕，失败时保留原文件。每次运行会清空并重建视频旁的同名目录作为工作目录；成功后删除，失败时保留。批量视频按顺序处理。
