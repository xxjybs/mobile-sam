# Orange Defect Segmentation - Heatmap Visualization Summary

## 问题 / Problem

用户在运行 `train.py` 后发现预测结果与真实掩码有较大差距，需要可视化工具来分析问题。

After running `train.py`, the user noticed a significant difference between predicted masks and ground truth, and needed visualization tools to analyze the problem.

## 解决方案 / Solution

创建了一套完整的热力图可视化工具，帮助分析和诊断模型预测与真实标签之间的差异。

Created a comprehensive set of heatmap visualization tools to analyze and diagnose differences between model predictions and ground truth labels.

## 新增文件 / New Files

### 1. 核心工具 / Core Tools

#### `visualize_heatmap_simple.py` ⭐ **推荐 / Recommended**
- **用途 / Purpose**: 简化版可视化工具，直接对比预测与真实掩码
- **特点 / Features**:
  - ✅ 无需加载模型 / No model loading required
  - ✅ 快速生成结果 / Fast result generation
  - ✅ 依赖少 / Minimal dependencies (numpy, matplotlib, pillow)
  - ✅ 支持 --help 不需要安装依赖 / --help works without installing dependencies

**基本用法 / Basic Usage**:
```bash
python visualize_heatmap_simple.py
```

#### `visualize_heatmap.py`
- **用途 / Purpose**: 高级版可视化工具，可选加载模型生成置信度图
- **特点 / Features**:
  - 📊 可生成预测置信度热力图 / Can generate prediction confidence heatmaps
  - 🖼️ 支持原始图像叠加 / Supports original image overlay
  - 🔍 更详细的分析 / More detailed analysis
  - 🔧 需要模型和更多依赖 / Requires model and more dependencies

**使用示例 / Usage Example**:
```bash
python visualize_heatmap.py \
    --pred_path ./save/mobile_sam_adapter/vit_change_2/pred/398.png \
    --gt_path ./data/orange/masks/398.png \
    --ckpt_path ./save/mobile_sam_adapter/vit_change_2/mobile_sam_adapter_best.pth \
    --img_name 398
```

### 2. 文档 / Documentation

#### `HEATMAP_VISUALIZATION_README.md`
- 中英文详细文档 / Detailed documentation in Chinese and English
- 包含用法示例、常见问题、批处理等 / Includes usage examples, FAQs, batch processing

#### `使用指南_热力图可视化.md`
- 详细的中文使用指南 / Detailed Chinese usage guide
- 指标解释（IoU, Precision, Recall, F1） / Metric explanations
- 分析建议和改进方向 / Analysis suggestions and improvement directions
- 批量处理示例 / Batch processing examples

#### `example_heatmap_usage.py`
- 使用示例脚本 / Example usage script
- 演示多种使用场景 / Demonstrates various use cases

## 功能特点 / Features

### 1. 差异分析可视化 / Difference Analysis Visualization

彩色编码显示不同类型的预测结果：
Color-coded visualization of different prediction types:

- 🟢 **绿色 / Green**: True Positive (TP) - 正确预测的缺陷 / Correctly detected defects
- 🔴 **红色 / Red**: False Positive (FP) - 误报 / False alarms (incorrect defect predictions)
- 🟡 **黄色 / Yellow**: False Negative (FN) - 漏报 / Missed defects
- ⚫ **黑色 / Black**: True Negative (TN) - 正确的背景 / Correct background

### 2. 评估指标 / Evaluation Metrics

自动计算并显示：
Automatically calculates and displays:

- **IoU** (Intersection over Union / 交并比): 预测与真实区域的重叠程度
- **Precision** (精确率): 预测为缺陷中真正是缺陷的比例
- **Recall** (召回率): 真实缺陷中被正确预测的比例
- **F1-Score**: 精确率和召回率的调和平均
- **混淆矩阵统计 / Confusion Matrix**: TP, FP, FN, TN 像素计数

### 3. 多种可视化 / Multiple Visualizations

生成的热力图包含 6 个子图：
Generated heatmap contains 6 subplots:

1. 预测掩码（灰度） / Predicted mask (grayscale)
2. 真实掩码（灰度） / Ground truth mask (grayscale)
3. 差异分析图（彩色） / Difference map (color-coded)
4. 预测掩码热力图（伪彩色） / Predicted mask heatmap (pseudo-color)
5. 真实掩码热力图（伪彩色） / Ground truth heatmap (pseudo-color)
6. 误差热力图（差异强度） / Error heatmap (difference intensity)

## 快速开始 / Quick Start

### 1. 安装依赖 / Install Dependencies

**简化版 / Simple version**:
```bash
pip install numpy matplotlib pillow
```

**完整版 / Advanced version**:
```bash
pip install numpy matplotlib pillow torch
pip install scipy  # 可选 / optional
```

### 2. 运行工具 / Run Tool

**默认使用 / Default usage** (for image 398):
```bash
python visualize_heatmap_simple.py
```

**自定义图像 / Custom image**:
```bash
python visualize_heatmap_simple.py --img_name 123
```

**完全自定义 / Fully custom**:
```bash
python visualize_heatmap_simple.py \
    --pred_path <path_to_prediction> \
    --gt_path <path_to_ground_truth> \
    --output_path <output_path>
```

### 3. 查看结果 / View Results

输出的热力图将包含：
The output heatmap will include:

- 📊 可视化对比图 / Visual comparison
- 📈 详细的评估指标 / Detailed evaluation metrics
- 🎨 彩色差异分析 / Color-coded difference analysis

## 输出示例 / Output Example

控制台输出 / Console output:
```
============================================================
✅ 热力图已保存: ./defect_heatmap_398.png
============================================================
📊 评估指标:
   IoU (交并比):     0.7845
   Precision (精确率): 0.8234
   Recall (召回率):   0.8912
   F1-Score:         0.8560

📈 混淆矩阵统计:
   True Positive  (TP - 正确检测):   45,678 像素
   False Positive (FP - 误报):        9,234 像素
   False Negative (FN - 漏报):        5,467 像素
   True Negative  (TN - 正确背景): 205,123 像素
============================================================
```

## 批量处理 / Batch Processing

### Python 脚本 / Python script:
```python
import subprocess

img_names = ['398', '399', '400', '401', '402']

for img_name in img_names:
    print(f"Processing {img_name}...")
    subprocess.run([
        'python', 'visualize_heatmap_simple.py',
        '--img_name', img_name
    ])
```

### Bash 脚本 / Bash script:
```bash
#!/bin/bash
for img in 398 399 400 401 402; do
    echo "Processing image $img..."
    python visualize_heatmap_simple.py --img_name $img
done
```

## 技术亮点 / Technical Highlights

1. **延迟导入 / Lazy Import**: 
   - 主要依赖在函数内部导入 / Main dependencies imported inside functions
   - `--help` 无需安装依赖即可查看 / `--help` works without installing dependencies

2. **字体回退 / Font Fallback**:
   - 自动检测可用的中文字体 / Auto-detects available Chinese fonts
   - 跨平台兼容性好 / Good cross-platform compatibility

3. **可选依赖 / Optional Dependencies**:
   - scipy 是可选的，使用 PIL 作为回退 / scipy is optional, PIL used as fallback
   - 灵活的依赖管理 / Flexible dependency management

4. **错误处理 / Error Handling**:
   - 完善的错误处理和提示 / Comprehensive error handling and messages
   - 友好的用户反馈 / User-friendly feedback

## 分析建议 / Analysis Suggestions

### 如果 IoU 较低 / If IoU is low:
- 检查差异图中的颜色分布 / Check color distribution in difference map
- 红色多 → 误报严重 / Red dominant → Many false positives
- 黄色多 → 漏报严重 / Yellow dominant → Many false negatives

### 如果精确率低 / If Precision is low:
- 模型把很多正常区域误判为缺陷 / Model incorrectly predicts many normal areas as defects
- 建议：增加负样本、调整阈值 / Suggestions: Add negative samples, adjust threshold

### 如果召回率低 / If Recall is low:
- 模型遗漏了很多真实缺陷 / Model misses many real defects
- 建议：增加正样本、降低阈值 / Suggestions: Add positive samples, lower threshold

## 常见问题 / Common Issues

### Q: 提示 "文件不存在" / File not found
**A**: 检查路径是否正确，文件是否存在 / Check if paths are correct and files exist

### Q: 缺少依赖模块 / Missing dependencies
**A**: 运行 `pip install numpy matplotlib pillow` / Run installation command

### Q: 中文显示为方框 / Chinese text shows as boxes
**A**: 这是字体问题，不影响功能 / Font issue, doesn't affect functionality

## 文件清单 / File List

```
mobile-sam/
├── visualize_heatmap_simple.py          # 简化版工具 (推荐)
├── visualize_heatmap.py                 # 完整版工具
├── HEATMAP_VISUALIZATION_README.md      # 中英文文档
├── 使用指南_热力图可视化.md              # 详细中文指南
└── example_heatmap_usage.py             # 使用示例
```

## 总结 / Summary

这套工具可以帮助您：
These tools help you:

1. ✅ 直观地看到预测与真实的差异 / Visualize prediction vs ground truth differences
2. ✅ 定量分析模型性能 / Quantitatively analyze model performance
3. ✅ 识别模型的弱点 / Identify model weaknesses
4. ✅ 为改进模型提供依据 / Provide basis for model improvement

## 下一步 / Next Steps

1. 运行可视化工具分析您的模型 / Run visualization tools to analyze your model
2. 根据指标和可视化结果调整模型 / Adjust model based on metrics and visualizations
3. 使用批量处理功能分析多个样本 / Use batch processing to analyze multiple samples
4. 根据分析结果改进训练策略 / Improve training strategy based on analysis

## 支持 / Support

如有问题或建议，请查看文档或提交 Issue。
For questions or suggestions, please refer to the documentation or submit an issue.

---

**Happy Analyzing! 祝分析顺利！** 🎉
