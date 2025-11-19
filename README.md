# 文本丰富度分析（MAIN）

这是一个用于对中文文本进行**丰富度指标计算**的 Python 项目。它利用 [LTP（Language Technology Platform）](https://ltp.ai/) 进行中文分词、词性标注和依存句法分析，并计算一系列基于词汇和句法属性的文本复杂度指标。

## 📌 项目概述

本项目主要实现的功能是：

1.  利用 LTP 对大规模中文文本（如 CSV 文件中的文本列）进行**批量**预处理（分词、词性标注、依存句法分析）。
2.  计算每个文本的七项复杂度指标：
    * **SR, SC, SN**：与名词化（Nominalization）相关的指标。
    * **Lexical Entropy (词汇丰富度)**：衡量词汇多样性。
    * **Syntactic Entropy (句法丰富度)**：衡量依存关系的多样性。
    * **Syntactic Entropy Norm (归一化句法丰富度)**：归一化后的句法熵。
    * **SA (Semantic Abstraction/语义丰富度)**：衡量句子中名词、动词、形容词、副词的丰富度。
3.  将计算结果以新列的形式，写回到原始 CSV 文件中。

## ⚙️ 依赖环境

### Python 依赖

您需要安装以下 Python 库：

```bash
pip install torch pandas tqdm
# 安装 LTP，这里使用了 v4 版本
pip install ltp
```
# 🚀 如何运行
## 1. 配置路径
在 main() 函数中，您需要配置输入文件的路径和文本列名：

Python
```
def main():
    # ========= 路径参数 =========
    # 待处理的 CSV 文件路径
    CSV_PATH = r"E:\A-word_complexity\Dataset\Weibo_Unemployment\Weibo_utils_clean_with_counts.csv"
    # 待处理的文本所在的列名
    TEXT_COL = "text_clean"
    # 结果保存路径
    SAVE_PATH = CSV_PATH.replace(".csv", "_with_metrics.csv")
    
    # ...
```
## 2. 执行主程序
运行 main 函数即可开始批量计算：
```
Bash

python your_script_name.py
```
程序会使用 tqdm 显示处理进度条，处理完成后，结果文件将被保存到配置的 SAVE_PATH。

## 3. 批量处理优化
程序采用了批量处理 (Batch Processing) 机制，通过设置 BATCH = 64，将文本分批次送入 GPU 上的 LTP 模型进行推理，这极大地提高了处理效率。
