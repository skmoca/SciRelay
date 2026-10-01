# SciRelay

DAI 2026 oral 论文 **SciRelay: Calibrated Bid-and-Relay for Communication-Bounded Scientific Evidence Retrieval** 的开源实现与实验记录。

作者：Wenbin Ma（马文彬），Jiangnan University（江南大学）。

- 仓库：https://github.com/skmoca/SciRelay
- 论文：[paper/scirelay-camera-ready.pdf](paper/scirelay-camera-ready.pdf)
- 论文源文件：[paper/main.tex](paper/main.tex)
- 代码采用 [MIT 许可证](LICENSE)；第三方数据与图标沿用各自许可证。

## 方法与证据

各检索智能体持有不相交的文档分片，先发送固定宽度、无原始分数的 bid sketch。协调器结合本地秩与查询词互补覆盖，选择最多 `B` 篇完整摘要，再用于下游推理。`B` 是文档数量预算，20 字节是 bid 字段模型。

仓库保留 SciFact 开发集、809 查询确认集、TREC-COVID 外部验证、稠密检索及固定 NLI 验证器的实验输出，也保留同构场景无可靠收益等负结果。故障实验不构成一般 Byzantine 安全保证。

## 安装

建议使用 Python 3.12；核心包要求 Python 3.10 或更高版本，冻结依赖的可用性取决于 Python 版本。

```powershell
git clone https://github.com/skmoca/SciRelay.git
cd SciRelay
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

Linux/macOS 可用 `source .venv/bin/activate` 激活环境，核心实验命令相同。

## 目录

- `src/scirelay/`：检索、投标、选择与通信预算实现。
- `configs/`：冻结实验配置与 68 个结果文件的校验清单。
- `scripts/`：实验、分析、下载与研究声明核验入口。
- `tests/`：协议与实现测试。
- `results/`：逐查询记录、统计比较和图表。
- `data/raw/scifact-beir/`：SciFact/BEIR 快照及数据许可说明。
- `paper/`：论文 LaTeX 源文件、PDF 与图形。
- [HISTORY.md](HISTORY.md)：实验开发与冻结过程。

TREC-COVID 大型原始语料、模型权重、embedding 缓存和重复投稿包不进入仓库。完整外部语料重跑前，在 PowerShell 中执行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_trec_covid.ps1
```

## 快速验证

```powershell
$env:PYTHONUTF8='1'
python -m pytest -q
python scripts/verify_research_claims.py
```

完整重跑会依次执行 SciFact 主实验、扩展、消融、held-out confirmation split、TREC-COVID 外部验证和开发集敏感性分析（论文附录 Table 10 使用其中的 $m$ 探测结果）：

```powershell
$env:PYTHONUTF8='1'
python scripts/run_all_research.py
```

完整重跑时间明显长于快速声明核验。TREC-COVID 原始语料不进入仓库，需要时运行 `scripts/download_trec_covid.ps1` 下载。

稠密与下游实验需要额外依赖和固定公开模型 revision：

```powershell
python -m pip install -r requirements-dense.txt
$env:USE_TF='0'
$env:TRANSFORMERS_NO_TF='1'
python scripts/encode_specter.py
python scripts/run_dense_research.py --config configs/dense_development.yaml
python scripts/analyze_dense_results.py --input results/dense/development/results_full.csv --output results/dense/development/analysis
python scripts/run_dense_research.py --config configs/dense_confirmation.yaml
python scripts/analyze_dense_results.py --input results/dense/confirmation/results_full.csv --output results/dense/confirmation/analysis
python scripts/run_nli_downstream.py
```

模型权重和本地 dense embedding cache 不进入仓库；脚本、revision、逐查询结果和 NLI pair scores 会进入公开实验记录。

## 引用

```bibtex
@inproceedings{ma2026scirelay,
  author = {Wenbin Ma},
  title = {SciRelay: Calibrated Bid-and-Relay for Communication-Bounded Scientific Evidence Retrieval},
  booktitle = {The Eighth International Conference on Distributed Artificial Intelligence},
  year = {2026}
}
```

ACM DOI 和最终出版元数据将在正式出版后更新。
