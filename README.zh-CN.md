# APOLLO-Timing 0.8.3

[English](README.md) | 中文

APOLLO-Timing 用于单星和双星脉冲星定时，支持命令行和 Python 接口。

## 功能

| 功能 | 用途 |
| --- | --- |
| 相位连接与参数拟合 | 从 PAR/TIM 搜索连接，按输入拟合标记调整参数并保存备选结果 |
| 模型评价 | 检查给定模型与脉冲路径，不重新拟合 |
| 观测更新 | 将新增 TOA 加入已有定时任务 |
| 轨道初始化与搜索 | 补全不完整双星输入，或在指定周期范围内寻找轨道 |
| 参数与模型比较 | 比较额外参数及不同双星模型 |
| 参数范围限制 | 设置参数硬区间和位置总角距离限制 |
| 单星 glitch 拟合 | 显式搜索一次永久频率跃变，可同时拟合频率导数跃变 |
| 观测规划 | 根据保留候选推荐后续观测时间 |
| Foldbank | 生成 ELL1 参数文件库与 PRESTO 折叠命令 |

支持的双星模型：`BT`、`DD`、`DDS`、`DDGR`、`DDH`、`DDK`、`ELL1`、`ELL1H`、`ELL1k`、`BT_piecewise`。

## 安装

Linux 或 WSL，Python 3.11 及以上。在源码根目录运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
apollo-timing --version
```

星历、台站钟差和 PRESTO 需自行准备。以下命令中的 PAR、TIM、FITS 和资源路径均替换为自己的文件；每次新运行使用不同的输出目录。

## 定时示例

### 自动选择单星或双星流程

```bash
apollo-timing --par source.par --tim observations.tim \
  --output "$HOME/apollo-runs/source"
```

### 单星与双星求解

```bash
apollo-timing --source-type single \
  --par single.par --tim observations.tim \
  --output "$HOME/apollo-runs/single"

apollo-timing --source-type binary \
  --par binary.par --tim observations.tim \
  --output "$HOME/apollo-runs/binary"
```

### 评价已有结果

```bash
apollo-timing --task evaluate \
  --par solution.par --tim solution.tim \
  --output "$HOME/apollo-runs/evaluate"
```

### 加入新观测

```bash
apollo-timing --task update \
  --par previous.par --tim historical.tim --update-tim new.tim \
  --output "$HOME/apollo-runs/update"
```

### 继续未完成的运行

输入、配置及资源不变时，在原命令后加 `--resume`：

```bash
apollo-timing --par source.par --tim observations.tim \
  --output "$HOME/apollo-runs/source" --resume
```

### 使用指定钟差和星历

```bash
apollo-timing --par source.par --tim observations.tim \
  --clock-dir /path/to/clocks --ephem-file /path/to/de421.bsp \
  --output "$HOME/apollo-runs/local-resources"
```

星历必须匹配 PAR 中的 `EPHEM`；也可用 `--ephem-dir /path/to/ephemerides` 指定星历目录。

## 轨道与参数示例

### 搜索未知轨道

```bash
apollo-timing --source-type binary \
  --par source.par --tim observations.tim \
  --discovery-period-min-days 1 --discovery-period-max-days 100 \
  --output "$HOME/apollo-runs/orbit-search"
```

### 设置参数与位置范围

以下 F0 范围仅为示例，应按输入频率修改；区间不会自动开放固定参数。

```bash
apollo-timing --source-type binary \
  --par binary.par --tim observations.tim \
  --parameter-bound F0 99.999999 100.000001 --sky-radius-arcmin 1.5 \
  --output "$HOME/apollo-runs/bounded"
```

### 比较额外参数

```bash
apollo-timing --source-type binary \
  --par binary.par --tim observations.tim \
  --explore-parameter PBDOT --exploration-bound PBDOT -1e-10 1e-10 \
  --output "$HOME/apollo-runs/pbdot"
```

### 比较双星模型

```bash
apollo-timing --binary-model-selection --binary-models ELL1 DD ELL1H \
  --par binary.par --tim observations.tim \
  --output "$HOME/apollo-runs/model-comparison"
```

### 单星 glitch 搜索

历元范围使用 TDB MJD，按自己的观测时间修改。

```bash
apollo-timing --source-type single --glitch-search \
  --par single.par --tim observations.tim \
  --glitch-epoch-range 58000 59000 --glitch-epoch-samples 32 \
  --output "$HOME/apollo-runs/glitch" --no-observation-plan
```

## 观测规划

从已有运行目录生成后续观测建议：

```bash
apollo-observe --run-output "$HOME/apollo-runs/source" \
  --output "$HOME/apollo-runs/observation-plan" --horizon-days 180 --top 10
```

## Foldbank

输入粗略 ELL1 模型、训练 TOA 和对应 SEARCH 模式 PSRFITS，生成候选参数文件及折叠脚本：

```bash
apollo-foldbank --par coarse-ell1.par --tim training.tim \
  --search epoch1.fits --search epoch2.fits \
  --validation-search heldout.fits \
  --pb-min 6.25 --pb-max 6.37 --toa-origin external_discovery \
  --output "$HOME/apollo-runs/foldbank"
```

周期范围和 TOA 来源声明须与实际输入一致。默认只生成脚本；安装 PRESTO 后加 `--run-presto` 可执行折叠。

## 输出

| 文件 | 内容 |
| --- | --- |
| `run_summary.json` | 运行状态与结果摘要 |
| `certification.json` | 候选验收结果与诊断 |
| `solution.par`、`solution.tim` | 保存的模型与脉冲路径 |
| `candidate_portfolio.json` | 保留的备选候选 |
| `residuals.csv`、`residuals.png` | 残差数据与图 |
| `observation_plan/` | 可生成时的观测建议 |

先检查 JSON 状态；存在 `solution.par` 不表示运行已通过验收。

## Python 接口

```python
from pathlib import Path
from apollo_timing.pipeline import solve_apollo_timing

result = solve_apollo_timing(
    par_path=Path("binary.par"),
    tim_path=Path("observations.tim"),
    output_dir=Path.home() / "apollo-runs" / "python-api",
    source_type="binary",
)
print(result.accepted, result.solution_par)
```

## 命令帮助

```bash
apollo-timing --help
apollo-bootstrap-binary --help
apollo-bootstrap-ell1 --help
apollo-observe --help
apollo-foldbank --help
```

## 许可

[MIT](LICENSE)
