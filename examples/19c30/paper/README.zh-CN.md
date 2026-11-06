# 19C30 论文实验

[English](README.md) | 中文

本目录包含论文中 PSR J1912-1215 的定时实验：

| 实验 | 提供的材料 |
| --- | --- |
| 时间前缀 | 12–23 会话的 12 次原始运行，包括输入、候选路径、拟合检查和耗时 |
| 搜索稳定性 | 17–19 会话的 15 次运行：基线、13 点相位网格、较大保留预算、F1 零初值和较宽 F1 范围 |
| 备选路径转移 | 将 16 会话备选限制到 15 会话后的固定路径拟合与检查 |
| PEPOCH 60604 | 完整数据的独立运行、拟合 PAR/TIM，以及与原脉冲路径的比较 |

可直接查看[前缀表](tables/prefixes.csv)、[对照表](tables/controls.csv)、[路径计数图](figures/paper-path-counts.png)和[定时残差图](figures/epoch60604-residuals.png)。这些图表根据保存的实验记录生成，不代表新的盲搜结果。

`recorded/` 保存历史结果。`frozen/prefix/` 是随 PEPOCH 60604 运行保留的历史代码，但尚未证实所有源文件均与原前缀实验相同；`frozen/controls/` 是稳定性对照保存的代码。两者与当前 `apollo_timing` 软件包分开使用。这里没有保存当时完整的 Python 环境，不保证不同依赖版本产生逐位相同的数值。

## 核对已有结果

安装项目依赖后，在仓库根目录运行：

```bash
python examples/19c30/paper/run.py --task verify \
  --output "$HOME/apollo-runs/paper-verify"
```

该命令重新统计通过验收的不同脉冲路径、核对参考路径，并输出汇总 JSON/CSV，不重新执行盲搜。原盲搜中的参考编号只用于推断后的比较。

## 生成图件

指定自己的 DE421 星历和所需台站钟差：

```bash
export PINT_CLOCK_OVERRIDE=/path/to/clocks

python examples/19c30/paper/run.py --task plot \
  --ephem-file /path/to/de421.bsp \
  --output "$HOME/apollo-runs/paper-plots"
```

输出前缀与对照的路径计数图，以及 PEPOCH 60604 的残差随 MJD、轨道相位图，均包含 PNG/PDF。公开脚本采用独立排版，不复制论文版式。极化、初步 newfitorbit 拟合及伴星群体分布图不属于 APOLLO 定时实验，未包含在本目录。

## 重新运行实验

运行全部 12 个时间前缀，或用 `--sessions` 选择部分会话数：

```bash
python examples/19c30/paper/run.py --task prefixes \
  --ephem-file /path/to/de421.bsp \
  --output "$HOME/apollo-runs/paper-prefixes"

python examples/19c30/paper/run.py --task prefixes --sessions 15 16 \
  --ephem-file /path/to/de421.bsp \
  --output "$HOME/apollo-runs/paper-prefixes-15-16"
```

运行全部 15 组稳定性对照：

```bash
python examples/19c30/paper/run.py --task controls --workers 1 \
  --ephem-file /path/to/de421.bsp \
  --output "$HOME/apollo-runs/paper-controls"
```

从保存的备选路径重做 16→15 会话固定路径诊断：

```bash
python examples/19c30/paper/run.py --task transfer \
  --ephem-file /path/to/de421.bsp \
  --output "$HOME/apollo-runs/paper-transfer"
```

在 PEPOCH 60604 重跑完整数据盲搜：

```bash
python examples/19c30/paper/run.py --task epoch60604 \
  --ephem-file /path/to/de421.bsp \
  --output "$HOME/apollo-runs/paper-epoch60604"
```

每次使用新的源码目录外输出路径。单次盲搜默认超时 3600 秒，可用 `--timeout` 修改。历史耗时不能用于估计其他机器的运行时间。

## 怎样读这些记录

论文前缀表统计的是通过验收的不同路径，不是全部完成的数值拟合。19 会话的第二个拟合未通过验收；转移后的 15 会话结果属于固定路径诊断，不改变原盲搜计数。稳定性对照没有证明搜索覆盖完整，最大似然比也不是积分证据比或路径概率。

公开文件去除了机器名、用户名、环境路径、原始日志和内部审阅草稿。`${BUNDLE}` 表示由脚本解析的本目录相对位置；`${EXTERNAL_RESOURCE}` 表示未提供的外部资源或旧构建位置。路径和注释清理后的请求、检查点只作为证据，不可直接用作断点缓存。

本次公开整理没有重跑整组长时实验。新执行的核对结果写到指定外部目录，不混入历史记录。
