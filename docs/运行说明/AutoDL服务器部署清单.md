# AutoDL 服务器部署清单

生成：2026-10-17。适用：把当前代码与数据搬到 AutoDL GPU 实例上跑 PU/CWRU/跨库实验。

当前八任务跨库实验请使用[AutoDL跨数据集实验命令清单](AutoDL跨数据集实验命令清单.md)。跨库统一每类10 query，完整上传 `cross_data_processed_ab5_ms_v2` 即可运行STFT及nostft，无需原始MAT，不执行预处理。下述 `pu_data_mat` 供单库相关入口使用，文件命名不满足当前跨库原始数据入口；跨库重建需另备PU原始命名MAT。`pu_preprocess.py`/`cwru_preprocess.py` 也不能替代跨库预处理。

## 1. 目录结构要求（硬约束）

`comparison_models/*/run.py` 通过 `HERE.parent.parent / 'IFMAML-master'` 引用主程序，**服务器上必须保持以下平级结构**（根目录名可任意）：

```text
autodl-tmp/（或 /root/）
├── IFMAML-master/      ← 主程序
└── comparison_models/  ← 三个独立对比模型
```

## 2. 必传：代码（约几 MB）

| 内容 | 说明 |
|---|---|
| `IFMAML-master/` 顶层全部 `*.py`（约 45 个） | 主程序入口、数据集/预处理/配置（config_pu、config_cwru、config_cross_dataset、classic_config）、`my_utils/`、`l2l_shim.py`（自实现，**不依赖 learn2learn 包**） |
| `IFMAML-master/my_utils/` | init_utils.py、train_utils.py |
| `IFMAML-master/docs/`（可选） | 运行说明 |
| `comparison_models/mu2025_amaml/` | `run.py`、`model.py`、`raw_reshape.py`、`make_tsne.py` |
| `comparison_models/wang2025_tubr_ml/` | `run.py`、`tubr_model.py`、`raw_reshape.py`、`method_eta_text.json` |
| `comparison_models/li2025_mfeml/` | `run.py`、`mfeml_model.py`、`raw_input.py` |

不需要：`.ps1` 脚本（Windows 专用）、`__pycache__`、测试脚本（test_*.py 可选传）。

## 3. 必传：数据（按跑什么选）

| 数据目录 | 大小 | 谁需要 |
|---|---|---|
| `IFMAML-master/pu_data_processed/` | 22M | 主程序 PU STFT 输入（5-way 系列全部任务） |
| `IFMAML-master/cwru_data_processed/` | 6.9M | 主程序 CWRU STFT 输入 |
| `IFMAML-master/pu_data_mat/` | 1.1G | raw1d 系列、AMAML(raw38)、MFEML/TUBR-ML(no_stft) 的原始信号 |
| `IFMAML-master/CWRU/` | 657M | 同上（CWRU 原始 MAT） |
| `IFMAML-master/cross_data_processed_ab5_ms_v2/` | 81M | 当前跨库方案 Cross-AB-MS-5way-targetval-v2（含 episode 清单） |
| `IFMAML-master/cwru_data_gao2025_stft/` | 85M | 仅跑 Gao2025 CWRU 任务时 |
| `IFMAML-master/cross_data_processed/` + `cross_data_processed_ab5_v1/` | 19M + 36M | 仅复现旧 3-way/AB5-v1 跨库批次时 |
| `IFMAML-master/hust_data_processed/` + HUST 原始数据 | 56M + 665M | 仅跑 HUST 实验 |

**核心组合约 1.9G**：pu_data_mat + CWRU + 两个 *_data_processed + cross_data_processed_ab5_ms_v2。

STFT 图片缓存也可不传、在服务器用 `pu_preprocess.py`/`cwru_preprocess.py` 从原始 MAT 重新生成；但直接传缓存可保证与本地结果**同数据逐位一致**（项目确定性复现已验证，数据一致才有可比性）。

## 4. 不传

- `results/`（335M）、`table_row_results/`（945M）、`model_save/`（532M）、`logs/`、`tmp/`、`outputs/验证记录/`、`实验数据归档_*`、`archive/`
- `comparison_models/*/runs/`（合计 ~240M；除非要在服务器独立重测已有权重——那样按需传对应 `best.pt` + 4 个 json）
- 各 `*_实验结果汇总_*.md/csv`（本地检索用，服务器不需要）

## 5. 打包命令（Windows Git Bash，在 new-ifmaml 根目录执行）

```bash
# 代码包（不含任何数据/结果）
tar --exclude='__pycache__' --exclude='*.pyc' -czvf autodl_code.tar.gz \
  IFMAML-master/*.py IFMAML-master/my_utils IFMAML-master/docs \
  comparison_models/mu2025_amaml comparison_models/wang2025_tubr_ml comparison_models/li2025_mfeml \
  --exclude='*/runs' --exclude='*/__pycache__' --exclude='*.md' --exclude='*.csv'

# 数据包（核心 1.9G）
tar -czvf autodl_data.tar.gz \
  IFMAML-master/pu_data_mat IFMAML-master/CWRU \
  IFMAML-master/pu_data_processed IFMAML-master/cwru_data_processed \
  IFMAML-master/cross_data_processed_ab5_ms_v2
```

上传：AutoDL 网盘 / `scp` / `rsync` 均可；AutoDL 对阿里云盘/网盘中转有内网加速。解压后确认平级结构即可。

## 6. 环境与路径坑

1. **依赖**（项目无 requirements.txt，按导入清单装）：
   `torch`（本地 2.4.1+cu121；服务器按实例 CUDA 版本选，建议 ≥2.0 以支持 `torch.func.functional_call`）、`numpy`、`scipy`、`opencv-python`、`pandas`、`matplotlib`、`scikit-learn`（t-SNE）、`Pillow`。
2. **硬编码路径**：`config_pu.py`/`config_cwru.py` 的 `root_path/mat_root` 及已存 `resolved_config.json` 均为本地 Windows 绝对路径。服务器上：
   - 对比模型：用 `--data-root` 覆盖即可；
   - 主程序：需改 `config_pu.py`/`config_cwru.py` 中的路径，或核对各入口是否支持 data-root 参数（`run_table_row.py` 支持；`maml_train_pic.py` 读 config）。
   - 建议：上传后先跑 `--action check`（对比模型）或 1 轮 smoke，确认数据路径与形状。
3. **设备差异**：已验证 CPU/CUDA 适应预测存在数值差异，**不能**把服务器结果与本机结果直接混入同一统计或宣称等价；固定在服务器同一卡上跑完整个实验组。
4. **seed/协议不变**：seed 3/24/38、5-way、15-query、1000 测试 episode 等协议全部沿用命令清单；跨库任务用 `outputs/cross_dataset_literature_20261016/Cross-AB-MS-5way-targetval-v2_最终方案.md` 的配置。
5. Python 版本：本地 3.13；AutoDL 常见 3.10/3.12 镜像，代码无版本敏感语法（用了 `from __future__ import annotations`），3.10+ 均可。

## 7. 上传后自检（建议顺序）

```bash
python comparison_models/mu2025_amaml/run.py --dataset pu --tasks P1 --seeds 3 --shot 1 --action check   # 数据路径/形状
python comparison_models/mu2025_amaml/run.py --dataset pu --tasks P1 --seeds 24 --shot 1 --action smoke  # 1 轮冒烟
# 主程序跨库 smoke 见 IFMAML-master/docs/跨数据集AB5运行说明.md
```
