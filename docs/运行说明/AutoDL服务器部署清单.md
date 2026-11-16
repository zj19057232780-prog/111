# AutoDL服务器部署清单（2026-10-19）

当前运行 CWRU/PU 独立实验，参见[正式命令](独立数据集_正式实验命令.md)。

必须保持平级目录：
```text
项目根目录/
  IFMAML-master/
  comparison_models/
```

上传 IFMAML-master 顶层运行用 .py、my_utils/、optional_models/；以及 comparison_models 下 mu2025_amaml、wang2025_tubr_ml、li2025_mfeml 的 .py/.json 文件。不要漏掉 optional_models，否则旧组件兼容导入失败。

数据按任务上传：PU 的 pu_data_processed（STFT）和 pu_data_mat（nostft）；CWRU 的 CWRU 原始MAT及 __file__/metadata.txt。统一入口会为CWRU STFT生成逐任务独立缓存，也可通过 --data-map 指定核查过的逐任务图片目录。

无需上传 archive、历史 results/model_save/table_row_results、comparison_models/*/runs、tmp、outputs、跨库缓存或河内HUST数据。华中HUST尚未接入，不能上传旧河内缓存替代。

本地验证环境为 Python3.8、PyTorch2.4.1+cu121；依赖 numpy、scipy、opencv-python、pandas、matplotlib、scikit-learn、Pillow。按服务器GPU环境配置PyTorch，运行前确认：
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
cd /root/autodl-tmp/你的项目/IFMAML-master
python -u run_experiment.py --dataset pu --tasks P1 --method maml --input nostft --seed 3 --action check --device cuda
```

数据不在默认相对位置时用 --data-root / --raw-root / --data-map。正式实验同一组使用一致硬件、数据和代码；旧结果补测需要完整运行目录和匹配源码快照。
