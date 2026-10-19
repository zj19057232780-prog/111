# AutoDL服务器旧入口修复（2026-10-17）

服务器usage仅接受小写模型名、1/5-shot，缺少input、checkpoint-root、result-root、epochs；与本地当前入口不一致，仅把AMAML改成amaml无法解决后续参数问题。tmux未安装，/path/to为占位路径。单模型单seed八任务直接用Python入口。

上传 `outputs/autodl_cross_commands_20261017/autodl_cross_code_current.tar.gz` 到服务器 `/root/autodl-tmp/ifmaml/`。包内55个当前代码/方法资源文件，包含主程序、my_utils、三个对比模型，不含数据和结果；已逐文件核对包内容与本地SHA256。

没有训练正在使用这些源码时，备份并更新：

```bash
cd /root/autodl-tmp/ifmaml
tar -tzf autodl_cross_code_current.tar.gz > /tmp/ifmaml_code_files.txt
tar -czf "autodl_code_backup_$(date +%Y%m%d_%H%M%S).tar.gz" --ignore-failed-read -T /tmp/ifmaml_code_files.txt
tar -xzf autodl_cross_code_current.tar.gz
cd IFMAML-master
python run_cross_dataset.py --help
```

新版help应包含input、checkpoint-root、result-root、epochs，shots为1/3/5。备份时提示个别文件原先不存在可由新包补齐，其他错误应先处理。

```bash
cd /root/autodl-tmp/ifmaml/IFMAML-master
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 MPLBACKEND=Agg
python -c "import torch; from run_cross_dataset import parse_args; from config_cross_dataset import resolve_cross_config; from cross_dataset_data import CrossDataset; parse_args(['--method','AMAML','--input','stft','--shots','1','--epochs','500']); assert torch.cuda.is_available(), 'CUDA unavailable'; CrossDataset(resolve_cross_config(dict(task='T1',method='amaml',input_mode='stft',processed_root='./cross_data_processed_ab5_ms_v2'))); print('CLI, CUDA and cache protocol OK')" && \
python -u run_cross_dataset.py --task all --method amaml --input stft --shots 1 --seed 3 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft --result-root ./results/autodl_cross_ab5_ms_v2_stft --train --test
```

检查通过才启动AMAML/STFT/1-shot/seed3八任务，前台运行请保持终端连接。本次仅准备更新包与步骤，尚未连接、更新或验证服务器。
