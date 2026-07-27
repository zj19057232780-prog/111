# Project Maintenance Instructions

当前目录是实际 Git 仓库，但项目总文档位于上一级目录：

```text
C:\Users\JieZhang\Desktop\new-ifmaml\项目文档.md
```

每次修改 `IFMAML-master` 下的代码、配置、数据协议、实验结果或说明文档后，都必须同步更新该项目文档，并在文档末尾“项目变更记录”中追加记录。

尤其注意同步：

- `config_pu.py` 和 `config_cwru.py` 的默认实验配置；
- `maml_model.py` 中新增、删除或调整的模型模块；
- `maml_train_pic.py`、`l2l_shim.py`、`pu_dataset.py`、`cwru_dataset.py` 中的训练、采样、划分逻辑；
- `results`、`model_save` 中对应的新实验结果；
- 和论文创新点、评测风险有关的新结论。
