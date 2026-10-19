"""
模型转换脚本：将PyTorch MAML模型转换为ONNX格式

用法：
    conda activate ifmaml2
    pip install onnx onnxruntime onnxscript
    python pytorch_to_onnx.py
"""

import subprocess
import sys

def check_and_install_dependencies():
    """检查并安装依赖"""
    required = ['onnx', 'onnxruntime', 'onnxscript']
    missing = []

    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print("\n正在安装缺失的依赖...")
        for package in missing:
            print(f"  pip install {package}")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        print("依赖安装完成！\n")

check_and_install_dependencies()

import torch
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from maml_model import Net4CNN
from config_pu import PU_CONFIG


def pytorch_to_onnx(
    model_path: str,
    output_path: str,
    input_shape: tuple = (1, 1, 64, 64),
    opset_version: int = 11
):
    """
    将PyTorch模型转换为ONNX格式

    Parameters
    ----------
    model_path : str
        PyTorch模型文件路径 (.pth 或 .pt)
    output_path : str
        输出ONNX文件路径 (.onnx)
    input_shape : tuple
        输入张量形状 (batch_size, channels, height, width)
    opset_version : int
        ONNX opset版本
    """
    print("=" * 70)
    print("PyTorch → ONNX 模型转换")
    print("=" * 70)

    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return False

    print(f"\n1. 加载模型: {model_path}")

    # 获取配置参数
    cfg = PU_CONFIG
    n_way = cfg['n_way']
    h_size = 64
    layers = 4
    sample_len = 256
    feat_size = (sample_len // 2**layers) * h_size
    in_channels = cfg.get('in_channels', 1)

    # 创建模型实例
    model = Net4CNN(
        output_size=n_way,
        hidden_size=h_size,
        layers=layers,
        channels=in_channels,
        embedding_size=feat_size
    )

    # 加载权重
    state_dict = torch.load(model_path, weights_only=True, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()  # 重要：转换为评估模式

    print(f"   模型架构:")
    print(f"   - 输入通道: {in_channels}")
    print(f"   - 隐藏层大小: {h_size}")
    print(f"   - 层数: {layers}")
    print(f"   - 输出类别: {n_way}")
    print(f"   - 特征维度: {feat_size}")

    # 创建dummy输入
    print(f"\n2. 创建dummy输入: {input_shape}")
    dummy_input = torch.randn(input_shape)

    # 尝试导出为ONNX
    print(f"\n3. 导出为ONNX格式...")
    print(f"   输出路径: {output_path}")
    print(f"   Opset版本: {opset_version}")

    try:
        # ONNX导出
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        # 验证ONNX模型
        print(f"\n4. 验证ONNX模型...")
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("   [OK] ONNX模型验证通过")

        # 获取文件大小
        pytorch_size = os.path.getsize(model_path) / (1024 * 1024)
        onnx_size = os.path.getsize(output_path) / (1024 * 1024)

        print(f"\n5. 文件信息:")
        print(f"   PyTorch模型: {pytorch_size:.2f} MB")
        print(f"   ONNX模型: {onnx_size:.2f} MB")
        print(f"   压缩比: {pytorch_size/onnx_size:.2f}x")

        print("\n" + "=" * 70)
        print("转换成功！")
        print("=" * 70)

        # 测试推理
        print("\n6. 测试推理...")
        import onnxruntime as ort

        # ONNX输出
        ort_session = ort.InferenceSession(output_path)
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name

        onnx_output = ort_session.run(
            [output_name],
            {input_name: dummy_input.numpy()}
        )[0]

        # PyTorch输出
        with torch.no_grad():
            pytorch_features, pytorch_predictions = model(dummy_input)

        print(f"   PyTorch predictions shape: {pytorch_predictions.shape}")
        print(f"   PyTorch features shape: {pytorch_features.shape}")
        print(f"   ONNX output shape: {onnx_output.shape}")

        # 比较predictions（两个模型都应该输出5维分类结果）
        if pytorch_predictions.shape == onnx_output.shape:
            diff = abs(pytorch_predictions.detach().numpy() - onnx_output).max()
            print(f"   最大差异: {diff:.6f}")
            if diff < 1e-4:
                print("   [OK] PyTorch和ONNX输出一致")
            else:
                print("   [!] 存在差异（可能由于数值精度）")
        else:
            print(f"   [!] 输出形状不匹配：PyTorch={pytorch_predictions.shape}, ONNX={onnx_output.shape}")
            print("   （可能需要调整ONNX导出配置）")

        return True

    except Exception as e:
        print(f"\n错误: 转换失败 - {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    # 模型路径
    model_path = r".\model_save\MAML_PU_test_best"
    output_path = r".\model_save\MAML_PU_test_best.onnx"

    # 输入形状 (batch_size, channels, height, width)
    # 灰度图: channels=1, 64x64图像
    input_shape = (1, 1, 64, 64)

    # 转换为ONNX
    success = pytorch_to_onnx(
        model_path=model_path,
        output_path=output_path,
        input_shape=input_shape,
        opset_version=11
    )

    if success:
        print(f"\n\nONNX模型已保存到: {os.path.abspath(output_path)}")
        print("\n下一步：")
        print("1. 使用ONNX Runtime进行推理加速")
        print("2. 部署到生产环境或边缘设备")
        print("3. 使用Netron可视化模型: https://netron.app/")
    else:
        print("\n转换失败，请检查错误信息。")
        exit(1)


if __name__ == '__main__':
    main()
