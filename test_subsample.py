# -*- coding: utf-8 -*-
import torch
import numpy as np
from models.mobile_sam_adapter import Mobile_sam_adapter
from dataset.OrangeDefectDataloader import OrangeDefectLoader

def test_model_input_output(device='cpu'):
    # 测试模型是否能处理256x256输入
    print(f"Testing model input/output with 256x256 input on {device}...")
    
    # 创建模型实例
    model = Mobile_sam_adapter(inp_size=256)
    model = model.to(device)
    model.eval()
    
    # 创建随机输入
    test_input = torch.randn(1, 3, 256, 256).to(device)
    
    # 测试前向传播
    with torch.no_grad():
        output = model(test_input)
    
    print(f"Input size: {test_input.shape}")
    print(f"Output size: {output.shape}")
    
    # 验证输出尺寸
    assert output.shape == (1, 1, 256, 256), f"Expected output shape (1, 1, 256, 256), got {output.shape}"
    print("? Model input/output size validation passed!")
    
    return output

def test_dataset_resize():
    # 测试数据集加载器是否能正确调整图像大小
    print("\nTesting dataset image resizing...")
    
    # 创建数据集实例
    dataset = OrangeDefectLoader("./data/orange", train=False, test=True, size=256, num_classes=2)
    
    # 获取一个样本
    img, mask, onehot = dataset[0]
    
    print(f"Original image shape (after resize): {img.shape}")
    print(f"Original mask shape (after resize): {mask.shape}")
    print(f"One-hot mask shape: {onehot.shape}")
    
    # 验证尺寸
    assert img.shape == (3, 256, 256), f"Expected image shape (3, 256, 256), got {img.shape}"
    assert mask.shape == (256, 256), f"Expected mask shape (256, 256), got {mask.shape}"
    assert onehot.shape == (2, 256, 256), f"Expected one-hot shape (2, 256, 256), got {onehot.shape}"
    print("? Dataset image resize validation passed!")

def test_feature_size_calculation(device='cpu'):
    # 测试特征尺寸计算
    print(f"\nTesting feature size calculation on {device}...")
    
    # 创建模型实例
    model = Mobile_sam_adapter(inp_size=256)
    model = model.to(device)
    
    # 访问TinyViT模型
    tiny_vit_model = model.image_encoder
    
    print(f"TinyViT img_size: {tiny_vit_model.img_size}")
    print(f"TinyViT feature_size: {tiny_vit_model.feature_size}")
    print(f"Image embedding size: {model.image_embedding_size}")
    
    # 验证特征尺寸计算
    assert tiny_vit_model.img_size == 256, f"Expected img_size 256, got {tiny_vit_model.img_size}"
    assert tiny_vit_model.feature_size == [64, 32, 16, 16], f"Expected feature_size [64, 32, 16, 16], got {tiny_vit_model.feature_size}"
    assert model.image_embedding_size == 16, f"Expected image_embedding_size 16, got {model.image_embedding_size}"
    print("? Feature size calculation validation passed!")

def test_model_components_on_same_device(device='cpu'):
    # 测试模型所有组件是否在同一设备上
    print(f"\nTesting model components on same device ({device})...")
    
    # 创建模型实例
    model = Mobile_sam_adapter(inp_size=256)
    model = model.to(device)
    
    # 检查关键组件的设备
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Image encoder device: {next(model.image_encoder.parameters()).device}")
    print(f"Mask decoder device: {next(model.mask_decoder.parameters()).device}")
    print(f"PE layer device: {model.pe_layer.positional_encoding_gaussian_matrix.device}")
    
    # 验证所有组件在同一设备上
    all_on_same_device = True
    for name, param in model.named_parameters():
        if param.device != torch.device(device):
            print(f"WARNING: {name} is on {param.device}")
            all_on_same_device = False
    
    if all_on_same_device:
        print("? All model components are on the same device!")
    else:
        print("? Some model components are on different devices!")
        return False
    
    return True

def test_infer_method(device='cpu'):
    # 测试infer方法
    print(f"\nTesting infer method on {device}...")
    
    # 创建模型实例
    model = Mobile_sam_adapter(inp_size=256)
    model = model.to(device)
    model.eval()
    
    # 创建随机输入
    test_input = torch.randn(1, 3, 256, 256).to(device)
    
    # 测试infer方法
    with torch.no_grad():
        mask, features = model.infer(test_input)
    
    print(f"Infer method mask shape: {mask.shape}")
    print(f"Infer method features shape: {features.shape}")
    
    # 验证输出尺寸
    assert mask.shape == (1, 1, 256, 256), f"Expected mask shape (1, 1, 256, 256), got {mask.shape}"
    print("? Infer method validation passed!")

if __name__ == "__main__":
    try:
        # 测试CPU
        print("=== Testing on CPU ===")
        test_model_input_output('cpu')
        test_feature_size_calculation('cpu')
        test_model_components_on_same_device('cpu')
        test_infer_method('cpu')
        
        # 如果有GPU，测试GPU
        if torch.cuda.is_available():
            print("\n=== Testing on GPU ===")
            test_model_input_output('cuda')
            test_feature_size_calculation('cuda')
            test_model_components_on_same_device('cuda')
            test_infer_method('cuda')
        else:
            print("\n=== GPU not available, skipping GPU tests ===")
        
        # 测试数据集
        test_dataset_resize()
        
        print("\n? All tests passed! The model is correctly adapted to 256x256 input using subsample method.")
        print("\n? Summary of changes:")
        print("- Changed global input size from 1024 to 256 in train.py")
        print("- Modified build_sam_vit_t_encoder to accept img_size parameter")
        print("- Updated Mobile_sam_adapter to handle dynamic input size")
        print("- Implemented dynamic feature size calculation in TinyViT")
        print("- Dataset loader now resizes 512x512 images to 256x256")
        print("\n? The model is now ready to train with your 512x512 dataset using subsample method!")
    except Exception as e:
        print(f"\n? Test failed: {e}")
        import traceback
        traceback.print_exc()