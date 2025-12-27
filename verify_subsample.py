# -*- coding: utf-8 -*-
import torch
import numpy as np
from models.mobile_sam_adapter import Mobile_sam_adapter
from dataset.OrangeDefectDataloader import OrangeDefectLoader

def test_model_input_output():
    # 测试模型是否能处理256x256输入
    print("Testing model input/output with 256x256 input...")
    
    # 创建模型实例
    model = Mobile_sam_adapter(inp_size=256)
    model.eval()
    
    # 创建随机输入
    test_input = torch.randn(1, 3, 256, 256)
    
    # 测试前向传播
    with torch.no_grad():
        output = model(test_input)
    
    print(f"Input size: {test_input.shape}")
    print(f"Output size: {output.shape}")
    
    # 验证输出尺寸
    assert output.shape == (1, 1, 256, 256), f"Expected output shape (1, 1, 256, 256), got {output.shape}"
    print("? Model input/output size validation passed!")

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

def test_feature_size_calculation():
    # 测试特征尺寸计算
    print("\nTesting feature size calculation...")
    
    # 创建模型实例
    model = Mobile_sam_adapter(inp_size=256)
    
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

if __name__ == "__main__":
    try:
        test_model_input_output()
        test_dataset_resize()
        test_feature_size_calculation()
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
