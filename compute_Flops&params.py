import torch
from ptflops import get_model_complexity_info
from models import model_dict

def test_model_flops(model_name, input_size=(3, 1024, 1024), detail=False):
    # 从字典中实例化模型
    if model_name not in model_dict:
        raise ValueError(f"Model {model_name} not found! Available: {list(model_dict.keys())}")
    model = model_dict[model_name]().cuda()

    # 计算 FLOPs 和 Params
    flops, params = get_model_complexity_info(
        model, input_size,
        as_strings=True,
        print_per_layer_stat=detail,  # ✅ 打印每一层
        verbose=detail
    )
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Input size: {input_size}")
    print(f"Total FLOPs: {flops}")
    print(f"Total Params: {params}")
    print("=" * 60)

    # 前向推理，检查输出
    x = torch.randn(2, *input_size).cuda()

    model = model.cuda()
    with torch.no_grad():
        out = model(x)

    if isinstance(out, (tuple, list)):
        print("Output contains multiple tensors:")
        for i, o in enumerate(out):
            print(f" - Output[{i}] shape: {o.shape}")
    elif isinstance(out, dict):
        print("Output is a dict:")
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                print(f" - {k}: {v.shape}")
            else:
                print(f" - {k}: {type(v)}")
    else:
        print("Output shape:", out.shape)

if __name__ == "__main__":
    # 总 FLOPs / Params
    # test_model_flops('sam2_adapter_tiny', input_size=(3, 1024, 1024), detail=False)

    # 打印每一层 FLOPs / Params
    print("\n>>> Per-layer detail:")
    img_size = 1024
    test_model_flops('mobile_sam_adapter', input_size=(3, img_size, img_size), detail=True)
    '''
    choices=['PVMNet', 'sam2_adapter_tiny', 'mobile_sam_adapter', 'SegFormerB0', 'SegFormerB1']
    '''