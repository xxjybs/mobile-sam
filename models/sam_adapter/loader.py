# sam_adapter_loader.py

import torch
import yaml
from models.sam_adapter.modeling.models import make as make_model

def sam_adapter_loader(config_path):

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)


    model_spec = config['model']
    model = make_model(model_spec).cuda()

    if config.get('sam_checkpoint') is not None:
        sam_checkpoint = torch.load(config['sam_checkpoint'])
        # 加载权重并获得加载结果
        load_result = model.load_state_dict(sam_checkpoint, strict=False)

        # 获取未加载成功的参数名
        missing_keys = set(load_result.missing_keys)
        unexpected_keys = set(load_result.unexpected_keys)

        print("\n�� 模型各参数加载情况：")
        for name, param in model.named_parameters():
            if name in missing_keys:
                print(f"❌ {name:<60} ⟶ [未加载]")
            else:
                print(f"✅ {name:<60} ⟶ [已加载]")

        # 可选：提示有哪些额外的（不在模型中的）权重
        if unexpected_keys:
            print("\n⚠️ 以下参数在 checkpoint 中存在但模型中没有使用：")
            for key in unexpected_keys:
                print(f"   - {key}")

    for name, para in model.named_parameters():
        if "image_encoder" in name and "prompt_generator" not in name:
            para.requires_grad_(False)

    model_total_params = sum(p.numel() for p in model.parameters())
    model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('model_grad_params:', model_grad_params, '\nmodel_total_params:', model_total_params)

    return model
from ptflops import get_model_complexity_info
if __name__ == "__main__":

    model = sam_adapter_loader(config_path="./cod-sam-vit-l.yaml")
    print(model)
    # model = build_sam_vit_t_encoder().cuda()
    size = 1024
    input_size = (3, size, size)
    detail = False
    # 计算 FLOPs 和 Params
    flops, params = get_model_complexity_info(
        model, input_size,
        as_strings=True,
        print_per_layer_stat=detail,  # ✅ 打印每一层
        verbose=detail
    )
    print("=" * 60)
    print(f"Input size: {input_size}")
    print(f"Total FLOPs: {flops}")
    print(f"Total Params: {params}")
    print("=" * 60)

    # 前向推理，检查输出
    x = torch.randn(1, *input_size).cuda()
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

