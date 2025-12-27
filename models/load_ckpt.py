import torch


def load_checkpoint(model, checkpoint_pth: str, device, model_name=None):
    """加载预训练权重，并打印详细匹配信息，同时冻结已加载参数"""
    ckpt = torch.load(checkpoint_pth, map_location=device)

    if "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    model_state = model.state_dict()

    # ==== 匹配情况统计 ====
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("\n=== 权重加载报告 ===")
    print(f"总参数数: {len(model_state)}")
    print(f"✅ 成功加载: {len(model_state) - len(missing)}")
    print(f"❌ 未加载 (模型里有但权重缺失): {len(missing)}")
    print(f"⚠️ 未使用 (权重里有但模型不需要): {len(unexpected)}")

    print_detail = False
    if print_detail:
        # ✅ 已加载参数
        loaded_keys = set(model_state.keys()) - set(missing)
        print("\n--- 成功加载的参数 ---")
        for k in sorted(loaded_keys):
            print(f"✅[LOADED] {k} {tuple(model_state[k].shape)}")

        # ❌ 缺失的参数
        if missing:
            print("\n--- 缺失的参数 (模型需要但 ckpt 没有) ---")
            for k in missing:
                print(f"❌[MISSING] {k} {tuple(model_state[k].shape)}")

        # ⚠️ 未使用的参数
        if unexpected:
            print("\n--- 未使用的参数 (ckpt 有但模型不需要) ---")
            for k in unexpected:
                print(f"⚠️[UNUSED] {k} {tuple(state_dict[k].shape)}")


    if model_name == "sam2_adapter_tiny" or model_name == "mobile_sam_adapter":
        for name, para in model.named_parameters():
            if "image_encoder" in name and "prompt_generator" not in name:
                para.requires_grad_(False)
            # elif "mask_decoder" in name:
            #     para.requires_grad_(False)

    model_total_params = sum(p.numel() for p in model.parameters())
    model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('model_grad_params:', model_grad_params, '\nmodel_total_params:', model_total_params)
