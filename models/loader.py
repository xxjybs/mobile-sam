from models.FastSegFormer.fast_segformer import *
import numpy as np
from models.sam_adapter.loader import sam_adapter_loader

def load_model(model_config):
    model_type = model_config['name']

    if model_type == 'FastSegFormers12':
        model = FastSegFormer(
            num_classes=model_config['num_classes'],
            pretrained=False,
            backbone="poolformer_s12",
            Pyramid="multiscale",
            fork_feat=True,
            cnn_branch=True
        )
        if model_config['pretrain']:
            model_path = 'checkpoints/poolformer_s12_ImageNet_1k_224x224.pth'
            print('Load weights {}.'.format(model_path))

            model_dict = model.state_dict()
            load_key, no_load_key, temp_dict = [], [], {}
            device = torch.device("cuda", 0)
            pretrained_dict = torch.load(model_path, map_location=device)
            backbone_stat_dict = {}
            for i in pretrained_dict.keys():
                backbone_stat_dict["common_backbone." + i] = pretrained_dict[i]
            pretrained_dict.update(backbone_stat_dict)
            for k, v in pretrained_dict.items():
                if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                    temp_dict[k] = v
                    load_key.append(k)
                else:
                    no_load_key.append(k)
            model_dict.update(temp_dict)
            model.load_state_dict(model_dict)
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")
        return model


    elif model_type == 'FastSegFormerV2_s0':
        model = FastSegFormer(
            num_classes=model_config['num_classes'],
            pretrained=False,
            backbone="efficientformerV2_s0",
            Pyramid="multiscale",
            fork_feat=True,
            cnn_branch=True
        )
        if model_config['pretrain']:
            model_path = 'checkpoints/EfficientformerV2_s0_ImageNet_1k_224x224.pth'
            print('Load weights {}.'.format(model_path))

            model_dict = model.state_dict()
            load_key, no_load_key, temp_dict = [], [], {}
            device = torch.device("cuda", 0)
            pretrained_dict = torch.load(model_path, map_location=device, weights_only=False)['model']
            backbone_stat_dict = {}
            for i in pretrained_dict.keys():
                backbone_stat_dict["common_backbone." + i] = pretrained_dict[i]
            pretrained_dict.update(backbone_stat_dict)
            for k, v in pretrained_dict.items():
                if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                    temp_dict[k] = v
                    load_key.append(k)
                else:
                    no_load_key.append(k)
            model_dict.update(temp_dict)
            model.load_state_dict(model_dict)
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")
        return model

    elif model_type == 'Sam-Adapter':
        model = sam_adapter_loader(
            config_path="models/sam_adapter/cod-sam-vit-l.yaml"
        )

        return model

    else:
        raise ValueError(f'Unknown model type: {model_type}')

