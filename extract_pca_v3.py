import torch
import argparse
import numpy as np
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.dataset import Compose

def parse_args():
    parser = argparse.ArgumentParser(description='Extract PCA bases for GFL (MMDet 3.x)')
    parser.add_argument('config', help='Phase 1 的配置文件路径')
    parser.add_argument('checkpoint', help='Phase 1 训练好的 .pth 模型路径')
    parser.add_argument('--save-path', default='pca_bases.pt', help='保存路径')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. 加载配置
    cfg = Config.fromfile(args.config)
    
    # 2. 构建 Runner (这是 3.x 获取模型和数据的最快方式)
    # 我们不需要 work_dir, 只是借用 runner 来 build 组件
    runner = Runner.from_cfg(cfg)
    
    # 3. 加载权重
    runner.load_checkpoint(args.checkpoint)
    model = runner.model
    model.eval()
    
    # 4. 获取 Dataloader
    # 注意：这里使用 train_dataloader，因为我们要分析训练数据的分布
    dataloader = runner.train_dataloader
    
    # 5. 定义目标层 (根据 GFL 结构调整)
    target_layers = [
        'backbone.layer1', 'backbone.layer2', 
        'backbone.layer3', 'backbone.layer4',
        'neck.fpn_convs.0.conv', # FPN P3
        'neck.fpn_convs.1.conv', # FPN P4
        'neck.fpn_convs.2.conv',  # FPN P5
        'neck.fpn_convs.3.conv',
        'neck.fpn_convs.4.conv',
        'bbox_head.cls_convs.0.conv',
        'bbox_head.cls_convs.1.conv',
        'bbox_head.cls_convs.2.conv',
        'bbox_head.cls_convs.3.conv',
        'bbox_head.reg_convs.0.conv',
        'bbox_head.reg_convs.1.conv',
        'bbox_head.reg_convs.2.conv',
        'bbox_head.reg_convs.3.conv'

    ]
    
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            if isinstance(output, tuple): output = output[0]
            activations[name] = output.detach()
        return hook

    # 注册 Forward Hooks
    handles = []
    for name, module in model.named_modules():
        if name in target_layers:
            handles.append(module.register_forward_hook(get_activation(name)))
            print(f"Registered hook for: {name}")
    
    # 6. 收集特征并计算协方差
    cov_matrices = {}
    print("Start scanning Phase 1 data...")
    
    max_iters = 200
    for i, data_batch in enumerate(dataloader):
        if i >= max_iters: break
        
        with torch.no_grad():
            # MMEngine 的 val_step 或 test_step 需要特定格式
            # 这里我们手动把数据送入 model 的 data_preprocessor 再 forward
            data_batch = model.data_preprocessor(data_batch, training=False)
            # 调用 backbone 和 neck 提取特征
            # GFL 通常是 extract_feat -> bbox_head
            # 我们只需要跑一遍 forward 触发 hook 即可
            model(inputs=data_batch['inputs'], data_samples=data_batch['data_samples'], mode='predict')
        
        for name in target_layers:
            if name not in activations: continue
            feat = activations[name]
            # [B, C, H, W] -> [N, C]
            b, c, h, w = feat.shape
            feat_reshaped = feat.permute(0, 2, 3, 1).reshape(-1, c)
            
            # 计算 X^T * X
            update = torch.matmul(feat_reshaped.T, feat_reshaped)
            
            if name not in cov_matrices:
                cov_matrices[name] = update
            else:
                cov_matrices[name] += update
        
        if (i+1) % 20 == 0: print(f"Processed {i+1} batches...")

    # 7. SVD 分解
    pca_bases = {}
    threshold = 0.90
    
    print("Computing SVD...")
    for name, mat in cov_matrices.items():
        eigenvalues, eigenvectors = torch.linalg.eigh(mat)
        # 翻转为降序
        eigenvalues = torch.flip(eigenvalues, [0]) # 行翻转
        eigenvectors = torch.flip(eigenvectors, [1]) # 列翻转
        
        total = torch.sum(eigenvalues)
        running = 0
        k = 0
        for idx in range(len(eigenvalues)):
            running += eigenvalues[idx]
            if running / total >= threshold:
                k = idx + 1
                break
        
        pca_bases[name] = eigenvectors[:, :k].cpu()
        print(f"Layer {name}: Kept {k}/{len(eigenvalues)} bases.")

    torch.save(pca_bases, args.save_path)
    print(f"Done! Saved to {args.save_path}")

if __name__ == '__main__':
    main()