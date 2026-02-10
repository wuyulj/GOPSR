# my_hooks/projection_hook.py
import torch
from mmengine.hooks import Hook
from mmdet.registry import HOOKS
import os

@HOOKS.register_module()
class GradientProjectionHook(Hook):
    def __init__(self, pca_bases_path):
        super().__init__()
        self.pca_bases_path = pca_bases_path
        self.bases = None

    def before_run(self, runner):
        """在训练开始前，加载基底并注册 tensor hook"""
        # 确保路径存在
        if not os.path.exists(self.pca_bases_path):
            raise FileNotFoundError(f"PCA bases file not found at: {self.pca_bases_path}")

        print(f"\n>>> [GradientProjectionHook] Loading PCA bases from {self.pca_bases_path}...")
        self.bases = torch.load(self.pca_bases_path)
        
        # 将基底移动到模型所在的设备 (通常是 cuda:0)
        # 注意：runner.model 可能是 DistributedDataParallel，所以取 parameters 来确定设备
        device = next(runner.model.parameters()).device
        for k, v in self.bases.items():
            self.bases[k] = v.to(device)

        # 遍历模型参数，注册 backward hook
        count = 0
        for name, param in runner.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # 这里的 name 可能会包含 "module." (DDP) 或 "backbone." 等前缀
            # 我们需要模糊匹配：只要 param 的名字包含我们在 Phase 1 保存的 key 即可
            matched_key = None
            for key in self.bases.keys():
                # 必须匹配层名，且必须是权重 (weight)，bias 不需要投影
                if key in name and 'weight' in name:
                    matched_key = key
                    break
            
            if matched_key:
                # 定义投影函数 (闭包)
                # 这个函数会接收原始梯度 grad，返回修正后的梯度
                # 使用 default argument 绑定当前的 basis，防止闭包变量泄露
                def get_projection_func(basis):
                    def projection_func(grad):
                        # grad shape: [Out, In, k, k]
                        # basis shape: [In, K]
                        
                        # 保护性代码：如果梯度形状不对（例如 bias），直接返回
                        if grad.dim() < 2: 
                            return grad
                        
                        original_shape = grad.shape
                        out_c, in_c = original_shape[0], original_shape[1]
                        
                        # 确保 basis 的维度和梯度的输入维度匹配
                        if basis.shape[0] != in_c:
                            # 这种情况极少发生，除非网络结构变了
                            return grad

                        # Reshape: [Out*k*k, In]
                        grad_flat = grad.view(-1, in_c)
                        
                        # P = G * M * M^T
                        # matmul 自动处理 broadcasting
                        # [N, In] x [In, K] -> [N, K]
                        proj_step1 = torch.matmul(grad_flat, basis)
                        # [N, K] x [K, In] -> [N, In]
                        proj = torch.matmul(proj_step1, basis.T)
                        
                        # G_new = G - P
                        grad_new = grad_flat - proj
                        
                        return grad_new.view(original_shape)
                    return projection_func

                # 注册 Hook
                # register_hook 返回一个 handle，虽然这里我们不存 handle，但它会生效
                param.register_hook(get_projection_func(self.bases[matched_key]))
                count += 1
        
        print(f">>> [GradientProjectionHook] Successfully registered backward hooks for {count} layers/weights.\n")