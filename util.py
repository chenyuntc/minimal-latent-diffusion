import random
import datetime
import os
import numpy as np
import torch
import torch.distributed as torchdist


class _Dist:
    @staticmethod
    def _run_dist() -> bool:
        return "WORLD_SIZE" in os.environ

    @staticmethod
    def init():
        """Initialize distributed training env if launched as distributed"""
        if _Dist._run_dist() and not torchdist.is_initialized():
            torchdist.init_process_group(
                torchdist.Backend.NCCL, timeout=datetime.timedelta(seconds=7200)
            )

    @staticmethod
    def size() -> int:
        return int(os.getenv("WORLD_SIZE", 1))

    @staticmethod
    def local_rank() -> int:
        return int(os.getenv("LOCAL_RANK", 0))

    @staticmethod
    def rank() -> int:
        return int(os.getenv("RANK", 0))

    @staticmethod
    def broadcast(tensor: torch.Tensor, src=0) -> torch.Tensor:
        """Broadcast tensor from rank 0 to all other ranks.
        
        Args:
            tensor: CUDA tensor to be broadcasted from rank 0
            src: Source rank to broadcast the tensor from
            
        Returns:
            Broadcasted tensor
        """
        if _Dist._run_dist():
            torchdist.broadcast(tensor, src=src, group=torchdist.GroupMember.WORLD)
        return tensor

    @staticmethod
    def allreduce(tensor: torch.Tensor) -> torch.Tensor:
        """All Reduce operation specifically for CUDA tensors.
        
        Args:
            tensor: CUDA tensor to be reduced
            
        Returns:
            Reduced tensor (average across all processes)
        """
        if _Dist._run_dist():
            torchdist.all_reduce(
                tensor, op=torchdist.ReduceOp.SUM, group=torchdist.GroupMember.WORLD
            )
            tensor.div_(_Dist.size())
        return tensor


# Create the namespace
dist = _Dist()


def dist_sync_grad(model: torch.nn.Module):
    """Average the gradients of a given model across all processes."""
    params = [param for param in model.parameters()]
    flat = torch.cat([param.grad.flatten() for param in params])
    flat = dist.allreduce(flat)
    flat = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    grads = flat.split([param.numel() for param in params])
    for param, grad in zip(params, grads):
        param.grad = grad.reshape(param.shape)


#
def set_seed_all(seed=42, rank=0):
    """Sets the seed across numpy, torch and the built-in random module."""
    seed = seed + rank * 1024
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
