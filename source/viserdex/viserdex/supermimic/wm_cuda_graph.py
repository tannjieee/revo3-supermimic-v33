"""Capture WM loss/backward without changing Adam, clipping or replay sampling."""
import torch


class CapturedWMLoss:
    def __init__(self, trainer, batch):
        if torch.device(trainer.device).type != "cuda" or trainer.distributed.enabled:
            raise ValueError("WM CUDA capture requires single-GPU execution")
        self.trainer = trainer
        self.inputs = {k: v.clone() for k, v in batch.items()}
        self.parameters = tuple(trainer.wm.parameters())
        stream = torch.cuda.Stream(device=trainer.device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                trainer.opt_wm.zero_grad(set_to_none=False)
                warmup = trainer._wm_loss(self.inputs)
                warmup[0].backward()
                if trainer.cfg.dex_reference:
                    trainer._independent_failure_loss.backward()
                    trainer._independent_failure_loss = None
                    trainer.opt_failure.zero_grad(set_to_none=False)
            trainer.opt_wm.zero_grad(set_to_none=False)
        torch.cuda.current_stream().wait_stream(stream)
        self.grads = tuple(p.grad for p in self.parameters)
        if any(g is None for g in self.grads):
            raise ValueError("Every WM parameter must participate in captured loss")
        import gc
        gc.collect()
        torch.cuda.synchronize(trainer.device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.outputs = trainer._wm_loss(self.inputs)
            self.outputs[0].backward()
            if trainer.cfg.dex_reference:
                trainer._independent_failure_loss.backward()
        if trainer.cfg.dex_reference:
            trainer._independent_failure_loss = None

    def __call__(self, batch):
        if batch.keys() != self.inputs.keys():
            raise ValueError("Captured WM batch schema changed")
        for key, value in batch.items():
            target = self.inputs[key]
            if value.shape != target.shape or value.dtype != target.dtype or value.device != target.device:
                raise ValueError(f"Captured WM input changed: {key}")
            target.copy_(value)
        # Policy updates may set WM gradients to None. Reattach the exact
        # addresses captured by backward, then clear before this optimizer step.
        for parameter, grad in zip(self.parameters, self.grads):
            parameter.grad = grad
        self.trainer.opt_wm.zero_grad(set_to_none=False)
        if self.trainer.cfg.dex_reference:
            self.trainer.opt_failure.zero_grad(set_to_none=False)
        self.graph.replay()
        return self.outputs
