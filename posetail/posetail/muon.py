"""DualOptimizer: present two optimizers (official torch.optim.Muon for 2D hidden matrices +
AdamW for everything else) as a single optimizer to the training loop.

torch.optim.Muon is strictly 2D-only (raises on non-2D params), so it must be paired with a
standard optimizer for biases/embeddings/heads/norms. This thin wrapper delegates the few methods
the loop/train_iteration use (param_groups, step, zero_grad, state_dict) so the rest of train.py is
unchanged. Gradient clipping for the dual case is done directly on model.parameters() in
train_iteration (detected via the `_opts` attribute), since fabric.clip_gradients takes one optimizer.
"""


class DualOptimizer:
    def __init__(self, opt_muon, opt_adam, muon_warmup_steps=0):
        self.opt_muon = opt_muon
        self.opt_adam = opt_adam
        self._opts = [opt_muon, opt_adam]
        # Optional linear LR warmup for the Muon groups (the AdamW-SF half warms up on its own).
        # Muon hits full LR from step 0 by default, which destabilizes a *higher* encoder Muon LR
        # on a converged init; warmup lets us push the encoder Muon LR up for faster adaptation.
        self.muon_warmup_steps = max(0, int(muon_warmup_steps))
        self._muon_base_lrs = [g['lr'] for g in self.opt_muon.param_groups]
        self._gstep = 0

    @property
    def param_groups(self):
        return list(self.opt_muon.param_groups) + list(self.opt_adam.param_groups)

    @property
    def state(self):
        # Merged per-parameter state of both inner optimizers. Needed so the resume path in
        # load_checkpoint (`for state in optimizer.state.values(): ... v.to(device)`) works on
        # the dual wrapper. Keys are param tensors, disjoint across the two optimizers.
        merged = {}
        for o in self._opts:
            merged.update(getattr(o, 'state', {}))
        return merged

    def train(self):
        # schedule-free optimizers swap params to the y (train) point; delegate to any that support it.
        for o in self._opts:
            if hasattr(o, 'train'):
                o.train()

    def eval(self):
        # schedule-free optimizers swap params to the x (averaged/eval) point.
        for o in self._opts:
            if hasattr(o, 'eval'):
                o.eval()

    def zero_grad(self, *a, **k):
        for o in self._opts:
            o.zero_grad(*a, **k)

    def step(self, *a, **k):
        if self.muon_warmup_steps > 0:
            self._gstep += 1
            f = min(1.0, self._gstep / self.muon_warmup_steps)
            for g, base in zip(self.opt_muon.param_groups, self._muon_base_lrs):
                g['lr'] = base * f
        for o in self._opts:
            o.step(*a, **k)

    def state_dict(self):
        return {'muon': self.opt_muon.state_dict(), 'adam': self.opt_adam.state_dict()}

    def load_state_dict(self, sd):
        self.opt_muon.load_state_dict(sd['muon'])
        self.opt_adam.load_state_dict(sd['adam'])
