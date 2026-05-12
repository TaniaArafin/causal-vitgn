"""Training loop for Causal-VITGN."""


import os
import time
from typing import Dict


import torch
from torch.optim import Adam
from torch.utils.data import DataLoader


from src.models.causal_vitgn import CausalVITGN
from src.training.losses import CausalVITGNLoss




class Trainer:
   def __init__(
       self,
       config: Dict,
       train_loader: DataLoader,
       val_loader: DataLoader,
   ):
       self.config = config
       self.train_loader = train_loader
       self.val_loader = val_loader


       self.device = self._select_device()
       self.model = CausalVITGN(config).to(self.device)


       c = config["causal"]
       self.criterion = CausalVITGNLoss(
           dag_penalty=c["dag_penalty"],
           sparsity_penalty=c["sparsity_penalty"],
           consistency_penalty=c["consistency_penalty"],
       )


       t = config["training"]
       self.optimizer = Adam(self.model.parameters(), lr=t["learning_rate"])
       self.epochs = t["epochs"]
       self.beta_anneal_epochs = t["beta_anneal_epochs"]
       self.grad_clip = t["grad_clip"]
       self.save_dir = config["logging"]["save_dir"]
       self.patience = t["early_stop_patience"]


       os.makedirs(self.save_dir, exist_ok=True)
       self.best_val_loss = float("inf")
       self.bad_epochs = 0


   def _select_device(self) -> torch.device:
       if torch.cuda.is_available():
           return torch.device("cuda")
       if torch.backends.mps.is_available():
           return torch.device("mps")
       return torch.device("cpu")


   def beta_schedule(self, epoch: int) -> float:
       return min(1.0, (epoch + 1) / max(self.beta_anneal_epochs, 1))


   def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
       return {
           k: v.to(self.device) if torch.is_tensor(v) else v
           for k, v in batch.items()
       }


   def train_epoch(self, epoch: int) -> Dict[str, float]:
       self.model.train()
       beta = self.beta_schedule(epoch)
       running = {"total": 0.0, "nll": 0.0, "kl": 0.0, "dag": 0.0, "sparsity": 0.0}
       steps = 0


       skipped = 0
       for batch in self.train_loader:
           batch = self._to_device(batch)
           self.optimizer.zero_grad()
           outputs = self.model(batch)
           losses = self.criterion(outputs, batch, self.model.causal_layer, beta)


           # Skip any batch whose loss is non-finite (NaN/Inf) so a single
           # bad batch cannot brick the whole training run.
           total_loss = losses["total"]
           if not torch.isfinite(total_loss):
               skipped += 1
               continue


           total_loss.backward()
           torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
           self.optimizer.step()


           for k in running:
               val = losses[k]
               running[k] += val.detach().item() if torch.is_tensor(val) else float(val)
           steps += 1


       if skipped > 0:
           print(f"  (skipped {skipped} non-finite batches this epoch)")
       return {k: v / max(steps, 1) for k, v in running.items()}


   @torch.no_grad()
   def validate(self) -> float:
      
       self.model.eval()
       total = 0.0
       steps = 0
       for batch in self.val_loader:
           batch = self._to_device(batch)
           outputs = self.model(batch)
           losses = self.criterion(
               outputs, batch, self.model.causal_layer, beta=1.0
           )
           total += float(losses["total"])
           steps += 1
       return total / max(steps, 1)


   def save_checkpoint(self, epoch: int, val_loss: float, name: str = "best_model.pt"):
       path = os.path.join(self.save_dir, name)
       torch.save(
           {
               "epoch": epoch,
               "model_state_dict": self.model.state_dict(),
               "optimizer_state_dict": self.optimizer.state_dict(),
               "val_loss": val_loss,
               "config": self.config,
           },
           path,
       )


   def fit(self):
       print(f"Training on device: {self.device}")
       for epoch in range(self.epochs):
           start = time.time()
           train_metrics = self.train_epoch(epoch)
           val_loss = self.validate()
           elapsed = time.time() - start


           print(
               f"Epoch {epoch:03d} | "
               f"train_total={train_metrics['total']:.3f} | "
               f"val_total={val_loss:.3f} | "
               f"nll={train_metrics['nll']:.3f} | "
               f"kl={train_metrics['kl']:.3f} | "
               f"dag={train_metrics['dag']:.6f} | "
               f"time={elapsed:.1f}s"
           )


           if val_loss < self.best_val_loss:
               self.best_val_loss = val_loss
               self.bad_epochs = 0
               self.save_checkpoint(epoch, val_loss)
               print(f"  -> saved checkpoint (val={val_loss:.3f})")
           else:
               self.bad_epochs += 1
               if self.bad_epochs >= self.patience:
                   print(f"Early stopping at epoch {epoch}.")
                   break





