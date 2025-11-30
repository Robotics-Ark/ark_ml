import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from contextlib import nullcontext
from arkml.core.algorithm import Trainer
from arkml.core.policy import BasePolicy
from arkml.algos.vla.pi05.models import flow_matching_loss
from tqdm import tqdm


class Pi05Trainer(Trainer):
    """
    Trainer class for Pi0.5 with stage-based training.
    """

    def __init__(
        self,
        model: BasePolicy,
        dataloader: DataLoader,
        device: str,
        lr: float,
        weight_decay: float,
        num_epochs: int,
        grad_accum: float,
        output_dir: str,
        use_bf16: bool,
        flow_alpha: float = 10.0,  # Weight for flow matching loss
        *,
        val_dataloader = None,
        eval_every: int = 1,
    ):
        self.model = model.to_device(device)
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.eval_every = max(1, int(eval_every))
        self.device = device
        self.num_epochs = num_epochs
        self.grad_accum = max(1, int(grad_accum))
        self.output_dir = output_dir
        self.flow_alpha = flow_alpha  # Weight for flow matching loss

        # Get trainable parameters
        self.trainable_params = self.model.get_trainable_params()

        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.trainable_params, lr=lr, weight_decay=weight_decay
        )

        # Device/AMP setup
        device_str = str(device)
        self.device_type = (
            "cuda"
            if torch.cuda.is_available()
            and (device_str.startswith("cuda") or getattr(device, "type", "") == "cuda")
            else "cpu"
        )
        self.use_bf16 = use_bf16
        # GradScaler only for CUDA fp16
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(self.device_type == "cuda" and not self.use_bf16)
        )

    def train_step_pretrain(self, batch):
        """
        Training step for pretraining stage:
        CE(text) + CE(FAST tokens)
        """
        # Extract relevant tensors from batch
        prefix_tokens = batch.get("prefix_tokens", None)
        target_tokens = batch.get("target_tokens", None)
        modality = batch.get("modality", None)
        actions_cont = batch.get("actions_cont", None)

        # Calculate cross-entropy loss for text tokens (subtask/qa/etc.)
        text_loss = 0.0
        if prefix_tokens is not None and target_tokens is not None:
            # Use a simple approach where prefix_tokens are used to predict target_tokens
            # This would require the model to have a text prediction head
            # For now, we'll focus on the FAST token loss
            pass

        # Calculate cross-entropy loss for FAST tokens if this is a robot action modality
        fast_loss = 0.0
        if modality is not None and actions_cont is not None:
            # Forward pass
            loss = self.model.forward(batch)
            # The model's forward method already handles the loss calculation
            # For pretrain, this would be based on FAST token prediction
            fast_loss = loss

        # Total pretrain loss
        total_loss = fast_loss

        return total_loss

    def train_step_posttrain(self, batch):
        """
        Training step for posttraining stage:
        CE(subtask) + alpha * flow_matching_loss
        """
        # Extract relevant tensors from batch
        prefix_tokens = batch.get("prefix_tokens", None)
        target_tokens = batch.get("target_tokens", None)
        modality = batch.get("modality", None)
        actions_cont = batch.get("actions_cont", None)

        # Get model prediction
        loss = self.model.forward(batch)

        # The model forward already includes flow matching loss when action is provided
        # We need to separately compute the subtask loss if applicable
        subtask_loss = 0.0
        flow_loss = 0.0

        # Extract flow loss specifically if we have action data
        if modality is not None and "action" in batch and actions_cont is not None:
            # This would be handled in the model's forward pass
            # For posttrain, we want to ensure flow matching loss is properly weighted
            pass

        # Total posttrain loss: subtask_loss + alpha * flow_loss
        # For now, we'll use the loss from the model forward pass
        # In a full implementation, we'd separate the losses
        total_loss = loss

        return total_loss

    def train(self, stage: str = "pretrain"):
        """
        Main training loop that switches behavior based on training stage.
        """
        self.model.set_train_mode()

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            self.optimizer.zero_grad(set_to_none=True)

            progress_bar = tqdm(
                enumerate(self.dataloader),
                total=len(self.dataloader),
                desc=f"{stage} Epoch {epoch + 1}/{self.num_epochs}",
                leave=False,
            )

            for i, batch in progress_bar:
                # Choose autocast context
                if self.device_type == "cuda":
                    ac_dtype = torch.bfloat16 if self.use_bf16 else torch.float16
                    ac = torch.autocast("cuda", dtype=ac_dtype)
                else:
                    ac = (
                        torch.autocast("cpu", dtype=torch.bfloat16)
                        if self.use_bf16
                        else nullcontext()
                    )

                with ac:
                    if stage == "pretrain":
                        loss = self.train_step_pretrain(batch)
                    elif stage == "posttrain":
                        loss = self.train_step_posttrain(batch)
                    else:
                        # Default to pretrain behavior for unknown stages
                        loss = self.train_step_pretrain(batch)

                # Gradient accumulation
                loss_to_backprop = loss / self.grad_accum

                if self.device_type == "cuda" and not self.use_bf16:
                    self.scaler.scale(loss_to_backprop).backward()
                else:
                    loss_to_backprop.backward()

                step_now = ((i + 1) % self.grad_accum == 0) or (
                    i + 1 == len(self.dataloader)
                )
                if step_now:
                    if self.device_type == "cuda" and not self.use_bf16:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.trainable_params, max_norm=1.0
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.trainable_params, max_norm=1.0
                        )
                        self.optimizer.step()

                    self.optimizer.zero_grad(set_to_none=True)

                epoch_loss += float(loss.item())
                num_batches += 1

                progress_bar.set_postfix({"loss": loss.item()})

            avg_epoch_loss = epoch_loss / max(1, num_batches)
            print(f"[{stage} epoch {epoch + 1}] loss={avg_epoch_loss:.6f}")

    def save_checkpoints(self, epoch: int):
        """
        Save backbone and flow expert checkpoints separately.
        """
        # Create epoch-specific directory
        epoch_dir = os.path.join(self.output_dir, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)

        # Save backbone separately
        backbone_path = os.path.join(epoch_dir, "backbone.pth")
        if hasattr(self.model, 'backbone'):
            torch.save(self.model.backbone.state_dict(), backbone_path)
            print(f"[checkpoint] Saved backbone to {backbone_path}")

        # Save flow expert separately
        flow_expert_path = os.path.join(epoch_dir, "flow_expert.pth")
        if hasattr(self.model, 'flow_head'):
            torch.save(self.model.flow_head.state_dict(), flow_expert_path)
            print(f"[checkpoint] Saved flow expert to {flow_expert_path}")

        # Save full model
        full_model_path = os.path.join(epoch_dir, "full_model.pth")
        torch.save(self.model.state_dict(), full_model_path)
        print(f"[checkpoint] Saved full model to {full_model_path}")

    def fit(self, *args, **kwargs):
        """
        Run the complete training process based on training stage from config.
        """
        # Get training stage from model config or use default
        training_stage = getattr(self.model, 'training_stage', 'pretrain')

        print(f"Starting training in {training_stage} stage")

        # Perform training based on stage
        if training_stage == "pretrain":
            self.train(stage="pretrain")
        elif training_stage == "posttrain":
            self.train(stage="posttrain")
        else:
            # Handle combined training if needed
            print(f"Unknown stage {training_stage}, defaulting to pretrain")
            self.train(stage="pretrain")

        # Save final checkpoints
        self.save_checkpoints("final")

        return {"status": "completed", "final_stage": training_stage}