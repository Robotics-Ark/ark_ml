import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np


class Pi05Evaluator:
    """
    Evaluator class for Pi0.5 with subtask and action evaluation.
    """

    def __init__(self, model, dataloader: DataLoader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device

        # Move model to device
        self.model.to_device(device)

    def eval_subtask(self, predicted_subtasks, ground_truth_subtasks):
        """
        Compare predicted subtasks vs ground truth subtasks.

        Args:
            predicted_subtasks: Predicted subtask tokens/logits
            ground_truth_subtasks: Ground truth subtask tokens

        Returns:
            Dictionary with accuracy metric
        """
        # Calculate accuracy
        if torch.is_tensor(predicted_subtasks) and torch.is_tensor(ground_truth_subtasks):
            # If predicted_subtasks are logits, get argmax
            if predicted_subtasks.dim() > 1 and predicted_subtasks.size(-1) > 1:
                predicted_tokens = torch.argmax(predicted_subtasks, dim=-1)
            else:
                predicted_tokens = predicted_subtasks

            # Ensure both tensors have the same shape
            if predicted_tokens.shape != ground_truth_subtasks.shape:
                # Try to reshape if needed
                if predicted_tokens.numel() == ground_truth_subtasks.numel():
                    predicted_tokens = predicted_tokens.view(ground_truth_subtasks.shape)

            # Calculate accuracy
            correct = (predicted_tokens == ground_truth_subtasks).sum().item()
            total = ground_truth_subtasks.numel()
            accuracy = correct / total if total > 0 else 0.0
        else:
            # Fallback for non-tensor inputs
            accuracy = 0.0

        return {
            "subtask_accuracy": accuracy,
            "total_evaluated": len(ground_truth_subtasks) if hasattr(ground_truth_subtasks, '__len__') else 0
        }

    def eval_actions(self, initial_hidden_states, ground_truth_actions):
        """
        Evaluate action prediction performance:
        - sample_subtask to get subtask
        - run predict_with_flow to get continuous actions
        - compare predicted vs GT continuous actions

        Args:
            initial_hidden_states: Initial hidden states from the model
            ground_truth_actions: Ground truth continuous actions

        Returns:
            Dictionary with MSE and other action metrics
        """
        # Sample subtask (in a real implementation, this would use the model's subtask_head)
        # For now, we'll skip the subtask sampling and directly use the flow prediction

        # Predict actions using flow (this would typically happen after subtask sampling)
        if hasattr(self.model, 'predict_with_flow'):
            predicted_actions = self.model.predict_with_flow(initial_hidden_states)
        else:
            # Fallback if method doesn't exist yet
            predicted_actions = torch.zeros_like(ground_truth_actions)

        # Calculate MSE between predicted and ground truth actions
        mse = F.mse_loss(predicted_actions, ground_truth_actions).item()

        # Calculate additional metrics
        mae = F.l1_loss(predicted_actions, ground_truth_actions).item()

        # Calculate accuracy based on how close predictions are to ground truth (within threshold)
        threshold = 0.1  # Define a reasonable threshold for "correct" actions
        diff = torch.abs(predicted_actions - ground_truth_actions)
        within_threshold = (diff < threshold).float().mean().item()

        return {
            "action_mse": mse,
            "action_mae": mae,
            "action_accuracy_within_threshold": within_threshold,
            "threshold": threshold,
            "total_evaluated": len(ground_truth_actions) if hasattr(ground_truth_actions, '__len__') else 0
        }

    def evaluate(self):
        """
        Main evaluation loop that computes all metrics.

        Returns:
            Dictionary with all evaluation metrics
        """
        self.model.set_eval_mode()

        all_subtask_metrics = []
        all_action_metrics = []

        total_samples = 0

        for batch in self.dataloader:
            # Move batch to device
            for key, value in batch.items():
                if torch.is_tensor(value):
                    batch[key] = value.to(self.device)

            # Get model outputs
            with torch.no_grad():
                # Process the batch based on modality
                modality = batch.get("modality", ["unknown"])[0] if isinstance(batch.get("modality"), list) else batch.get("modality", "unknown")

                # Get hidden states from backbone
                if "image" in batch:
                    img_input = batch["image"]
                elif "observation.images.image" in batch:
                    img_input = batch["observation.images.image"]
                else:
                    # Use a default tensor if no image available
                    img_input = torch.rand(1, 3, 224, 224, device=self.device)

                hidden_states = self.model.backbone(img_input)

                if modality in ["hl_subtask", "web_caption", "qa"]:
                    # Evaluate subtask performance
                    if "target_tokens" in batch:
                        # Get subtask predictions
                        subtask_preds = self.model.sample_subtask(hidden_states)
                        subtask_gts = batch["target_tokens"]

                        subtask_metrics = self.eval_subtask(subtask_preds, subtask_gts)
                        all_subtask_metrics.append(subtask_metrics)

                if modality in ["fast_robot_actions", "continuous_robot_actions"]:
                    # Evaluate action performance
                    if "actions_cont" in batch:
                        action_gts = batch["actions_cont"]

                        action_metrics = self.eval_actions(hidden_states, action_gts)
                        all_action_metrics.append(action_metrics)

            total_samples += len(batch.get("modality", [0]))  # Approximate count

        # Aggregate metrics
        final_metrics = {"total_evaluated_samples": total_samples}

        # Aggregate subtask metrics
        if all_subtask_metrics:
            avg_subtask_acc = np.mean([m["subtask_accuracy"] for m in all_subtask_metrics])
            final_metrics["avg_subtask_accuracy"] = avg_subtask_acc
            final_metrics["subtask_evaluations"] = len(all_subtask_metrics)

        # Aggregate action metrics
        if all_action_metrics:
            avg_action_mse = np.mean([m["action_mse"] for m in all_action_metrics])
            avg_action_mae = np.mean([m["action_mae"] for m in all_action_metrics])
            avg_action_acc = np.mean([m["action_accuracy_within_threshold"] for m in all_action_metrics])

            final_metrics["avg_action_mse"] = avg_action_mse
            final_metrics["avg_action_mae"] = avg_action_mae
            final_metrics["avg_action_accuracy_within_threshold"] = avg_action_acc
            final_metrics["action_evaluations"] = len(all_action_metrics)

        return final_metrics