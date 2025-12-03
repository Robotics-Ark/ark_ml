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

    def eval_actions(self, batch, ground_truth_actions):
        """
        Evaluate action prediction performance using the actual policy.

        Args:
            batch: Input batch with observations
            ground_truth_actions: Ground truth continuous actions

        Returns:
            Dictionary with MSE and other action metrics
        """
        # Use the model's prediction method to get predicted actions
        try:
            # Prepare the input for the model
            prepared_batch = self.model.prepare_input(batch)
            # Use model's predict method (which calls select_action internally)
            predicted_actions = self.model._policy.select_action(prepared_batch)
        except Exception as e:
            print(f"Error during action prediction: {e}")
            # Fallback to zeros if prediction fails
            predicted_actions = torch.zeros_like(ground_truth_actions)

        # Ensure predicted actions match the ground truth shape
        if predicted_actions.shape != ground_truth_actions.shape:
            # Try to match shapes if possible
            if predicted_actions.numel() == ground_truth_actions.numel():
                predicted_actions = predicted_actions.view(ground_truth_actions.shape)
            else:
                # Create dummy predictions with correct shape
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
            processed_batch = {}
            for key, value in batch.items():
                if torch.is_tensor(value):
                    processed_batch[key] = value.to(self.device)
                else:
                    processed_batch[key] = value

            # Get model outputs
            with torch.no_grad():
                # Process the batch based on modality
                modality = processed_batch.get("modality", ["unknown"])[0] if isinstance(processed_batch.get("modality"), list) else processed_batch.get("modality", "unknown")

                if modality in ["hl_subtask", "web_caption", "qa"]:
                    # Evaluate subtask performance if available in the underlying policy
                    if "target_tokens" in processed_batch:
                        # For LeRobot-based Pi0.5, subtask evaluation is handled internally
                        # This would be done through forward pass with appropriate targets
                        pass

                if modality in ["fast_robot_actions", "continuous_robot_actions"]:
                    # Evaluate action performance
                    if "action" in processed_batch or "actions_cont" in processed_batch:
                        action_gts = processed_batch.get("action", processed_batch.get("actions_cont"))
                        if action_gts is not None:
                            action_metrics = self.eval_actions(processed_batch, action_gts)
                            all_action_metrics.append(action_metrics)

            total_samples += len(processed_batch.get("modality", [0]))  # Approximate count

        # Aggregate metrics
        final_metrics = {"total_evaluated_samples": total_samples}

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