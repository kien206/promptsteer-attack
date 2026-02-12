
import torch
import numpy as np
from textattack.goal_functions.classification import ClassificationGoalFunction
from textattack.goal_function_results import ClassificationGoalFunctionResult
from textattack.shared import AttackedText
from globalenv import QUICK_TOKEN_ID_DICT, INST_TEMPLATE, INST_SYS_ANS_PREFIX_alt1, Anth_YN_QUESTION_PROFIX

"""
Custom TextAttack goal function to generate examples resistant to steering
"""
import torch
import numpy as np
from textattack.goal_functions.classification import ClassificationGoalFunction
from textattack.goal_function_results import ClassificationGoalFunctionResult
from textattack.shared import AttackedText
from globalenv import QUICK_TOKEN_ID_DICT, INST_TEMPLATE, INST_SYS_ANS_PREFIX_alt1, Anth_YN_QUESTION_PROFIX


class MinimizeSteeringImpact(ClassificationGoalFunction):
    """
    Goal: Generate examples where steering FAILS.
    Success: When prob_with_steering < threshold (steered model performs poorly)
    """
    
    def __init__(
        self,
        model_wrapper,
        steering_layers,
        steering_vectors,
        steering_alphas,
        target_improvement_threshold=0.3,
        maximizable=False,
        use_cache=False,
        query_budget=float("inf"),
        model_batch_size=32,
    ):
        """
        Args:
            model_wrapper: Your model wrapper (LlamaWrapper/QwenWrapper)
            steering_layers: List of layer indices to apply steering
            steering_vectors: Dict of {layer: vector} for steering
            steering_alphas: List of alpha values for each layer
            target_improvement_threshold: Success if prob_with_steering < this (default: 0.3 = 30%)
        """
        super().__init__(
            model_wrapper,
            maximizable=maximizable,
            use_cache=use_cache,
            query_budget=query_budget,
            model_batch_size=model_batch_size,
        )
        
        self.steering_layers = steering_layers
        self.steering_alphas = steering_alphas
        self.target_improvement_threshold = target_improvement_threshold
        
        # Move steering vectors to the same device as the model
        self.steering_vectors = {}
        for layer, vector in steering_vectors.items():
            self.steering_vectors[layer] = vector.to(self.model.device)
        
        self.yes_token_id = QUICK_TOKEN_ID_DICT[" Yes"]
        self.no_token_id = QUICK_TOKEN_ID_DICT[" No"]
        
        # Initialize original baseline (will be set in init_attack_example)
        self.original_prob_no_steering = None
    
    def init_attack_example(self, attacked_text, ground_truth_output):
        """Called before attacking to store original baseline"""
        result, search_over = super().init_attack_example(attacked_text, ground_truth_output)
        
        original_prob_no_steering = attacked_text.attack_attrs.get("prob_no_steering")
        
        if self.original_prob_no_steering is None:
            text = attacked_text.text
            prob_no_steering_dist = self._get_probability(text, ground_truth_output, with_steering=False)
            original_prob_no_steering = prob_no_steering_dist[ground_truth_output].item()

        attacked_text.attack_attrs["original_prob_no_steering"] = original_prob_no_steering
    
        # Also store in self for convenience
        self.original_prob_no_steering = original_prob_no_steering
        # Silently accumulate scores
        prob_no_steering = attacked_text.attack_attrs.get("prob_no_steering")
        prob_with_steering = attacked_text.attack_attrs.get("prob_with_steering")
        
        if prob_no_steering is not None and prob_with_steering is not None:
            self.total_score_no_steering += prob_no_steering
            self.total_score_with_steering += prob_with_steering
            self.num_examples += 1
        
        return result, search_over
    
    def get_statistics(self):
        """
        Get accumulated statistics for sanity check
        
        Returns:
            dict with average scores
        """
        if self.num_examples == 0:
            return None
        
        avg_no_steering = self.total_score_no_steering / self.num_examples
        avg_with_steering = self.total_score_with_steering / self.num_examples
        attack_no_steering = self.attack_no_steering / self.num_examples
        attack_with_steering = self.attack_with_steering / self.num_examples

        return {
            'num_examples': self.num_examples,
            'avg_prob_no_steering': avg_no_steering,
            'avg_prob_with_steering': avg_with_steering,
            'avg_steering_improvement': avg_with_steering - avg_no_steering,
        }
    
    def reset_statistics(self):
        """Reset the accumulated statistics"""
        self.total_score_no_steering = 0.0
        self.total_score_with_steering = 0.0
        self.attack_no_steering = 0.0
        self.attack_with_steering = 0.0
        self.num_examples = 0
        
    def _is_goal_complete(self, model_output, attacked_text):
        """
        Success if:
        1. Steered model gives LOW probability to correct answer (steering fails)
        2. Non-steered model still performs reasonably (question is valid)
        
        Args:
            model_output: Tensor [P(No), P(Yes)] WITH steering applied
            attacked_text: The perturbed text
        
        Returns:
            True if steering fails AND baseline is maintained
        """
        # perturbed_prob_with_steering = attacked_text.attack_attrs.get("prob_with_steering")
        # perturbed_prob_no_steering = attacked_text.attack_attrs.get("prob_no_steering")
        
        # # Handle None cases
        # if perturbed_prob_with_steering is None or perturbed_prob_no_steering is None:
        #     return False
        
        # # Condition 1: Steering makes model perform poorly on PERTURBED text
        # steering_fails = perturbed_prob_with_steering < self.target_improvement_threshold
        
        # # Condition 2: Perturbed baseline is still reasonable (question is answerable)
        # baseline_maintained = perturbed_prob_no_steering >= self.baseline_tolerance
        
        # # Optional Condition 3: Perturbed baseline hasn't changed too much from original
        # if self.original_prob_no_steering is not None:
        #     baseline_change = abs(self.original_prob_no_steering - perturbed_prob_no_steering)
        #     baseline_stable = baseline_change < 0.1  # Allow up to 20% change
        # else:
        #     baseline_stable = True  # Skip this check if original not available
        
        # # All conditions must be met
        # is_complete = steering_fails and baseline_maintained and baseline_stable
        
        # return is_complete
        score = -self._get_score(model_output, attacked_text)
        
        is_complete = score < self.target_improvement_threshold
        
        return is_complete
    
    def _get_score(self, model_output, attacked_text):
        """
        Score = prob_with_steering (lower is better)
        
        We want to minimize the steered model's probability on the correct answer.
        Penalize if baseline drops too much (question becomes invalid).
        """
        prob_with_steering = attacked_text.attack_attrs.get("prob_with_steering")
        prob_no_steering = attacked_text.attack_attrs.get("prob_no_steering")
        # original_prob_no_steering = attacked_text.attack_attrs.get("original_prob_no_steering")
        # # Primary objective: minimize steered probability
        # # steering_score = prob_with_steering
        
        # if original_prob_no_steering is not None:
        #     baseline_diff = abs(original_prob_no_steering - prob_no_steering)
        # else:
        #     baseline_diff = 0.0
        
        # return -abs(prob_with_steering - prob_no_steering)
        return -prob_with_steering
    
    def _get_probability(self, text, ground_truth_output, with_steering=False):
        """
        Get the probability distribution over classes
        
        Args:
            text: Question text
            ground_truth_output: 1 for Yes, 0 for No
            with_steering: Whether to apply steering vectors
        
        Returns:
            Tensor of shape (2,) with [P(No), P(Yes)]
        """
        # Ensure suffix is present
        if not text.endswith(Anth_YN_QUESTION_PROFIX):
            question_with_suffix = text + Anth_YN_QUESTION_PROFIX
        else:
            question_with_suffix = text
        
        # Format input
        formatted_input = INST_TEMPLATE.format(
            question_with_suffix,
            INST_SYS_ANS_PREFIX_alt1
        )
        
        # Tokenize
        input_ids = self.model.tokenizer(
            formatted_input,
            return_tensors="pt",
            padding=True
        ).to(self.model.device)
        
        # Reset model
        self.model.reset_all()
        
        # Apply steering if requested
        if with_steering:
            for i, layer in enumerate(self.steering_layers):
                vector = self.steering_vectors[layer] * self.steering_alphas[i]
                vector = vector.to(self.model.device)
                self.model.set_add_activations(layer, vector)
        
        # Get logits
        logits = self.model.get_last_logits(input_ids)
        
        # Extract Yes/No logits
        yes_logit = logits[self.yes_token_id].item()
        no_logit = logits[self.no_token_id].item()
        
        # Calculate probability distribution
        probs = torch.softmax(torch.tensor([no_logit, yes_logit]), dim=0)
        
        return probs
    
    def _call_model_uncached(self, attacked_text_list):
        """
        Evaluate the model on a list of attacked texts.
        Calculate BOTH steered and non-steered probabilities,
        and store them in attack_attrs BEFORE returning.
        
        Returns:
            Tensor of shape (batch_size, 2) with probabilities (WITH steering)
        """
        outputs = []
        
        for attacked_text in attacked_text_list:
            text = attacked_text.text
            ground_truth = self.ground_truth_output
            
            # Calculate probability WITHOUT steering
            prob_no_steering_dist = self._get_probability(text, ground_truth, with_steering=False)
            prob_no_steering = prob_no_steering_dist[ground_truth].item()
            
            # Calculate probability WITH steering
            prob_with_steering_dist = self._get_probability(text, ground_truth, with_steering=True)
            prob_with_steering = prob_with_steering_dist[ground_truth].item()
            
            # Calculate steering improvement (for analysis)
            steering_improvement = prob_with_steering - prob_no_steering
            
            # Store in attack_attrs BEFORE processing
            attacked_text.attack_attrs["steering_improvement"] = steering_improvement
            attacked_text.attack_attrs["prob_no_steering"] = prob_no_steering
            attacked_text.attack_attrs["prob_with_steering"] = prob_with_steering
            
            self.attack_no_steering += prob_no_steering
            self.attack_with_steering += prob_with_steering
            # Return the steered probabilities for classification
            outputs.append(prob_with_steering_dist)
        
        # Stack into batch tensor
        batch_output = torch.stack(outputs)
        
        # Process through parent's method to ensure proper format
        return self._process_model_outputs(attacked_text_list, batch_output)
    
    def extra_repr_keys(self):
        """Additional keys for representation."""
        keys = super().extra_repr_keys()
        keys.extend(["steering_layers", "target_improvement_threshold", "baseline_tolerance"])
        return keys