import torch
import numpy as np
from textattack.models.wrappers import ModelWrapper
from model_wrapper import LlamaWrapper, QwenWrapper
from globalenv import QUICK_TOKEN_ID_DICT, INST_TEMPLATE, INST_SYS_ANS_PREFIX_alt1, Anth_YN_QUESTION_PROFIX


class PersonaModelWrapper(ModelWrapper):
    """
    Wrapper to make LayerNavigator models compatible with TextAttack.
    This wrapper handles the persona classification task (Yes/No answers).
    """
    
    def __init__(self, model_wrapper, task_name):
        """
        Args:
            model_wrapper: LlamaWrapper or QwenWrapper instance
            task_name: One of the Anth_MAIN tasks
            use_positive_prompt: Whether to use positive or negative persona prompt
        """
        self.model = model_wrapper
        self.task_name = task_name
        
        # Token IDs for Yes/No
        self.yes_token_id = QUICK_TOKEN_ID_DICT[" Yes"]
        self.no_token_id = QUICK_TOKEN_ID_DICT[" No"]
        
    def __call__(self, text_input_list):
        """
        Args:
            text_input_list: List of question strings
        Returns:
            Tensor of shape (batch_size, 2) with [No_prob, Yes_prob]
        """
        batch_size = len(text_input_list)
        outputs = []
        
        for text in text_input_list:
            # Format the input with the instruction template
            if text.endswith(Anth_YN_QUESTION_PROFIX):
                # Already has suffix, keep as is
                question_with_suffix = text
            else:
                # Add suffix
                question_with_suffix = text + Anth_YN_QUESTION_PROFIX
            
            # steering here
            # Format the input with the instruction template
            formatted_input = INST_TEMPLATE.format(
                question_with_suffix,  # Now includes suffix
                INST_SYS_ANS_PREFIX_alt1
            )
            
            # Tokenize
            input_ids = self.model.tokenizer(
                formatted_input,
                return_tensors="pt",
                padding=True
            ).to(self.model.device)
            
            # Get logits for the last token
            logits = self.model.get_last_logits(input_ids)
            
            # Extract Yes/No probabilities
            yes_logit = logits[self.yes_token_id].item()
            no_logit = logits[self.no_token_id].item()
            
            # Convert to probabilities
            probs = torch.softmax(
                torch.tensor([no_logit, yes_logit]), 
                dim=0
            )
            outputs.append(probs)
        
        # Stack into batch
        return torch.stack(outputs)
    
    def get_grad(self, text_input):
        """
        Get gradient of loss with respect to input tokens.
        Required for white-box attacks.
        """
        raise NotImplementedError(
            "Gradient-based attacks not yet supported for this wrapper"
        )
    
    def _tokenize(self, inputs):
        """Helper method for tokenization"""
        return [
            self.model.tokenizer.convert_ids_to_tokens(
                self.model.tokenizer(x, truncation=True)["input_ids"]
            )
            for x in inputs
        ]