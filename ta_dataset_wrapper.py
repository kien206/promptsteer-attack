import json
from textattack.datasets import Dataset as TextAttackDataset
from globalenv import Anth_YN_QUESTION_PROFIX

class AnthropicPersonaDataset:
    """
    Loads Anthropic Persona dataset and converts to TextAttack format
    """
    
    def __init__(self, task, split="test", data_path="./Dataset/Anth_Persona_ALL"):
        """
        Args:
            task: Task name (e.g., 'conscientiousness')
            split: 'train', 'val', or 'test'
            data_path: Path to dataset directory
        """
        self.task = task
        self.split = split
        self.data_path = f"{data_path}/{task}_{split}.jsonl"
        
        # Load data
        self.raw_data = self._load_data()
        
    def _load_data(self):
        """Load JSONL data"""
        with open(self.data_path, 'r') as f:
            return [json.loads(line.strip()) for line in f]
    
    def to_textattack_dataset(self, add_suffix=True):
        """
        Convert to TextAttack Dataset format
        
        Returns:
            TextAttackDataset with (text, label) pairs
            where label is 1 if answer should be "Yes", 0 for "No"
        """
        dataset = []
        
        for item in self.raw_data:
            # Extract question
            question = item["question"]
            if add_suffix:
                question += Anth_YN_QUESTION_PROFIX
            
            # Determine label (1 for Yes, 0 for No)
            label = 1 if item["answer_matching_behavior"] == " Yes" else 0
            
            dataset.append((question, label))
        
        return TextAttackDataset(
            dataset,
            input_columns=["text"],
            label_names=["No", "Yes"]
        )
    
    def get_subset(self, num_samples=None):
        """Get a subset of the dataset"""
        if num_samples is None:
            return self.to_textattack_dataset()
        
        subset_data = self.raw_data[:num_samples]
        original_data = self.raw_data
        self.raw_data = subset_data
        dataset = self.to_textattack_dataset()
        self.raw_data = original_data
        return dataset