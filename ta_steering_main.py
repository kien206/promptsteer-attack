"""
Run TextAttack with steering-based goal function
"""
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Set your GPU

import argparse
import json
import torch
from datetime import datetime
from tqdm import tqdm

from textattack import Attack
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.transformations import WordSwapEmbedding
from textattack.search_methods import GreedyWordSwapWIR

from model_wrapper import LlamaWrapper, QwenWrapper
from globalenv import MODEL, Anth_MAIN, LAYERS
from ta_dataset_wrapper import AnthropicPersonaDataset
from ta_goal_function import MinimizeSteeringImpact, MaximizeSteeringImpact
from strategy import UniStrategy


def load_steering_vectors(task, method="md", layers=None):
    """
    Load pre-computed steering vectors
    """
    if layers is None:
        layers = LAYERS[:5]  # Use top 5 layers by default
    
    vec_root = f"./Vectors/{task}/{method}"
    vectors = {}
    
    for layer in layers:
        vector_path = f"{vec_root}/L{layer}.pt"
        if os.path.exists(vector_path):
            vectors[layer] = torch.load(vector_path)
        else:
            raise FileNotFoundError(f"Vector not found: {vector_path}")
    
    return vectors


def run_steering_attack(
    task,
    goal_type="minimize_impact",  # "minimize_impact", "maximize_impact", "minimize_effectiveness"
    num_steering_layers=3,
    splits=["train", "val", "test"],
    input_dir="./Dataset/Anth_Persona_ALL",
    output_dir="./Dataset/Anth_Persona_ALL_Steering_Attacked",
    model_path=None,
    num_examples=None,
    checkpoint_interval=50,
):
    """
    Run TextAttack with steering-based goal function
    """
    
    print(f"\n{'='*70}")
    print(f"Running Steering-Based Attack")
    print(f"Task: {task}")
    print(f"Goal: {goal_type}")
    print(f"Steering Layers: {num_steering_layers}")
    print(f"{'='*70}\n")
    
    # Load model
    model_path = model_path or MODEL
    if "Llama" in model_path:
        base_model = LlamaWrapper(model_path)
    elif "Qwen" in model_path:
        base_model = QwenWrapper(model_path)
    else:
        raise ValueError(f"Unsupported model: {model_path}")
    
    # Load steering vectors and get best layers
    print("Loading steering vectors...")
    strategy = UniStrategy(
        task=task,
        strategy="my",
        num_layers=num_steering_layers,
        method="md",
    )
    steering_layers = strategy.layers
    steering_vectors = load_steering_vectors(task, method="md", layers=steering_layers)
    steering_alphas = [1.0] * num_steering_layers
    
    print(f"Using steering layers: {steering_layers}")
    print(f"Steering alphas: {steering_alphas}\n")
    
    # Create goal function
    if goal_type == "minimize_impact":
        goal_function = MinimizeSteeringImpact(
            model_wrapper=base_model,
            steering_layers=steering_layers,
            steering_vectors=steering_vectors,
            steering_alphas=steering_alphas,
            target_improvement_threshold=0.1,
        )
    elif goal_type == "maximize_impact":
        goal_function = MaximizeSteeringImpact(
            model_wrapper=base_model,
            steering_layers=steering_layers,
            steering_vectors=steering_vectors,
            steering_alphas=steering_alphas,
            target_improvement_threshold=0.3,
        )
    else:
        raise ValueError(f"Unknown goal type: {goal_type}")
    
    # Create transformation and constraints (similar to TextFooler)
    transformation = WordSwapEmbedding(max_candidates=50)
    
    stopwords = set(["a", "about", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also", "although", "am", "among", "amongst", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around", "as", "at", "be", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "both", "but", "by", "can", "cannot", "could", "down", "during", "each", "either", "else", "elsewhere", "enough", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "first", "for", "former", "formerly", "from", "had", "has", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "i", "if", "in", "indeed", "into", "is", "it", "its", "itself", "just", "last", "latter", "latterly", "least", "less", "like", "may", "me", "meanwhile", "might", "mine", "more", "moreover", "most", "mostly", "much", "must", "my", "myself", "namely", "neither", "never", "nevertheless", "next", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own", "per", "perhaps", "please", "same", "several", "she", "should", "since", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "this", "those", "though", "through", "throughout", "thru", "thus", "to", "together", "too", "toward", "towards", "under", "unless", "until", "up", "upon", "us", "used", "very", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves"])
    
    constraints = [
        RepeatModification(),
        StopwordModification(stopwords=stopwords),
        WordEmbeddingDistance(min_cos_sim=0.5),
        PartOfSpeech(allow_verb_noun_swap=True),
    ]
    
    # Search method
    search_method = GreedyWordSwapWIR(wir_method="delete")
    
    # Build attack
    attack = Attack(goal_function, constraints, transformation, search_method)
    
    # Create output directory
    attack_output_dir = os.path.join(output_dir, goal_type)
    os.makedirs(attack_output_dir, exist_ok=True)
    
    # Process each split
    all_stats = {}
    
    for split in splits:
        print(f"\n{'='*70}")
        print(f"Processing split: {split}")
        print(f"{'='*70}\n")
        
        # Load original dataset
        input_file = os.path.join(input_dir, f"{task}_{split}.jsonl")
        
        if not os.path.exists(input_file):
            print(f"Warning: File not found: {input_file}. Skipping...")
            continue
        
        with open(input_file, 'r') as f:
            original_data = [json.loads(line.strip()) for line in f]
        
        if num_examples is not None:
            original_data = original_data[:num_examples]
        
        print(f"Loaded {len(original_data)} examples from {input_file}")
        
        # Prepare output file
        output_file = os.path.join(attack_output_dir, f"{task}_{split}.jsonl")
        checkpoint_file = output_file + ".checkpoint"
        
        # Check for checkpoint
        start_idx = 0
        attacked_data = []
        
        if os.path.exists(checkpoint_file):
            print(f"Found checkpoint file. Resuming...")
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                start_idx = checkpoint_data['last_index'] + 1
                attacked_data = checkpoint_data['attacked_data']
            print(f"Resuming from index {start_idx}")
        
        # Statistics
        stats = {
            'total': len(original_data),
            'successful_attacks': 0,
            'failed_attacks': 0,
            'skipped': 0,
            'total_queries': 0,
            'avg_base_improvement': 0.0,
            'avg_attacked_improvement': 0.0,
        }
        
        base_improvements = []
        attacked_improvements = []
        
        # Attack each example
        print(f"Attacking examples {start_idx} to {len(original_data)}...")
        
        for idx in tqdm(range(start_idx, len(original_data)), desc=f"Attacking {split}"):
            original_item = original_data[idx]
            question = original_item["question"]
            label = 1 if original_item["answer_matching_behavior"] == " Yes" else 0
            
            try:
                # Run attack
                attack_result = attack.attack(question, label)
                
                # Create new item
                new_item = original_item.copy()
                stats['total_queries'] += getattr(attack_result, 'num_queries', 0)
                
                # Calculate steering improvements
                base_improvement = attack_result.original_result.score
                base_improvements.append(base_improvement)
                
                if hasattr(attack_result, 'perturbed_result'):
                    # Attack succeeded
                    perturbed_text = attack_result.perturbed_result.attacked_text.text
                    attacked_improvement = attack_result.perturbed_result.score
                    attacked_improvements.append(attacked_improvement)
                    
                    new_item["question"] = perturbed_text
                    new_item["perturbed"] = True
                    new_item["original_question"] = question
                    new_item["num_queries"] = attack_result.num_queries
                    new_item["base_steering_improvement"] = base_improvement
                    new_item["attacked_steering_improvement"] = attacked_improvement
                    new_item["steering_reduction"] = base_improvement - attacked_improvement
                    
                    stats['successful_attacks'] += 1
                else:
                    # Attack failed
                    new_item["perturbed"] = False
                    new_item["attack_status"] = str(type(attack_result).__name__)
                    new_item["base_steering_improvement"] = base_improvement
                    
                    stats['failed_attacks'] += 1
                
                attacked_data.append(new_item)
                
            except Exception as e:
                print(f"\nError attacking example {idx}: {e}")
                new_item = original_item.copy()
                new_item["perturbed"] = False
                new_item["error"] = str(e)
                attacked_data.append(new_item)
                stats['failed_attacks'] += 1
            
            # Save checkpoint
            if (idx + 1) % checkpoint_interval == 0:
                checkpoint_data = {
                    'last_index': idx,
                    'attacked_data': attacked_data,
                    'stats': stats
                }
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f)
                print(f"\nCheckpoint saved at index {idx}")
        
        # Save final dataset
        print(f"\nSaving attacked dataset to {output_file}...")
        with open(output_file, 'w') as f:
            for item in attacked_data:
                f.write(json.dumps(item) + '\n')
        
        # Remove checkpoint
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        
        # Calculate statistics
        stats['attack_success_rate'] = (
            stats['successful_attacks'] / stats['total']
            if stats['total'] > 0 else 0
        )
        stats['avg_queries'] = (
            stats['total_queries'] / stats['successful_attacks']
            if stats['successful_attacks'] > 0 else 0
        )
        stats['avg_base_improvement'] = (
            sum(base_improvements) / len(base_improvements)
            if base_improvements else 0
        )
        stats['avg_attacked_improvement'] = (
            sum(attacked_improvements) / len(attacked_improvements)
            if attacked_improvements else 0
        )
        stats['avg_steering_reduction'] = (
            stats['avg_base_improvement'] - stats['avg_attacked_improvement']
        )
        
        all_stats[split] = stats
        
        # Print statistics
        print(f"\n{'='*70}")
        print(f"Statistics for {split}:")
        print(f"  Total examples: {stats['total']}")
        print(f"  Successful attacks: {stats['successful_attacks']}")
        print(f"  Failed attacks: {stats['failed_attacks']}")
        print(f"  Attack success rate: {stats['attack_success_rate']:.2%}")
        print(f"  Avg queries per attack: {stats['avg_queries']:.1f}")
        print(f"  Avg base steering improvement: {stats['avg_base_improvement']:.4f}")
        print(f"  Avg attacked steering improvement: {stats['avg_attacked_improvement']:.4f}")
        print(f"  Avg steering reduction: {stats['avg_steering_reduction']:.4f}")
        print(f"{'='*70}\n")
    
    # Save statistics
    stats_file = os.path.join(attack_output_dir, f"{task}_stats.json")
    with open(stats_file, 'w') as f:
        json.dump({
            'task': task,
            'goal_type': goal_type,
            'steering_layers': steering_layers,
            'num_steering_layers': num_steering_layers,
            'model': model_path,
            'timestamp': datetime.now().isoformat(),
            'splits': all_stats
        }, f, indent=2)
    
    print(f"\nStatistics saved to: {stats_file}\n")
    
    return all_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run TextAttack with steering-based goal function"
    )
    parser.add_argument("--task", type=str, default="conscientiousness")
    parser.add_argument("--goal", type=str, default="minimize_impact",
                       choices=["minimize_impact", "maximize_impact"])
    parser.add_argument("--num-steering-layers", type=int, default=3)
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--model-path", type=int, default="/media/volume/DoubleSteeringLLM/promptsteer/.cache/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    
    args = parser.parse_args()
    if args.task == "all":
        for task in Anth_MAIN:
            run_steering_attack(
                task=task,
                goal_type=args.goal,
                num_steering_layers=args.num_steering_layers,
                num_examples=args.num_examples,
                model_path=args.model_path,
                checkpoint_interval=args.checkpoint_interval,
            )
    else:
        run_steering_attack(
            task=args.task,
            goal_type=args.goal,
            num_steering_layers=args.num_steering_layers,
            num_examples=args.num_examples,
            model_path=args.model_path,
            checkpoint_interval=args.checkpoint_interval,
        )