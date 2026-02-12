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
from model_wrapper import LlamaWrapper, QwenWrapper
from globalenv import MODEL, Anth_MAIN, LAYERS
from strategy import UniStrategy
from textattack.goal_function_results import GoalFunctionResultStatus
from ta_goal_function.minimize_steering import MinimizeSteeringImpact
from ta_custom_recipe import TextFoolerWithCustomGoal, DeepWordBugWithCustomGoal

task2goal_map = {
    'textfooler': TextFoolerWithCustomGoal,
    'deepwordbug': DeepWordBugWithCustomGoal
}

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
    goal_type="minimize_impact",
    num_steering_layers=3,
    threshold=0.05,
    splits=["test"],
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
    
    goal_function = MinimizeSteeringImpact(
        model_wrapper=base_model,
        steering_layers=steering_layers,
        steering_vectors=steering_vectors,
        steering_alphas=steering_alphas,
        target_improvement_threshold=threshold,
        maximizable=False,
        use_cache=False,
        query_budget=float("inf"),
        model_batch_size=32,
    )
    # Build attack
    
    try:
        attack = task2goal_map[goal_type].build(goal_function)
    except KeyError:
        raise ValueError(f"Unsupported goal type: {goal_type}")
    
    # Create output directory
    attack_output_dir = os.path.join(output_dir, goal_type)
    os.makedirs(attack_output_dir, exist_ok=True)
    
    # Process each split
    all_stats = {}
    
    for split in splits:
        print(f"\n{'='*70}")
        print(f"Processing split: {split}")
        print(f"{'='*70}\n")
        
        goal_function.reset_statistics()
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
            'attack_avg_no_steer': 0,
            'attack_avg_with_steer': 0,
            'avg_attacked_improvement': 0
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
                # # Calculate base steering improvement
                base_no_steering = goal_function._get_probability(
                    question, label, with_steering=False
                )[label].item()
                
                base_with_steering = goal_function._get_probability(
                    question, label, with_steering=True
                )[label].item()
                
                base_improvement = base_with_steering - base_no_steering
                
                # Run attack
                attack_result = attack.attack(question, label)
                # base_no_steering = attack_result.original_result.attacked_text.attack_attrs.get("prob_no_steering")
                # base_with_steering = attack_result.original_result.attacked_text.attack_attrs.get("prob_with_steering")
                # base_improvement = attack_result.original_result.attacked_text.attack_attrs.get("steering_improvement")
                print("====BASELINE: ", base_no_steering, base_with_steering)

                # Create new item
                new_item = original_item.copy()
                stats['total_queries'] += getattr(attack_result, 'num_queries', 0)
                
                # Check the actual goal status
                if hasattr(attack_result, 'perturbed_result'):
                    perturbed_result = attack_result.perturbed_result
                    perturbed_text = perturbed_result.attacked_text.text
                    
                    # Get attacked steering improvement
                    attacked_improvement = perturbed_result.attacked_text.attack_attrs.get(
                        "steering_improvement", 
                        None
                    )
                    
                    # If not stored, calculate it
                    if attacked_improvement is None:
                        attacked_no_steering = goal_function._get_probability(
                            perturbed_text, label, with_steering=False
                        )[label].item()
                        
                        attacked_with_steering = goal_function._get_probability(
                            perturbed_text, label, with_steering=True
                        )[label].item()
                        
                        attacked_improvement = attacked_with_steering - attacked_no_steering

                    stats['attack_avg_no_steer'] += perturbed_result.attacked_text.attack_attrs["prob_no_steering"] / len(original_data)
                    stats['attack_avg_with_steer'] += perturbed_result.attacked_text.attack_attrs["prob_with_steering"] / len(original_data)
                    stats['avg_attacked_improvement'] += attacked_improvement / len(original_data)
                    
                    # Check if goal was ACTUALLY achieved
                    print("====ATTACKED: ", perturbed_result.attacked_text.attack_attrs.get("prob_no_steering", None),  perturbed_result.attacked_text.attack_attrs.get("prob_with_steering", None))
                    goal_achieved = perturbed_result.goal_status
                    # print(perturbed_result.goal_status)
                    new_item["question"] = perturbed_text
                    new_item["perturbed"] = True
                    new_item["original_question"] = question
                    new_item["num_queries"] = attack_result.num_queries
                    new_item["goal_achieved"] = goal_achieved  # NEW: Track if goal was met
                    
                    # Store metrics
                    new_item["base_steering_improvement"] = base_improvement
                    new_item["attacked_steering_improvement"] = attacked_improvement
                    new_item["steering_reduction"] = base_improvement - attacked_improvement
                    
                    # Update stats based on ACTUAL goal achievement
                    if goal_achieved == GoalFunctionResultStatus.SUCCEEDED:
                        stats['successful_attacks'] += 1
                        # base_improvements.append(base_improvement)
                    elif goal_achieved == GoalFunctionResultStatus.SKIPPED:
                        stats['skipped'] += 1
                    else:
                        stats['failed_attacks'] += 1
                    base_improvements.append(base_improvement)
                    attacked_improvements.append(attacked_improvement)
                else:
                    # No perturbation found (skipped or other reason)
                    new_item["perturbed"] = False
                    new_item["attack_status"] = str(type(attack_result).__name__)
                    new_item["base_steering_improvement"] = base_improvement
                    new_item["goal_achieved"] = False
                    
                    if "Skipped" in str(type(attack_result).__name__):
                        stats['skipped'] += 1
                    else:
                        stats['failed_attacks'] += 1
                
                attacked_data.append(new_item)
            
            except Exception as e:
                print(f"\nError attacking example {idx}: {e}")
                import traceback
                traceback.print_exc()
                
                new_item = original_item.copy()
                new_item["perturbed"] = False
                new_item["error"] = str(e)
                new_item["goal_achieved"] = False
                attacked_data.append(new_item)
                stats['failed_attacks'] += 1
        
            # Save checkpoint periodically
            if (idx + 1) % checkpoint_interval == 0:
                checkpoint_data = {
                    'last_index': idx,
                    'attacked_data': attacked_data,
                    'stats': stats
                }
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f)
                print(f"\nCheckpoint saved at index {idx}")
                print(f"  Successful: {stats['successful_attacks']}")
                print(f"  Failed: {stats['failed_attacks']}")
                print(f"  Skipped: {stats['skipped']}")
        
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
        
        all_stats[split] = stats
        

    stats_summary = goal_function.get_statistics()

    all_stats[split]['base_probability'] = stats_summary
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


# def main():
#     parser = argparse.ArgumentParser(
#         description="Run TextAttack on Anthropic Persona dataset and save perturbed versions"
#     )
#     parser.add_argument(
#         "--task",
#         type=str,
#         choices=Anth_MAIN + ["all"],
#         default="conscientiousness",
#         help="Task to attack (or 'all' for all tasks)"
#     )
#     parser.add_argument(
#         "--attack",
#         type=str,
#         choices=list(ATTACK_RECIPES.keys()),
#         default="textfooler",
#         help="Attack recipe to use"
#     )
#     parser.add_argument(
#         "--splits",
#         type=str,
#         nargs="+",
#         default=["train", "val", "test"],
#         help="Dataset splits to attack (e.g., train val test)"
#     )
#     parser.add_argument(
#         "--input-dir",
#         type=str,
#         default="./Dataset/Anth_Persona_ALL",
#         help="Directory containing original dataset"
#     )
#     parser.add_argument(
#         "--output-dir",
#         type=str,
#         default="./Dataset/Anth_Persona_ALL_Attacked",
#         help="Directory to save attacked dataset"
#     )
#     parser.add_argument(
#         "--model",
#         type=str,
#         default=None,
#         help="Model path (uses globalenv.MODEL if not specified)"
#     )
#     parser.add_argument(
#         "--num-examples",
#         type=int,
#         default=None,
#         help="Number of examples to attack per split (default: all)"
#     )
#     parser.add_argument(
#         "--checkpoint-interval",
#         type=int,
#         default=50,
#         help="Save checkpoint every N examples"
#     )
    
#     args = parser.parse_args()
    
#     # Run attack(s)
#     if args.task == "all":
#         print(f"\n{'='*70}")
#         print(f"Running {args.attack} attack on ALL tasks")
#         print(f"{'='*70}\n")
        
#         all_results = {}
#         for task in Anth_MAIN:
#             try:
#                 stats = run_attack(
#                     task=task,
#                     attack_name=args.attack,
#                     splits=args.splits,
#                     input_dir=args.input_dir,
#                     output_dir=args.output_dir,
#                     model_path=args.model,
#                     num_examples=args.num_examples,
#                     checkpoint_interval=args.checkpoint_interval,
#                 )
#                 all_results[task] = stats
#             except Exception as e:
#                 print(f"\nError processing task {task}: {e}")
#                 continue
        
#         # Print overall summary
#         print("\n" + "="*70)
#         print("OVERALL SUMMARY - ALL TASKS")
#         print("="*70)
#         for task, splits_stats in all_results.items():
#             print(f"\n{task}:")
#             for split, stats in splits_stats.items():
#                 print(f"  {split}: {stats['attack_success_rate']:.2%} success rate "
#                       f"({stats['successful_attacks']}/{stats['total']} examples)")
#         print("="*70 + "\n")
        
#         # Save combined summary
#         summary_file = os.path.join(
#             args.output_dir, 
#             args.attack, 
#             f"all_tasks_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
#         )
#         with open(summary_file, 'w') as f:
#             json.dump(all_results, f, indent=2)
#         print(f"Combined summary saved to: {summary_file}\n")
        
#     else:
#         run_attack(
#             task=args.task,
#             attack_name=args.attack,
#             splits=args.splits,
#             input_dir=args.input_dir,
#             output_dir=args.output_dir,
#             model_path=args.model,
#             num_examples=args.num_examples,
#             checkpoint_interval=args.checkpoint_interval,
#         )

if __name__ == "__main__":
    input_dir = "./Dataset/Anth_Persona_ALL"
    output_dir = "./Dataset/Anth_Persona_ALL_Steering_Attacked"
    attack_name = "deepwordbug"
    all_results = {}

    
    for task in Anth_MAIN:
        print(f"\n{'-'*70}")
        print(f"{task}")
        print(f"{'-'*70}\n")
        try:
            run_steering_attack(
                task=task,
                goal_type=attack_name,
                num_steering_layers=1,
                threshold=0.3,
                num_examples=None,
                output_dir=output_dir,
                input_dir=input_dir,
                model_path="/media/volume/DoubleSteeringLLM/promptsteer/.cache/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
                checkpoint_interval=50,
            )
        except Exception as e:
            print(f"\nError processing task {task}: {e}")
            continue
