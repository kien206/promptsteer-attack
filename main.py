from model_wrapper import LlamaWrapper,QwenWrapper
from dataset import *
from get_results import *
from get_vec import *
from globalenv import *
from get_score import *
from strategy import UniStrategy

import json
import os

if __name__ == "__main__":
    ### Step 1.
    ### Complement the model path to the first line of the *globalenv.py* file.

    ### Step 2.
    ### Run the following code to get the main results in our paper.


    # Get Model
    if "Llama" in MODEL:
        model = LlamaWrapper(MODEL)
    elif "Qwen" in MODEL:
        model = QwenWrapper(MODEL)
    else:
        raise NotImplementedError("Model Not Implemented")

    outputs = {}
    for task in Anth_MAIN:
        print(f"############## Task: {task} ################")

        # Get Base Results
        test_dataset = UniDataset(
            task = task,
            train = False,
            set = "test",
        )

        # test_dataset = UniDatasetAttacked(
        #     attack_name="deepwordbug",
        #     task = task,
        #     train = False,
        #     set = "test",
        # )
        
        base_prob = get_raw_BASE_results(
            model=model,
            test_dataset=test_dataset,
        )
        print(f"Base Prob: {base_prob}")

        # Get Vectors
        train_dataset = UniDataset(
            task = task,
            train = True,
            set = "train",
        )
        uni_generate_vectors(
            method="md",
            model=model,
            layers=LAYERS,
            dataset=train_dataset,
        )

        # Get Score
        get_score(
            layers=LAYERS,
            dataset=train_dataset,
            vec_task=task,
            vec_method="md",
            acts_pre="standard",
        )

        # Get LayerNavigtor Results
        outputs[task] = []
        for num_layers in range(1, 8):
            strategy = UniStrategy(
                task=task,
                strategy="my",
                num_layers=num_layers,
                method="md",
            )
            test_prob = get_raw_results(
                model=model,
                layers=strategy.layers,
                test_dataset=test_dataset,
                Alphas=[1.0]*num_layers,
                train_task=task,
                train_method="md",
            )
            print(f"Layers: {num_layers}, Test Prob: {test_prob}")
            output = {"Layers": strategy.layers, "Score": test_prob, "Diff": test_prob - base_prob}
            outputs[task].append(output)

    output_path = f"./test_result/"
    os.makedirs(output_path, exist_ok=True)
    with open(f"{output_path}/attacked_dwb_test_results.json", 'w') as f:
        json.dump(outputs, f, indent=2)
    del test_dataset, train_dataset

