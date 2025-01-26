import re
import json
import os

def read_predictions(eval_file, is_handcrafted=False):
    with open(eval_file, 'r') as file:
        data = file.read()
    predictions = {}
    if not is_handcrafted:
        pattern = r"Prediction for (\d+\.json):(.*?)(?=Prediction for|$)"
    else:
        pattern = r"Prediction for ([\d\+\-\.]+\.json):(.*?)(?=Prediction for|$)"

    blocks = re.finditer(pattern, data, re.DOTALL)
    
    for block in blocks:
        content = block.group(2)
        idx = block.group(1)
        agent_name_match = re.search(r"Agent Name:\s*([\w_]+)", content)
        step_number_match = re.search(r"Step Number:\s*(\d+)", content)
        if agent_name_match and step_number_match:
            agent_name = agent_name_match.group(1)
            step_number = step_number_match.group(1)
            predictions[idx] = {
                'predicted_agent': agent_name,
                'predicted_step': f"{step_number}"
            }
    print("================================")
    print(len(predictions))
    return predictions

def read_actual_data(labeled_json):
    with open(labeled_json, 'r') as file:
        data = json.load(file)
    return data['mistake_agent'], data['mistake_step']

def evaluate_accuracy(predictions,data_path):
    correct_agent = 0
    correct_step = 0
    total_files = len(predictions)
    
    for idx, pred in predictions.items():
        labeled_file = os.path.join(data_path,f"{idx}")
        print(f"Checking file: {idx}")
        if os.path.exists(labeled_file):
            actual_agent, actual_step = read_actual_data(labeled_file)
            print(f"Predicted agent: {pred['predicted_agent']}, Predicted step: {pred['predicted_step']}")
            print(f"Actual agent: {actual_agent}, Actual step: {actual_step}")
            print("================================")
            print("\n")
            if pred['predicted_agent'] == actual_agent:
                correct_agent += 1
            if pred['predicted_step'] == actual_step:
                correct_step += 1
    
    agent_accuracy = (correct_agent / total_files) * 100
    step_accuracy = (correct_step / total_files) * 100
    print(f"correct_agent: {correct_agent}, correct_step: {correct_step}, total_files: {total_files}")
    return agent_accuracy, step_accuracy

# data_path
data_path = 'path_to_your_data'
# specify whether it is handcrafted systems
is_handcrafted = "xx" # False or True

eval_file = 'eval.txt'
predictions = read_predictions(eval_file,is_handcrafted)
agent_accuracy, step_accuracy = evaluate_accuracy(predictions,data_path)

print(f"Agent Accuracy: {agent_accuracy}%")
print(f"Step Accuracy: {step_accuracy}%")
