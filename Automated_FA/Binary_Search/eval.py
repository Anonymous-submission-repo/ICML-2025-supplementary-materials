import os
import json
from openai import AzureOpenAI
import random

def analyze_chat_history_in_directory(directory_path,is_handcrafted):
    json_files = sorted(
        [f for f in os.listdir(directory_path) if f.endswith('.json')],
        key=lambda x: int(''.join(filter(str.isdigit, x)))
    )
    
    for json_file in json_files:
        file_path = os.path.join(directory_path, json_file)
        with open(file_path, 'r') as f:
            data = json.load(f)

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        answer = data.get("ground_truth", "")
        find_error_in_segment(chat_history, problem, answer, 0, len(chat_history) - 1, json_file,is_handcrafted)

def find_error_in_segment(chat_history, problem, answer, start, end, json_file, is_handcrafted):
    if start == end:
        report_error(chat_history, problem, start, json_file,is_handcrafted)
        return
    
    index_agent = "role" if is_handcrafted else "name"
    
    chat_content = "\n".join(
        f"{chat_history[i][index_agent]}: {chat_history[i]['content']}"
        for i in range(start, end + 1)
    )
    
    mid = (start + end) // 2
    upper_half =f"from step {start} to step {mid}."
    lower_half = f"from step {mid + 1} to step {end}."
    prompt = construct_prompt(problem, answer, chat_content, f"from step {start} to {end}", upper_half, lower_half)
    client = AzureOpenAI(
        api_key="",  # put your API key here
        api_version="2024-08-01-preview",
        azure_endpoint = "" # put your endpoint here
        )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1024
    )
    result = response.choices[0].message.content.strip().lower()
    print(result)


    if "upper half" in result:
        find_error_in_segment(chat_history, problem, answer, start, mid, json_file)
    elif "lower half" in result:
        find_error_in_segment(chat_history, problem, answer, mid + 1, end, json_file)
    else:
        if random.randint(0, 1) == 0:
            find_error_in_segment(chat_history, problem, answer, start, mid, json_file)
        else:
            find_error_in_segment(chat_history, problem, answer, mid + 1, end, json_file)

def construct_prompt(problem, answer, chat_content, range_description, upper_half, lower_half):
    return (
        "You are an AI assistant tasked with analyzing a segment of a multi-agent conversation. Multiple agents are collaborating to address a user query, with the goal of resolving the query through their collective dialogue.\n" 
        "Your primary task is to identify location of the most critical mistake, and determine the single step in the conversation where this error occurs, ultimately leading to the failure in resolving the user’s query."
        f"The problem to address is as follows: {problem}\n"
        f"The Answer for the problem is: {answer}\n" # If you want to evaluate without the ground truth, please remove this line.
        f"Review the following conversation range {range_description}:\n\n{chat_content}\n\n"
        "Based on your analysis, predict whether the error is more likely to be located in the upper or lower half of the segment."
        "lower half is defined as the range " + lower_half + " and upper half is defined as the range " + upper_half + "."
        "Please provide your prediction as either 'upper half' or 'lower half'. Remember, your answer should be based on identifying the mistake that directly contributes to the failure in resolving the user's query. If no clear error is evident, consider the most critical step you believe is responsible for the failure, allowing for subjective judgment, and base your answer on that." 
    )

def report_error(chat_history, problem, step, json_file,is_handcrafted):
    index_agent = "role" if is_handcrafted else "name"
    entry = chat_history[step]

    print(f"Prediction for {json_file}:\n")
    print(f"Agent Name: {entry[index_agent]}")
    print(f"Step Number: {step}")
    print("\n" + "="*50 + "\n")



# path_to_your_data
directory_path = 'path/'
# is_handcrafted
is_handcrafted = False

analyze_chat_history_in_directory(directory_path,is_handcrafted)
