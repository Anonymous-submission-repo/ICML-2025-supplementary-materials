import os
import json
from openai import AzureOpenAI

def analyze_chat_history_in_directory(directory_path, is_handcrafted= False):
    json_files = sorted(
        [f for f in os.listdir(directory_path) if f.endswith('.json')],
        key=lambda x: int(''.join(filter(str.isdigit, x)))
    )
   
    index_agent = "role" if is_handcrafted else "name"
    
    for json_file in json_files:
        file_path = os.path.join(directory_path, json_file)

        with open(file_path, 'r') as f:
            data = json.load(f)

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        chat_content = "\n".join([
            f"{entry[index_agent]}: {entry['content']}" for entry in chat_history
        ])
        content = ""
        for idx, entry in enumerate(chat_history):
            content += f"Step {idx} - {entry['name']}: {entry['content']}\n"
            prompt = (
                f"You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multi-agent conversation aimed at solving a real-world problem. The problem being addressed is: {problem}. "
                f"The Answer for the problem is: {data.get('ground_truth', '')}\n" # If you want to evaluate without the ground truth, please remove this line.
                f"Here is the conversation history up to the current step: {content} "
                "Your task is to determine whether the most recent agent's action contains an error that could hinder the problem-solving process. "
                "Please respond with 'Yes' or 'No' and provide a clear explanation for your judgment. "
                "Note: Please avoid being overly critical in your evaluation. "
                "Respond in the format: 1. Yes/No. 2. Reason for the judgment."
            )


            client = AzureOpenAI(
                api_key="",  # put your API key here
                api_version="2024-08-01-preview",
                azure_endpoint = "" # put your endpoint here
            )
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
                max_tokens=1024
            )
            answer = response.choices[0].message.content.strip()
            print(answer)

            if "yes" in answer.lower():
                print(f"Prediction for {json_file}:")
                print(f"Agent Name: {entry['name']}")
                print(f"Step Number: {idx}")
                print("\n" + "="*50 + "\n")
                break
            elif "no" in answer.lower() and idx == len(chat_history) - 1:
                print(f"No mistakes found in file {json_file}")
                print("="*50)


# path_to_your_data
directory_path = 'path/'
# is_handcrafted
is_handcrafted = False

analyze_chat_history_in_directory(directory_path,is_handcrafted)
