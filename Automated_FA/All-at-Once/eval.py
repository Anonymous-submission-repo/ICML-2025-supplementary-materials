import os
import json
from openai import AzureOpenAI

def analyze_chat_history_in_directory(directory_path,is_handcrafted= False):
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
        prompt = (
            "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem. "
            f"The problem is:  {problem} \n"
			f"The Answer for the problem is: {data.get('ground_truth', '')}\n" # If you want to evaluate without the ground truth, please remove this line.
            "Identify which agent made an error, at which step, and explain the reason for the error. "
            "Here's the conversation:\n\n" + chat_content +
            "\n\nBased on this conversation, please predict the following:\n"
            "1. The name of the agent who made a mistake that should be directly responsible for the wrong solution to the real world problem. If there are no agents that make obvious mistakes, decide one single agent in your mind. Directly output the name of the Expert.\n"
            "2. In which step the mistake agent first made mistake. For example, in a conversation structured as follows: "
            """
            {
                "agent a": "xx",
                "agent b": "xxxx",
                "agent c": "xxxxx",
                "agent a": "xxxxxxx"
            },
            """

            "each entry represents a 'step' where an agent provides input. The 'x' symbolizes the speech of each agent. If the mistake is in agent c's speech, the step number is 2. If the second speech by 'agent a' contains the mistake, the step number is 3, and so on. Please determine the step number where the first mistake occurred.\n"
            "3. The reason for your prediction."
            "Please answer in the format: Agent Name: (Your prediction)\n Step Number: (Your prediction)\n Reason for Mistake: \n"
        )

        client = AzureOpenAI(
            api_key="",   # put your API key here 
            api_version="2024-08-01-preview",
            azure_endpoint = "" # put your Azure endpoint here
        )
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
            max_tokens=1024
        )
        result = response.choices[0].message.content
        print(f"Prediction for {json_file}:")
        print(result)
        print("\n" + "="*50 + "\n")

# path_to_your_data
directory_path = 'path/'
# is_handcrafted
is_handcrafted = False

analyze_chat_history_in_directory(directory_path,is_handcrafted)
