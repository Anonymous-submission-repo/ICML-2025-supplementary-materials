import pandas as pd
from datasets import load_dataset
from autogen.agentchat.contrib.captainagent import CaptainAgent, CaptainUserProxyAgent
from autogen import UserProxyAgent, config_list_from_json
import autogen
import sys
import json
import re
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'


def read_line_after_marker(filename, marker):
    with open(filename, 'r') as file:
        lines = file.readlines() 

    content_after_marker = None
    for i in range(len(lines) - 1): 
        if marker in lines[i]:
            content_after_marker = lines[i + 1].strip() 
            break

    return content_after_marker

def extract_dict_from_file(filename):
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
        
        index_of_dollars = -1
        for i, line in enumerate(lines):
            if "$$$$$" in line:
                index_of_dollars = i + 1
                break
        
        if index_of_dollars != -1 and index_of_dollars < len(lines):
            data_line = lines[index_of_dollars].strip()
            data_dict = eval(data_line) 
            return data_dict
        else:
            return "No matching data found."
    
    except FileNotFoundError:
        return "The file was not found."
    except Exception as e:
        return f"An error occurred: {str(e)}"


class CustomCaptainAgent(CaptainAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.modify_model_config()

    def modify_model_config(self):
        self.DEFAULT_NESTED_CONFIG['autobuild_init_config']['builder_model'] = 'gpt-4o'
        self.DEFAULT_NESTED_CONFIG['autobuild_init_config']['agent_model'] = 'gpt-4o'
        self.DEFAULT_NESTED_CONFIG['max_turns'] = 15

############################################################################################################
# specify the dataset name
data_name = "xxxxx" # assistant_bench or GAIA 

if data_name == "assistant_bench":
    ds = load_dataset("AssistantBench/AssistantBench")
elif data_name == "GAIA":
    ds = load_dataset("gaia-benchmark/GAIA", "2023_all")
validation_data = ds['validation']

llm_config = {
    "temperature": 0.3,
    "config_list": config_list_from_json("OAI_CONFIG_LIST"),
}

correct = 0
wrong_list = []
correct_list = []

data_length = 33 if data_name == "assistant_bench" else 164

for i in range(0, data_length):
    if data_name == "assistant_bench":
        task_id = validation_data[i]['id']
    else:
        task_id = validation_data[i]['task_id']
    if task_id :
        file = open(f'results_txt/{i}.txt', 'w')
        sys.stdout = file
        print(f'Processing Task ID: {task_id}')
        captain_agent = CustomCaptainAgent(
            name="captain_agent",
            llm_config=llm_config,
            agent_lib="captainagent_expert_library.json",
            tool_lib="default",
            code_execution_config={"use_docker": False, "work_dir": "groupchat"}

        )
        user_proxy = UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER"
        )
        if data_name == "assistant_bench":
            query = validation_data[i]['task'] + " Please solve the task carefully."
            ANSWER = validation_data[i]['answer']
        else:
            query = validation_data[i]['Question'] + " Please solve the problem carefully."
            if validation_data[i]['file_name']:
                query = query + "The attached file path is: " + f'../2023/validation/{validation_data[i]["file_name"]}.'
        
            ANSWER = validation_data[i]['Final answer']
            
        history = user_proxy.initiate_chat(captain_agent, message=query)
        
        result = user_proxy._oai_messages[captain_agent][-2]
        result = result['content']
        chat_history = captain_agent.chat_history
        cap_history = captain_agent.assistant.chat_messages[captain_agent.executor]
        
        raw_history = chat_history.copy()
        raw_history.insert(0, cap_history)

        check_sys_msg = """You are a helpful AI assistant. You will use your coding and language skills to verify the answer.
        You are given:
        1. A problem.
        2. A reply with the answer to the problem.
        3. A ground truth answer.
        Please do the following:
        1. Extract the answer in the reply: "The answer is <answer extracted>".
        2. Check whether the answer in the reply matches the ground truth answer. When comparison is not obvious (for example, 3*\\sqrt(6) and 7.348), you may write code to check the answer and wait for the user to execute the code.
        3. After everything is done, please choose a reply from the following options:
        - "The answer is correct."
        - "The answer is approximated but should be correct. Correct Answer: <ground truth answer> | Answer extracted: <answer extracted>."
        - "The answer is incorrect. Correct Answer: <ground truth answer> | Answer extracted: <answer extracted>."
        - "The reply doesn't contain an answer." """

        answer_checker = autogen.AssistantAgent(
            name="checker",
            llm_config=llm_config.copy(),
            system_message=check_sys_msg)

        checker_proxy = autogen.UserProxyAgent(
            "checker_proxy",
            human_input_mode="NEVER",
            code_execution_config={
            "work_dir": "coding",
            "use_docker": False,
        },
        max_consecutive_auto_reply=5,
        default_auto_reply="TERMINATE",
        is_termination_msg=lambda x: x.get("content", "").lower()
        and (
        "the answer is correct" in x.get("content", "").lower()
        or "the answer is incorrect" in x.get("content", "").lower()
        or "the reply doesn't contain an answer" in x.get("content", "").lower()
        or "the answer is approximated but should be correct" in x.get("content", "").lower()
        ),
    )
        is_correct = False
        message_to_check = "[Problem]: " + query + f"\n[Reply]: {result}\n\n[Ground truth answer]: " + ANSWER
        checker_proxy.initiate_chat(answer_checker, message=message_to_check)
        answer = checker_proxy._oai_messages[answer_checker][-1]['content']
        print(answer)
        print("++++++++++")
        if "The answer is correct." in answer or "The answer is approximated but should be correct" in answer or "the answer is correct." in answer or "the answer is approximated but should be correct" in answer:
            is_correct = True
            correct += 1
            correct_list.append(i)
        else:
            wrong_list.append(i)	

     
        filename = f'results_txt/{i}.txt' 
        marker = '$$$$$$$$$$$$$$$$$'
        system_prompt = extract_dict_from_file(filename)
   
        chat_history_new = []
        for index, history in enumerate(chat_history):
            history_new = {}
            history_new["history"] = history
            history_new["is_correct"] = False if index != len(chat_history) - 1 else is_correct
            history_new["system_prompt"] = system_prompt
        
            chat_history_new.append(history_new)
    
        results_json = []
        results_json.append({
            "is_correct": is_correct,
            "question": validation_data[i]['task'] if data_name == "assistant_bench" else validation_data[i]['Question'],
            "question_ID": task_id,
            "level": validation_data[i]['difficulty'] if data_name == "assistant_bench" else validation_data[i]['Level'],
            "ground_truth": ANSWER,
            "proceed_history": chat_history_new,
            "raw_history": raw_history,
            })
     
    
        with open(f'results/{i}.json', 'w') as json_file:
            json.dump(results_json, json_file, indent=4)


sys.stdout = sys.__stdout__

print(f"Number of correct answers: {correct}")
print("List of wrong questions:", wrong_list)
print("List of correct questions:", correct_list)
