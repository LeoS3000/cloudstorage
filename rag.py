# Universal RAG Experiment Script
import json, os, sys
import psutil
import torch
from typing import Any, Generator, List, Optional
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer,GenerationConfig
from llama_index.schema import Document

GPU = True
#GPU = False 

STAGING = True

if GPU:
    torch.set_default_dtype(torch.float16)
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def save_list_to_json(lst, filename):
  """ Save Files """
  with open(filename, 'w') as file:
    json.dump(lst, file)

def wr_dict(filename,dic):
  """ Write Files """
  try:
    if not os.path.isfile(filename):
      data = []
      data.append(dic)
      with open(filename, 'w') as f:
        json.dump(data, f)
    else:      
      with open(filename, 'r') as f:
        data = json.load(f)
        data.append(dic)
      with open(filename, 'w') as f:
          json.dump(data, f)
  except Exception as e:
    print("Save Error:", str(e))
  return

def rm_file(file_path):
  """ Delete Files """
  if os.path.exists(file_path):
    os.remove(file_path)
    print(f"File {file_path} removed successfully.")

def _depth_first_yield(json_data: Any, levels_back: int, collapse_length: 
                       Optional[int], path: List[str], ensure_ascii: bool = False,
                      ) -> Generator[str, None, None]:
  """ Do depth first yield of all of the leaf nodes of a JSON.
      Combines keys in the JSON tree using spaces.
      If levels_back is set to 0, prints all levels.
      If collapse_length is not None and the json_data is <= that number
      of characters, then we collapse it into one line.
  """
  if isinstance(json_data, (dict, list)):
    # only try to collapse if we're not at a leaf node
    json_str = json.dumps(json_data, ensure_ascii=ensure_ascii)
    if collapse_length is not None and len(json_str) <= collapse_length:
      new_path = path[-levels_back:]
      new_path.append(json_str)
      yield " ".join(new_path)
      return
    elif isinstance(json_data, dict):
      for key, value in json_data.items():
        new_path = path[:]
        new_path.append(key)
        yield from _depth_first_yield(value, levels_back, collapse_length, new_path)
    elif isinstance(json_data, list):
      for _, value in enumerate(json_data):
        yield from _depth_first_yield(value, levels_back, collapse_length, path)
    else:
      new_path = path[-levels_back:]
      new_path.append(str(json_data))
      yield " ".join(new_path)


class JSONReader():
  """JSON reader.
     Reads JSON documents with options to help suss out relationships between nodes.
  """
  def __init__(self, is_jsonl: Optional[bool] = False,) -> None:
    """Initialize with arguments."""
    super().__init__()
    self.is_jsonl = is_jsonl

  def load_data(self, input_file: str) -> List[Document]:
    """Load data from the input file."""
    documents = []
    with open(input_file, 'r') as file:
      load_data = json.load(file)
    for data in load_data:
      metadata = {"title": data['title'], 
                  "published_at": data['published_at'],
                  "source":data['source']}
      documents.append(Document(text=data['body'], metadata=metadata))
    return documents


def run_query(tokenizer, model, model_name, messages, temperature=0.1, max_new_tokens=512, **kwargs):
    messages = [ {"role": "user", "content": messages}, ]
    if GPU == True:
        if "Qwen" in model_name:
            input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", enable_thinking=False).cuda()
        else:   
            input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").cuda()
    else:
        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")

    generation_config = GenerationConfig(do_sample=True, temperature=temperature,
                                                **kwargs,)
    with torch.no_grad():
        generation_output = model.generate(
                        input_ids=input_ids,
                        generation_config=generation_config,
                        #pad_token_id=tokenizer.unk_token_id,
                        pad_token_id=tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                        output_scores=True,
                        max_new_tokens=max_new_tokens,
                    )
        #full_output = tokenizer.decode(generation_output[0], skip_special_tokens=False)
        
        generated_tokens = generation_output[0][input_ids.shape[-1]:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    """
        if "Llama-3" in model_name:
            # Llama 3 often includes the header, we can be more aggressive.
            if "<|start_header_id|>assistant<|end_header_id|>" in response:
                response = response.split("<|start_header_id|>assistant<|end_header_id|>")[1]
        elif "Qwen" in model_name:
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    """
        
    if STAGING:
        print(f"\n--- Model: {model_name} ---")
        print(f"Cleaned response: '{response}'")
        print("--------------------------")

    return response

def initialise_and_run_model(save_name, input_stage_1, model_name):

    print(f"Initializing model: {model_name}")
    if GPU == True:
        model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                     device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                               device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                               device_map="cpu")
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                   device_map="cpu")

  # You can change this instruction prompt if you want, but be careful. This one
  # is carefully tested and if you do not return information as defined here,
  # evaluation will fail.
    prefix = """Below is a question followed by some context from different sources. 
            Please answer the question based on the context. 
            The answer to the question is a word or entity. 
            If the provided information is insufficient to answer the question, 
            respond 'Insufficient Information'. 
            Answer directly without explanation."""

    print(f"Loading Stage 1 results from: {input_stage_1}")
    with open(input_stage_1, 'r') as file:
        doc_data = json.load(file)

    print('Remove saved file if exists.')
    rm_file(save_name)

    save_list = []
    for d in tqdm(doc_data, desc=f"RAG with {os.path.basename(model_name)}"):
        retrieval_list = d['retrieval_list']
        context = '--------------'.join(e['text'] for e in retrieval_list)
        prompt = f"{prefix}\n\nQuestion:{d['query']}\n\nContext:\n\n{context}"
        response = run_query(tokenizer, model, model_name, prompt)
        #print(response)
        save = {}
        save['query'] = d['query']
        save['prompt'] = prompt
        save['model_answer'] = response
        save['gold_answer'] = d['answer']
        save['question_type'] = d['question_type']
        #print(save)
        save_list.append(save)

  # Save Results
    print(f"Query processing completed. Saving the results to {save_name}")
    save_list_to_json(save_list,save_name)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(1)
        
    model_name = sys.argv[1]
    input_stage_1 = sys.argv[2]
    
    base_input_name = os.path.splitext(os.path.basename(input_stage_1))[0]
    model_identifier = model_name.split('/')[-1]
    
    if STAGING:
        output_file = f"/zone/home/s4979785/output/test_rag_{base_input_name}_{model_identifier}.json"
    else:
        output_file = f"/zone/home/s4979785/output/rag_{base_input_name}_{model_identifier}.json"

    initialise_and_run_model(output_file, input_stage_1, model_name)

    if STAGING:
        process = psutil.Process(os.getpid())
        peak_wset_bytes = process.memory_info().rss
        # peak_wset represents the peak working set size in bytes.
        peak_wset_gb = peak_wset_bytes / (1024 * 1024 * 1024)
        print(f"Peak working set size: {peak_wset_gb:.2f} GB")