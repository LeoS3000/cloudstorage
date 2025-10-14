import json, sys
from tqdm import tqdm
import re
from collections import Counter
from tqdm import tqdm
import os
import nltk
from bert_score import score as bert_score_calc

def count_overlap(text1, text2):
  """
  Calculates the number of overlapping words between two texts.
  Returns: (overlap_count, length_of_text1, length_of_text2)
  """
  # Normalize texts by lowercasing and removing punctuation
  t1 = re.sub(r'[^A-Za-z0-9 ]+', '', text1.lower())
  t2 = re.sub(r'[^A-Za-z0-9 ]+', '', text2.lower())
  words1 = t1.split()
  words2 = t2.split()
  len1 = len(words1)
  len2 = len(words2)

  # Use a frequency counter for text1 to handle duplicate words correctly
  counts1 = Counter(words1)
  overlap_count = 0
  for word in words2:
    if counts1[word] > 0:
      overlap_count += 1
      counts1[word] -= 1

  return overlap_count, len1, len2

def comp_metrics_new(pred_list, gold_list):
  """Calculates token-overlap-based Precision, Recall, and F1 score."""
  prec_list, recall_list, f1_list = [], [], []
  for gold, pred in zip(gold_list, pred_list):

    # Correctly unpack the return values
    overlap, gold_len, pred_len = count_overlap(gold, pred)

    # Precision = overlap / prediction_length
    precision = float(overlap) / pred_len if pred_len > 0 else 0.0
    # Recall = overlap / gold_length
    recall = float(overlap) / gold_len if gold_len > 0 else 0.0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    prec_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)

  # Return the micro-average for each metric
  return sum(prec_list) / len(prec_list), sum(recall_list) / len(recall_list), sum(f1_list) / len(f1_list)

def calculate_exact_match(pred_list, gold_list):
    """Calculates the percentage of predictions that are an exact match."""
    exact_match_count = 0
    total_count = len(pred_list)
    if total_count == 0:
        return 0.0
    for pred, gold in zip(pred_list, gold_list):
        norm_pred = re.sub(r'[^\w\s]', '', pred.lower()).strip()
        norm_gold = re.sub(r'[^\w\s]', '', gold.lower()).strip()
        if norm_pred == norm_gold:
            exact_match_count += 1
    return (exact_match_count / total_count) * 100

def calculate_bertscore(pred_list, gold_list):
    """Calculates the average BERTScore F1 for the predictions."""
    if not pred_list or not gold_list:
        return 0.0
    # Suppress verbose output from bert_score
    P, R, F1 = bert_score_calc(pred_list, gold_list, model_type='roberta-large', lang='en', verbose=False)
    return F1.mean().item()

def get_gold(query_data, query):
  """Finds the gold answer for a given query."""
  for q in query_data:
    if q['query'] == query:
      return q['answer']
  return ''

def extract_answer(input_string):
  """Extracts the answer from the model's potentially verbose output."""
  # This regex is more robust to variations like "The answer is: '...'"
  match = re.search(r'answer is[:\s]*["\']?(.*?)["\']?$', input_string, re.IGNORECASE)
  return match.group(1).strip() if match else input_string.strip()

def run_evaluation(predictions_file, gold_labels_file, log_file):
  try:
    with open(predictions_file, 'r') as fh:
      doc_data = json.load(fh)
  except FileNotFoundError:
    print(f"Error: Prediction file not found at '{predictions_file}'", file=log_file)
    return
  except json.JSONDecodeError as e:
    print(f"Error decoding JSON from '{predictions_file}': {e}", file=log_file)
    # Skip this file and continue
    return


  with open(gold_labels_file, 'r') as fh:
    query_data = json.load(fh)

  type_data = {}
  overall_items = []

  print(f"Evaluating file: {predictions_file}", file=log_file)
  for d in tqdm(doc_data, desc=f"Processing {os.path.basename(predictions_file)}"):

    item = {
        'pred': extract_answer(d['model_answer']),
        'gold': get_gold(query_data, d['query']),
    }

    if item['gold']:
      question_type = d['question_type']
      if question_type not in type_data:
        type_data[question_type] = []
      type_data[question_type].append(item)
      overall_items.append(item)

  print("-" * 30, file=log_file)
  # Evaluate per question type
  for question_type, items in type_data.items():
    pred_list = [item['pred'] for item in items]
    gold_list = [item['gold'] for item in items]

    if not items: continue

    precision, recall, f1 = comp_metrics_new(pred_list, gold_list)
    exact_match = calculate_exact_match(pred_list, gold_list)
    bert_f1 = calculate_bertscore(pred_list, gold_list)

    print(f"Question Type: {question_type}", file=log_file)
    print(f" Token Precision: {precision:.2f}", file=log_file)
    print(f" Token Recall:    {recall:.2f}", file=log_file)
    print(f" Token F1 Score:  {f1:.2f}", file=log_file)
    print(f" Exact Match:     {exact_match:.2f}%", file=log_file)
    print(f" BERTScore F1:    {bert_f1:.2f}", file=log_file)
    print(file=log_file)

  print("-" * 30, file=log_file)
  # Evaluate overall
  if overall_items:
    overall_pred_list = [item['pred'] for item in overall_items]
    overall_gold_list = [item['gold'] for item in overall_items]

    overall_precision, overall_recall, overall_f1 = comp_metrics_new(overall_pred_list, overall_gold_list)
    overall_exact_match = calculate_exact_match(overall_pred_list, overall_gold_list)
    overall_bert_f1 = calculate_bertscore(overall_pred_list, overall_gold_list)

    print(f"Overall Metrics:", file=log_file)
    print(f" Token Precision: {overall_precision:.2f}", file=log_file)
    print(f" Token Recall:    {overall_recall:.2f}", file=log_file)
    print(f" Token F1 Score:  {f1:.2f}", file=log_file)
    print(f" Exact Match:     {overall_exact_match:.2f}%", file=log_file)
    print(f" BERTScore F1:    {overall_bert_f1:.2f}", file=log_file)
    print("-" * 30, file=log_file)


# prediction_file = 'output/llama2.json'
# prediction_file = "/content/drive/MyDrive/NLP2/output/qwen-qwen-reranker_CLEANED.json" # This file caused a JSONDecodeError

prediction_file = sys.argv[1]
gold_labels = 'data/rag.json'
log_file_path = f"/zone/home/s4979785/output/rag_eval_{prediction_file}.json"

try:
  nltk.data.find('tokenizers/punkt')
except LookupError:
  print("NLTK 'punkt' model not found. Downloading...")
  nltk.download('punkt', quiet=True)

try:
  nltk.data.find('tokenizers/punkt_tab')
except LookupError:
  print("NLTK 'punkt_tab' model not found. Downloading...")
  nltk.download('punkt_tab', quiet=True)

# Open the log file in append mode
with open(log_file_path, 'a') as log_file:
      run_evaluation(prediction_file, gold_labels, log_file)