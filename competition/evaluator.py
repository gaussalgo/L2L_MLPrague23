import argparse
import random
from typing import List, Tuple

import torch
from promptsource.templates import DatasetTemplates
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from superglue_tasks import all_task_classes

parser = argparse.ArgumentParser()

parser.add_argument("--model_name_or_path", type=str,
                    help="HuggingFace model id, or local path (checkpoint) with evaluated model and tokenizer.")
parser.add_argument("--tasks", default="axb,boolq,cb,wsc,copa,multirc,rte,wic,record,axg", type=str,
                    help="Coma-separated list of SuperGLUE tasks' ids. See default values for selection.")
parser.add_argument("--firstn", default=1000, type=int,
                    help="Number of samples used for evaluation within each task.")
parser.add_argument("--num_demonstrations", default=3, type=int,
                    help="Number of task demonstrations used in evaluation.")
random.seed(42)

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

selected_tasks_ids = args.tasks.split(",")
selected_tasks_classes = [SCls for SCls in all_task_classes
                          if any(t_id in SCls.promptsource_id for t_id in selected_tasks_ids)]

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)


def construct_fewshot_prompt(sample: Tuple[str, str, str],
                             sample_demonstrations: List[Tuple[str, str, str]]) -> str:
    demonstrations_str = "\n".join(["Input: %s Prediction: %s" % demo[:2] for demo in sample_demonstrations])
    return demonstrations_str + "\nInput: %s Prediction:" % sample[0]


max_evaluations = []

for task_i, SGLUETaskClass in enumerate(selected_tasks_classes):
    task_evaluations = []

    all_templates = DatasetTemplates(SGLUETaskClass.promptsource_id).all_template_names
    for template_i, template_name in enumerate(all_templates):
        evaluations = []
        task = SGLUETaskClass(prompts_template=template_name, firstn=args.firstn)
        if not task.data:
            # not applicable promptsource template
            continue

        for predicted_sample in tqdm(task.data,
                                     desc="Collecting prediction: Task %s/%s, Template %s/%s"
                                          % (task_i, len(selected_tasks_classes), template_i, len(all_templates))):
            # demonstrations collection
            demonstrations = []
            while len(demonstrations) < args.num_demonstrations:
                # pick next demonstration - not yet picked and not predicted
                next_demonstration = next(demo for demo in random.sample(task.data, len(task.data))
                                          if demo not in demonstrations and demo[0] != predicted_sample[0])
                demonstrations.append(next_demonstration)
            # full input text and label:
            input_prompt = construct_fewshot_prompt(predicted_sample, demonstrations)
            label = predicted_sample[1]
            # prediction:
            inputs = tokenizer(input_prompt, return_tensors="pt")
            prediction_ids = model.generate(**inputs.to(model.device))
            prediction = tokenizer.batch_decode(prediction_ids, skip_special_tokens=True)[0]
            # evaluation: ROUGE-L
            evaluation = scorer.score(label, prediction)['rougeL'].recall
            evaluations.append(evaluation)

        print("Task %s template '%s' ROUGE-L: %s"
              % (task.promptsource_id, template_name, sum(evaluations) / len(evaluations)))
        task_evaluations.append(sum(evaluations) / len(evaluations))

    print("Task %s highest ROUGE-L: %s" % (task.promptsource_id, max(task_evaluations)))
    max_evaluations.append(max(task_evaluations))

print("Model %s overall score: %s" % (args.model_name_or_path, sum(max_evaluations) / len(max_evaluations)))
