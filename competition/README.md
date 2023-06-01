## In-context learning competition

Eventually, the quality of in-context learner needs to be evaluated on a diverse set of **unseen tasks**.

To make it easy for you, we prepared an evaluator, that will tell you how you stand on a diverse set of [SuperGLUE](https://super.gluebenchmark.com/tasks/) tasks.
You can evaluate your just-created model (or any other HuggingFace Seq2Seq model) as follows:

```shell
pip install -q rouge_score
cd competition
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python evaluator.py --model_name_or_path gaussalgo/mt5-base-priming-QA_en-cs
```
where you change `gaussalgo/mt5-base-priming-QA_en-cs` to other HF model identifier, or a path to your trained checkpoint.

If you want to make just a quick test, use smaller `--firstn` argument than default (1000), for instance `--firstn 50`. 
This will make the evaluation to finish 20-times faster.

### Your [optional] task:

1. Train an in-context learner that would beat the average ROUGE-L score of our base model
2. [Upload the model](https://huggingface.co/docs/transformers/model_sharing) to HuggingFace hub, with a short description of how you trained the model
3. Send us a link to uploaded HF model by June 15th

**We will promote all successful solvers to the NLP world by tagging you in a congrats LinkedIn post!**
