# https://super.gluebenchmark.com/tasks
import abc
import json
import os
from typing import List, Sequence, Tuple, Optional, Dict, Union
from urllib.request import urlopen
from zipfile import ZipFile

from datasets import Dataset
from tqdm import tqdm

from promptsource.templates import DatasetTemplates


class SuperGLUETask(abc.ABC):

    promptsource_id: str
    dataset: Union[Dataset, None] = None

    data_file: str
    data: List[Tuple[str, str, str]] = []  # input, label, category
    template: Union[str, None] = None

    def __init__(self,
                 prompts_template: str,
                 hf_dataset_identifiers: Optional[Sequence[str]] = None,
                 hf_dataset_split: Optional[str] = None,
                 firstn: Optional[int] = None,
                 cache_dir: Optional[str] = "."):

        self.cache_dir = cache_dir
        if hasattr(self, "url"):
            self.data_file = self._maybe_download()

        template = DatasetTemplates(self.promptsource_id)
        self.prompt = template[prompts_template]
        self.label = prompts_template.replace(" ", "_")
        self.firstn = firstn

        if hf_dataset_identifiers is not None:
            from datasets import load_dataset
            self.dataset = load_dataset(*hf_dataset_identifiers)[hf_dataset_split]
            if firstn is not None:
                self.dataset = self.dataset.select(range(min(firstn, len(self.dataset))))

    def _maybe_download(self) -> str:
        fname = self.url.split("/")[-1]
        target_fpath = os.path.join(self.cache_dir, fname)
        if os.path.exists(target_fpath):
            return target_fpath
        else:
            print("Downloading %s" % self.url)
            resp = urlopen(self.url)
            with open(target_fpath, "wb") as data_f:
                data_f.write(resp.read())

            assert os.path.exists(target_fpath)

            return target_fpath

    def __str__(self) -> str:
        if self.label is not None:
            return self.__class__.__name__ + "-%s-%s" % (self.label, str(self.template))
        else:
            return self.__class__.__name__ + ("-%s" % str(self.template))


class Broadcoverage(SuperGLUETask):

    url: str = "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/AX-b.zip"
    promptsource_id: str = "super_glue/axb"

    def __init__(self, prompts_template: str = "GPT-3 style", **kwargs):
        super().__init__(prompts_template, **kwargs)
        self._read_input_label_pairs()

    def verbalize(self, input_texts: Sequence[str], label: str) -> Tuple[str, str]:
        # format from `load_dataset("super_glue", "cb")["validation"][1]`
        example = {"sentence1": input_texts[0],
                   "sentence2": input_texts[1],
                   "label": 0 if "not" in label else 1}
        return self.prompt.apply(example)

    def _read_input_label_pairs(self) -> None:

        with ZipFile(self.data_file) as zipfile:
            with zipfile.open("AX-b/AX-b.jsonl") as f:
                for l in f.readlines():
                    if self.firstn is not None and len(self.data) >= self.firstn:
                        break

                    entry = json.loads(l)

                    input_str, label = self.verbalize((entry["sentence1"], entry["sentence2"]), entry["label"])
                    # note that "logic" attribute is why we do not use HF datasets here -- and might consider
                    # replacing it with other tasks as well
                    cat = entry["logic"] if "logic" in entry else ""

                    self.data.append((input_str, label, cat))


class BoolQ(SuperGLUETask):

    url: str = "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/BoolQ.zip"
    promptsource_id: str = "super_glue/boolq"

    def __init__(self, prompts_template: str = "GPT-3 Style", split: str = "train", **kwargs):
        super().__init__(prompts_template, ["super_glue", "boolq"], split, **kwargs)

        # self.data = [(*("Context: %s. %s" % (pair[0], pair[1]) for pair in self.prompt.apply(example)),  # type:ignore
        #              self._example_cat(example)) for example in dataset]  # List[(input, target, category) tuples]
        self.data = [(*self.prompt.apply(example), self._example_cat(example)) for example in self.dataset]  # type: ignore
        self.data = [(("Context: %s" % sample[0]).replace("\nAnswer", "? Yes, or No? \nAnswer").replace("\n", " "), sample[1], sample[2])
                     for sample in self.data]

    def _example_cat(self, example: Dict[str, str]) -> str:
        return example["question"].split()[0]


class CommitmentBank(SuperGLUETask):

    promptsource_id = "super_glue/cb"

    def __init__(self, prompts_template: str = "GPT-3 style", split: str = "train", **kwargs):
        super().__init__(prompts_template, ["super_glue", "cb"], split, **kwargs)

        # no explicit categories either in HF dataset or in the original SGlue sources
        # raise NotImplementedError()
        # TODO: where to find categories?
        self.data = [(*self.prompt.apply(example), str(None)) for example in self.dataset]  # type:ignore


class WinogradSchema(SuperGLUETask):

    promptsource_id: str = 'super_glue/wsc.fixed'

    def __init__(self, prompts_template: str = "GPT-3 Style", split: str = "train", **kwargs):
        super().__init__(prompts_template, ["super_glue", "wsc"], split, **kwargs)

        self.data = [(*self.prompt.apply(example), self._example_cat(example)) for example in self.dataset]  # type:ignore

    def _example_cat(self, example: Dict[str, str]) -> str:
        # as category, we use the disambiguated pronouns, i.e. we use "he" examples as demonstrations
        return example["span2_text"].split()[0]


class CoPA(SuperGLUETask):

    promptsource_id: str = 'super_glue/copa'

    def __init__(self, prompts_template: str = "…As a result, C1 or C2?", split: str = "train", **kwargs):
        super().__init__(prompts_template, ["super_glue", "copa"], split, **kwargs)

        self.data = [(*self.prompt.apply(example), self._example_cat(example)) for example in self.dataset]  # type:ignore
        # we found promptsource template does not work for some samples - we filter these cases
        self.data = [tup for tup in self.data if len(tup) == 3]

    def _example_cat(self, example: Dict[str, str], num_categories: int = 10) -> str:
        # we use maximum token-level intersection as example category
        prem_tokens = set(example["premise"].lower().split())
        max_intersection = max(map(lambda ch: len(set(ch.lower().split()).intersection(prem_tokens)) / len(prem_tokens),
                                   (example["choice1"], example["choice2"])))
        max_intersection_grouped = int(max_intersection * num_categories) / num_categories

        return str(max_intersection_grouped)


class MultiRC(SuperGLUETask):

    promptsource_id: str = 'super_glue/multirc'

    def __init__(self, prompts_template: str = "is… a correct answer?", split: str = "train", **kwargs):
        super().__init__(prompts_template, ["super_glue", "multirc"], split, **kwargs)

        self.data = [(*self.prompt.apply(example), example["question"].split()[0].lower())  # type:ignore
                     for example in tqdm(self.dataset)]


class RTE(SuperGLUETask):

    promptsource_id: str = 'super_glue/rte'

    def __init__(self, prompts_template: str = "GPT-3 style", split: str = "train", **kwargs):
        super().__init__(prompts_template, ["super_glue", "rte"], split, **kwargs)

        # TODO: we are missing categories
        self.data = [(*self.prompt.apply(example), str(None)) for example in tqdm(self.dataset)]  # type:ignore


class WiC(SuperGLUETask):

    promptsource_id: str = 'super_glue/wic'

    def __init__(self, prompts_template: str = "GPT-3-prompt", split: str = "train", **kwargs):
        super().__init__(prompts_template, ["super_glue", "wic"], split, **kwargs)

        # TODO: we are missing categories
        self.data = [(*self.prompt.apply(example), str(None)) for example in tqdm(self.dataset)]  # type:ignore


class ReCoRD(SuperGLUETask):

    promptsource_id: str = 'super_glue/record'

    def __init__(self, prompts_template: str = "pick_one_option", split: str = "train", **kwargs):
        super().__init__(prompts_template, ["super_glue", "record"], split, **kwargs)

        # TODO: we are missing categories
        self.data = [(*self.prompt.apply(example), str(None)) for example in tqdm(self.dataset)]  # type:ignore


class Winogender(SuperGLUETask):

    promptsource_id: str = 'super_glue/axg'

    def __init__(self, prompts_template: str = "GPT-3 style", split: str = "test", **kwargs):
        super().__init__(prompts_template, ["super_glue", "axg"], split, **kwargs)

        # TODO: we are missing categories
        self.data = [(*self.prompt.apply(example), str(None)) for example in tqdm(self.dataset)]  # type:ignore


all_task_classes = Broadcoverage, BoolQ, CommitmentBank, WinogradSchema, CoPA, MultiRC, RTE, WiC, ReCoRD, Winogender
