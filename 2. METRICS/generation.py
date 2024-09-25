import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np
from torch import Tensor

import evaluation_module.templates as templates
from evaluation_module.llm import GenerationError, HuggingFaceLLM


@dataclass
class GenerationConfig:
    model_name_or_path: str
    prompt_template_or_path: str
    tokenizer_name_or_path: Optional[str] = None
    model_class: Optional[str] = None
    tokenizer_class: Optional[str] = None
    huggingface_token: Optional[str] = None
    device: Optional[str] = None

    use_chat_template: bool = False
    generation_field: str = "_generated_text"
    logits_field_suffix: str = "_logits"
    new_token_ids_field_suffix: str = "_new_token_ids"
    prompt_token_ids_field_suffix: str = "_prompt_token_ids"
    target_token_ids_field_suffix: str = "_target_token_ids"
    target_fields: List[str] = field(default_factory=list)
    target_field_prompts: Dict[str, str] = field(default_factory=dict)
    variable_map: Dict[str, str] = field(default_factory=dict)
    generation_args: Dict = field(default_factory=dict)
    store_tensors_as_numpy: bool = False

    strip_string_patterns: List[str] = field(default_factory=list)


def load_config(config_file: str) -> GenerationConfig:
    with open(config_file, "r") as F:
        config = GenerationConfig(**json.load(F))

    return config


class Generation(object):
    """A class for generating text and log from a language model."""

    def __init__(self, config: GenerationConfig):
        model_args = {}
        if config.device:
            model_args["device_map"] = config.device

        self.llm = HuggingFaceLLM(model_name_or_path=config.model_name_or_path, model_class=config.model_class,
                                  tokenizer_name_or_path=config.tokenizer_name_or_path,
                                  tokenizer_class=config.tokenizer_class, model_args=model_args)
        if not self.llm.tokenizer.pad_token:
            self.llm.tokenizer.pad_token = self.llm.tokenizer.eos_token
            self.llm.tokenizer.pad_token_id = self.llm.tokenizer.eos_token_id
        self.llm.tokenizer.padding_side = "left"

        self.config = config
        if os.path.exists(self.config.prompt_template_or_path):
            with open(self.config.prompt_template_or_path, "rt") as F:
                self.config.prompt_template_or_path = F.read()
        elif os.path.exists(f"{templates.__path__[0]}/{self.config.prompt_template_or_path}"):
            with open(f"{templates.__path__[0]}/{self.config.prompt_template_or_path}", "rt") as F:
                self.config.prompt_template_or_path = F.read()

    def parse_prompt(self, variables: Dict[str, str], prompt_tpl: Optional[str] = None) -> str:
        prompt = prompt_tpl if prompt_tpl else self.config.prompt_template_or_path
        for variable in variables:
            if type(variable) in [str, int, float]:
                prompt = prompt.replace("{{" + variable + "}}", str(variables[variable]))
                if variable in self.config.variable_map:
                    prompt = prompt.replace("{{" + self.config.variable_map[variable] + "}}", variables[variable])
        return prompt

    def generate(self, data: Iterable[Dict]) -> Iterable[Dict]:

        self.llm.use_chat_template = self.config.use_chat_template

        for example in data:
            try:
                for target_field in self.config.target_fields:
                    if target_field in example:
                        prompt_tpl = None
                        if target_field in self.config.target_field_prompts:
                            prompt_tpl = self.config.target_field_prompts[target_field]
                        prompt = self.parse_prompt(example, prompt_tpl=prompt_tpl)
                        target_results = self.llm.generate(prompt=prompt, target=example[target_field],
                                                           return_logits=True, return_tokens=True,
                                                           generation_args=self.config.generation_args,
                                                           return_tensors=not self.config.store_tensors_as_numpy)
                        example[target_field + self.config.logits_field_suffix] = target_results["logits"]
                        example[target_field + "_target" +
                                self.config.logits_field_suffix] = target_results["target_logits"]
                        example[target_field + "_prompt" +
                                self.config.logits_field_suffix] = target_results["prompt_logits"]
                        example[target_field +
                                self.config.target_token_ids_field_suffix] = target_results["target_token_ids"]
                        example[target_field + self.config.new_token_ids_field_suffix] = target_results["new_token_ids"]
                        example[target_field +
                                self.config.prompt_token_ids_field_suffix] = target_results["prompt_token_ids"]

                if self.config.generation_field:
                    prompt = self.parse_prompt(example)
                    gen_results = self.llm.generate(prompt=prompt, return_logits=True, return_tokens=True,
                                                    return_tensors=not self.config.store_tensors_as_numpy,
                                                    generation_args=self.config.generation_args)
                    generated_text = gen_results["text"]
                    for re_pattern in self.config.strip_string_patterns:
                        generated_text = re.sub(re_pattern, "", generated_text)

                    example[self.config.generation_field] = generated_text
                    example[self.config.generation_field + self.config.logits_field_suffix] = gen_results["logits"]
                    example[self.config.generation_field +
                            self.config.new_token_ids_field_suffix] = gen_results["new_token_ids"]
                    example[self.config.generation_field +
                            self.config.prompt_token_ids_field_suffix] = gen_results["prompt_token_ids"]
                    example["error"] = None

            except GenerationError as e:
                example["error"] = str(e)

            yield example
