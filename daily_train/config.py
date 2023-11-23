import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Type, Union

import torch
from typing_extensions import Self

import daily_train.model
from daily_train.utils import find_multiple


@dataclass
class Config:
    name: str = ""
    hf_config: dict = field(default_factory=dict)
    block_size: int = 4096
    vocab_size: int = 50254
    padding_multiple: int = 512
    padded_vocab_size: Optional[int] = None
    n_layer: int = 16
    n_head: int = 32
    n_embd: int = 4096
    rotary_percentage: float = 0.25
    parallel_residual: bool = True
    bias: bool = True
    lm_head_bias: bool = False
    # to use multi-head attention (MHA), set this to `n_head` (default)
    # to use multi-query attention (MQA), set this to 1
    # to use grouped-query attention (GQA), set this to a value in between
    # Example with `n_head=4`
    # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
    # │ v ││ v ││ v ││ v │     │ v │    │ v │             │ v │
    # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
    #   │    │    │    │         │        │                 │
    # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
    # │ k ││ k ││ k ││ k │     │ k │    │ k │             │ k │
    # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
    #   │    │    │    │      ┌──┴──┐  ┌──┴──┐      ┌────┬──┴─┬────┐
    # ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐
    # │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │
    # └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘
    # ◀──────────────────▶  ◀──────────────────▶  ◀──────────────────▶
    #         MHA                    GQA                   MQA
    #   n_query_groups=4       n_query_groups=2      n_query_groups=1
    #
    # credit https://arxiv.org/pdf/2305.13245.pdf
    n_query_groups: Optional[int] = None
    shared_attention_norm: bool = False
    _norm_class: Literal["LayerNorm", "RMSNorm"] = "LayerNorm"
    norm_eps: float = 1e-5
    _mlp_class: Literal["GptNeoxMLP", "LLaMAMLP"] = "GptNeoxMLP"
    gelu_approximate: str = "none"
    intermediate_size: Optional[int] = None
    rope_condense_ratio: int = 1
    rope_base: int = 10000

    def __post_init__(self):
        if not self.name:
            self.name = self.hf_config.get("name", self.name)

        assert self.n_embd % self.n_head == 0
        self.head_size = self.n_embd // self.n_head

        # vocab size should be a power of 2 to be optimal on hardware. compute the closest value
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size, self.padding_multiple)
        else:
            # vocab size shouldn't be larger than padded vocab size
            self.vocab_size = min(self.vocab_size, self.padded_vocab_size)

        # compute the number of query groups
        if self.n_query_groups is not None:
            assert self.n_head % self.n_query_groups == 0
        else:
            self.n_query_groups = self.n_head

        # compute the intermediate size for MLP if not set
        if self.intermediate_size is None:
            if self._mlp_class == "LLaMAMLP":
                raise ValueError("The config needs to set the `intermediate_size`")
            self.intermediate_size = 4 * self.n_embd

        self.rope_n_elem = int(self.rotary_percentage * self.head_size)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        if name not in name_to_config:
            # search through all `config['hf_config']['name']`
            try:
                conf_dict = next(config for config in configs if name == config["hf_config"]["name"])
            except StopIteration:
                raise ValueError(f"{name!r} is not a supported config name")
        else:
            conf_dict = name_to_config[name]

        conf_dict = conf_dict.copy()
        if "condense_ratio" in kwargs:  # legacy name
            kwargs["rope_condense_ratio"] = kwargs.pop("condense_ratio")
        conf_dict.update(kwargs)
        return cls(**conf_dict)

    @classmethod
    def from_json(cls, path: Union[str, Path], **kwargs: Any) -> Self:
        with open(path, encoding="utf-8") as fp:
            json_kwargs = json.load(fp)
        if "condense_ratio" in json_kwargs:  # legacy name
            json_kwargs["rope_condense_ratio"] = json_kwargs.pop("condense_ratio")
        if "condense_ratio" in kwargs:  # legacy name
            kwargs["rope_condense_ratio"] = kwargs.pop("condense_ratio")
        if "org" in json_kwargs:  # legacy name
            json_kwargs["hf_config"] = {"name": json_kwargs["name"], "org": json_kwargs.pop("org")}
        if "org" in kwargs:  # legacy name
            kwargs["hf_config"] = {"name": kwargs.get("name", json_kwargs["name"]), "org": kwargs.pop("org")}
        json_kwargs.update(kwargs)
        return cls(**json_kwargs)

    @classmethod
    def from_checkpoint(cls, path: Path, **kwargs: Any) -> Self:
        """Automatically load `lit_config.json` and if it doesn't exist - a matching config from `daily_train/config.py`."""
        if (config_path := path / "lit_config.json").is_file():
            return cls.from_json(config_path, **kwargs)
        if (model_name := path.name) in name_to_config:
            return cls.from_name(model_name, **kwargs)
        raise FileNotFoundError(f"For {str(path)!r} neither 'lit_config.json' nor matching config exists.")

    @property
    def mlp_class(self) -> Type:
        # `self._mlp_class` cannot be the type to keep the config json serializable
        return getattr(daily_train.model, self._mlp_class)

    @property
    def norm_class(self) -> Type:
        # `self._norm_class` cannot be the type to keep the config json serializable
        if self._norm_class == "RMSNorm":
            from daily_train.rmsnorm import RMSNorm

            return RMSNorm
        return getattr(torch.nn, self._norm_class)


configs = []

baseline = [
    dict(
        name="baseline",
        hf_config=dict(),
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=12,
        n_embd=768,
        n_head=12,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",  # TinyLlama uses FusedRMSNorm
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=5632,
        n_query_groups=4,
    )
]
configs.extend(baseline)


##################
# Meta Code Llama
##################
code_llama = [
    # https://huggingface.co/codellama/CodeLlama-7b-hf/blob/main/config.json
    dict(
        name="CodeLlama-7b-hf",
        hf_config=dict(org="codellama", name="CodeLlama-7b-hf"),
        block_size=16384,
        vocab_size=32016,
        padding_multiple=16,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-05,
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-13b-hf/blob/main/config.json
    dict(
        name="CodeLlama-13b-hf",
        hf_config=dict(org="codellama", name="CodeLlama-13b-hf"),
        block_size=16384,
        vocab_size=32016,
        padding_multiple=16,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-05,
        _mlp_class="LLaMAMLP",
        intermediate_size=13824,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-34b-hf/blob/main/config.json
    dict(
        name="CodeLlama-34b-hf",
        hf_config=dict(org="codellama", name="CodeLlama-34b-hf"),
        block_size=16384,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=48,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-05,
        _mlp_class="LLaMAMLP",
        intermediate_size=22016,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-7b-Python-hf/blob/main/config.json
    dict(
        name="CodeLlama-7b-Python-hf",
        hf_config=dict(org="codellama", name="CodeLlama-7b-Python-hf"),
        block_size=16384,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-05,
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-13b-Python-hf/blob/main/config.json
    dict(
        name="CodeLlama-13b-Python-hf",
        hf_config=dict(org="codellama", name="CodeLlama-13b-Python-hf"),
        block_size=16384,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-05,
        _mlp_class="LLaMAMLP",
        intermediate_size=13824,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-34b-Python-hf/blob/main/config.json
    dict(
        name="CodeLlama-34b-Python-hf",
        hf_config=dict(org="codellama", name="CodeLlama-34b-Python-hf"),
        block_size=16384,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=48,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-05,
        _mlp_class="LLaMAMLP",
        intermediate_size=22016,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf/tree/main/config.json
    dict(
        name="CodeLlama-7b-Instruct-hf",
        hf_config=dict(org="codellama", name="CodeLlama-7b-Instruct-hf"),
        block_size=16384,
        vocab_size=32016,
        padding_multiple=16,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-05,
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-13b-Instruct-hf/blob/main/config.json
    dict(
        name="CodeLlama-13b-Instruct-hf",
        hf_config=dict(org="codellama", name="CodeLlama-13b-Instruct-hf"),
        block_size=2048,
        vocab_size=32016,
        padding_multiple=16,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-05,
        _mlp_class="LLaMAMLP",
        intermediate_size=13824,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf/blob/main/config.json
    dict(
        name="CodeLlama-34b-Instruct-hf",
        hf_config=dict(org="codellama", name="CodeLlama-34b-Instruct-hf"),
        block_size=16384,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=48,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-05,
        _mlp_class="LLaMAMLP",
        intermediate_size=22016,
        rope_base=1000000,
    ),
]
configs.extend(code_llama)



name_to_config = {config["name"]: config for config in configs}
