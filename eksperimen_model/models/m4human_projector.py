"""Actual frozen-U to frozen-language interface; no target or state bypass."""
import re

import torch
from torch import nn
from torch.nn import functional as F

MARKER = "\n[M4HUMAN_PHYSICAL_INPUT]\n"


class M4HumanPhysicalProjectorV1(nn.Module):
    def __init__(self, output_dim=1536):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 512), nn.GELU(), nn.Linear(512, output_dim))
        self.register_buffer("alpha", torch.tensor(1.0))
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, U):
        if U.shape[-1] != 256 or not torch.isfinite(U).all():
            raise ValueError("U must be finite exact final tokens with width 256")
        return self.alpha * self.net(U.float())

    @torch.no_grad()
    def calibrate(self, U, mask, text_embeddings):
        if not mask.any() or text_embeddings.numel() == 0:
            raise ValueError("Empty alpha calibration support")
        if U.ndim != 3 or mask.shape != U.shape[:2] or not torch.isfinite(U[mask]).all():
            raise ValueError("Invalid alpha calibration sensor support")
        physical_norm = self.net(torch.where(mask[..., None], U, torch.zeros_like(U)).float())[mask].norm(dim=-1).median()
        alpha = text_embeddings.float().norm(dim=-1).median() / physical_norm.clamp_min(1e-8)
        if not torch.isfinite(alpha) or not 1e-3 <= alpha <= 1e3:
            raise ValueError("Degenerate alpha calibration; audit preprocessing")
        self.alpha.copy_(alpha)
        return float(alpha)


def answer_only_loss(logits, labels):
    """One causal shift; each QA contributes equally regardless of answer length."""
    valid = labels[:, 1:] != -100
    if not valid.any(dim=1).all():
        raise ValueError("Every training QA must contain answer tokens")
    losses = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), labels[:, 1:], ignore_index=-100, reduction="none")
    return ((losses * valid).sum(1) / valid.sum(1)).mean()


class FrozenM4HumanLanguage(nn.Module):
    def __init__(self, llm, tokenizer, max_prefix_tokens=192, max_total_tokens=256):
        super().__init__()
        self.llm = llm.eval()
        self.tokenizer = tokenizer
        self.projector = M4HumanPhysicalProjectorV1(llm.config.hidden_size)
        self.max_prefix_tokens = max_prefix_tokens
        self.max_total_tokens = max_total_tokens
        self.checkpoint_language = False
        for parameter in llm.parameters():
            parameter.requires_grad_(False)
        if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
            raise ValueError("Pinned tokenizer must have pad and EOS IDs")

    def train(self, mode=True):
        super().train(mode)
        self.llm.eval()
        return self

    def prefix_ids(self, question, task, answers):
        if not question or MARKER.strip() in question or any(token in question for token in self.tokenizer.all_special_tokens):
            raise ValueError("Question contains marker/chat control tokens")
        system = ('Answer only one JSON object with exactly keys "task" and "answer". '
                  f'Task: {task}. Allowed answers: {", ".join(answers)}. No explanation.')
        messages = [{"role": "system", "content": system}, {"role": "user", "content": MARKER + question}]
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if rendered.count(MARKER) != 1:
            raise ValueError("Chat template must preserve one unique marker")
        pre, post = rendered.split(MARKER)
        return [self.tokenizer.encode(text, add_special_tokens=False) for text in (pre, post)]

    def answer_ids(self, canonical_answer):
        # Derive the pinned assistant terminator; never assume BOS/EOS duplication.
        sentinel = "M4HUMAN_ANSWER_SENTINEL"
        rendered = self.tokenizer.apply_chat_template([{"role": "user", "content": "x"}, {"role": "assistant", "content": sentinel}], tokenize=False, add_generation_prompt=False)
        if rendered.count(sentinel) != 1:
            raise ValueError("Cannot derive assistant terminator from chat template")
        terminator = rendered.split(sentinel)[1]
        ids = self.tokenizer.encode(canonical_answer, add_special_tokens=False) + self.tokenizer.encode(terminator, add_special_tokens=False)
        if not ids or self.tokenizer.eos_token_id not in ids:
            raise ValueError("Assistant terminator must contain pinned EOS")
        return ids

    def assemble(self, samples, training=False, text_only=False):
        embed = self.llm.get_input_embeddings()
        device, dtype = embed.weight.device, embed.weight.dtype
        sequences, masks, label_rows = [], [], []
        for sample in samples:
            pre, post = self.prefix_ids(sample["question"], sample["task"], sample["answers"])
            text_pre = embed(torch.tensor(pre, device=device, dtype=torch.long))
            text_post = embed(torch.tensor(post, device=device, dtype=torch.long))
            U = sample["U"].to(device).detach().clone()
            token_mask = sample["token_mask"].to(device=device, dtype=torch.bool)
            if U.ndim != 2 or token_mask.shape != U.shape[:1]:
                raise ValueError("U/mask shape mismatch")
            if not torch.isfinite(U[token_mask]).all():
                raise ValueError("Nonfinite sensor-valid U is a failure")
            U = torch.where(token_mask[:, None], U, torch.zeros_like(U))
            physical = self.projector(U).to(dtype=dtype)
            physical = torch.where(token_mask[:, None], physical, torch.zeros_like(physical))
            if text_only:
                physical, token_mask = physical[:0], token_mask[:0]
            prefix = torch.cat((text_pre, physical, text_post))
            mask = torch.cat((torch.ones(len(pre), device=device, dtype=torch.bool), token_mask, torch.ones(len(post), device=device, dtype=torch.bool)))
            if len(prefix) > self.max_prefix_tokens:
                raise ValueError("Prefix exceeds locked length; no silent truncation")
            labels = torch.full((len(prefix),), -100, device=device, dtype=torch.long)
            if training:
                answer = torch.tensor(self.answer_ids(sample["canonical_answer"]), device=device)
                prefix = torch.cat((prefix, embed(answer)))
                mask = torch.cat((mask, torch.ones_like(answer, dtype=torch.bool)))
                labels = torch.cat((labels, answer))
                if len(prefix) > self.max_total_tokens:
                    raise ValueError("Training sample exceeds locked total length")
            sequences.append(prefix)
            masks.append(mask)
            label_rows.append(labels)
        length = max(map(len, sequences))
        assembled = embed.weight.new_zeros((len(samples), length, embed.embedding_dim))
        attention = torch.zeros((len(samples), length), device=device, dtype=torch.long)
        labels = torch.full_like(attention, -100)
        for index, (sequence, mask, row_labels) in enumerate(zip(sequences, masks, label_rows)):
            start = 0 if training else length - len(sequence)
            assembled[index, start:start + len(sequence)] = sequence
            attention[index, start:start + len(sequence)] = mask
            labels[index, start:start + len(sequence)] = row_labels
        positions = (attention.cumsum(-1) - 1).clamp_min(0).masked_fill(attention == 0, 0)
        return dict(inputs_embeds=assembled, attention_mask=attention, position_ids=positions), labels

    def forward(self, samples):
        inputs, labels = self.assemble(samples, training=True)
        if self.training and self.checkpoint_language:
            from torch.utils.checkpoint import checkpoint
            # Frozen language parameters stay in eval mode. Checkpoint the input
            # gradient graph directly; HF's training-only flag would do nothing.
            def language_logits(embeddings, attention, positions):
                return self.llm(inputs_embeds=embeddings, attention_mask=attention,
                                position_ids=positions, use_cache=False).logits
            logits = checkpoint(language_logits, inputs['inputs_embeds'], inputs['attention_mask'],
                                inputs['position_ids'], use_reentrant=False)
        else:
            logits = self.llm(**inputs, use_cache=False).logits
        return answer_only_loss(logits, labels)

    @torch.no_grad()
    def reference_generate(self, samples, max_new_tokens=64, text_only=False):
        """Portable greedy reference path. Recomputes prefix to avoid version-specific ID slicing."""
        self.eval()
        inputs, _ = self.assemble(samples, text_only=text_only)
        ids = [[] for _ in samples]
        finished = torch.zeros(len(samples), device=inputs["inputs_embeds"].device, dtype=torch.bool)
        first_logits = None
        for _ in range(max_new_tokens):
            logits = self.llm(**inputs, use_cache=False).logits[:, -1].float()
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite language logits")
            if first_logits is None:
                first_logits = logits.clone()
            next_ids = logits.argmax(-1)
            active = ~finished
            for index in range(len(samples)):
                if active[index]:
                    ids[index].append(int(next_ids[index]))
            finished |= next_ids == self.tokenizer.eos_token_id
            if finished.all():
                break
            inputs["inputs_embeds"] = torch.cat((inputs["inputs_embeds"], self.llm.get_input_embeddings()(next_ids)[:, None]), 1)
            inputs["attention_mask"] = torch.cat((inputs["attention_mask"], active.long()[:, None]), 1)
            inputs["position_ids"] = (inputs["attention_mask"].cumsum(-1) - 1).clamp_min(0).masked_fill(inputs["attention_mask"] == 0, 0)
        return [{"raw_ids": row, "raw_text": self.tokenizer.decode(row, skip_special_tokens=True)} for row in ids], first_logits

    @torch.no_grad()
    def generate(self, samples, max_new_tokens=64, text_only=False):
        self.eval()
        inputs, _ = self.assemble(samples, text_only=text_only)
        raw_first = self.llm(**inputs, use_cache=False).logits[:, -1].float()
        # Qwen 4.45.2 computes and slices positions from attention_mask during cached
        # generation. Supplying fixed full-prefix position_ids prevents that update.
        generation_inputs = {key: value for key, value in inputs.items() if key != "position_ids"}
        generated = self.llm.generate(**generation_inputs, do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
            repetition_penalty=1.0, use_cache=True, pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id, return_dict_in_generate=True, output_logits=True)
        if not generated.logits or not torch.allclose(raw_first, generated.logits[0].float(), atol=1e-2, rtol=1e-3):
            raise ValueError("Pinned generation raw-first-logit gate failed")
        # inputs_embeds-only Qwen 4.45.2 returns generated IDs only. Enforce at runtime.
        if generated.sequences.shape[1] > max_new_tokens:
            raise ValueError("Unexpected embeddings-only returned ID layout; re-run runtime gate")
        results = []
        for sequence in generated.sequences.tolist():
            if self.tokenizer.eos_token_id in sequence:
                sequence = sequence[:sequence.index(self.tokenizer.eos_token_id) + 1]
            results.append({"raw_ids": sequence, "raw_text": self.tokenizer.decode(sequence, skip_special_tokens=True)})
        return results, raw_first


def load_frozen_qwen(config, device, precision="fp16_amp_grad_scaler"):
    revision = config.get("revision")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("Pin full 40-character model/tokenizer commit before scientific run")
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("Real Qwen protocol requires an explicit CUDA device")
    if precision not in {"fp16_amp_grad_scaler", "bf16", "fp32"}:
        raise ValueError("Unsupported precision policy")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 capability gate failed")
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if transformers.__version__ != config["transformers_version"]:
        raise ValueError("Transformers runtime differs from pinned config")
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=revision, trust_remote_code=False)
    dtype = {"fp16_amp_grad_scaler": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
    attention = config.get('attention_backend', 'eager')
    if attention not in ('eager', 'sdpa'):
        raise ValueError('attention_backend must be eager or native PyTorch sdpa')
    llm = AutoModelForCausalLM.from_pretrained(config["model"], revision=revision, trust_remote_code=False, use_safetensors=True, torch_dtype=dtype, attn_implementation=attention).to(device)
    return FrozenM4HumanLanguage(llm, tokenizer).to(device)
