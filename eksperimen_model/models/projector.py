"""
projector.py
Stage 4: Two-Layer MLP Projector & Physical-SLM Wrapper Module.
Aligns continuous physical latent representations (384d) with the frozen SLM embedding space (1536d for Qwen2.5-1.5B).
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, List
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel


class PhysicalToLLMProjector(nn.Module):
    """
    Two-Layer MLP Projector with Layer Normalization and GELU activation.
    Maps 384-dimensional physical latent representations to the SLM embedding space (1536d).
    Trainable parameters: ~1,971,968 (~1.97M)
    """
    def __init__(self, in_dim: int = 384, hidden_dim: int = 1024, out_dim: int = 1536, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N_phys_tokens, in_dim)
        Returns: (B, N_phys_tokens, out_dim)
        """
        return self.net(x)


class PhysicalSLMWrapper(nn.Module):
    """
    Unified Multimodal Architecture:
    - Trainable: PhysicalToLLMProjector (~1.97M params)
    - Strictly Frozen: Base Language Model (Qwen2.5-1.5B-Instruct ~1.54B params)
    
    Gradient Flow:
    Forward pass retains activation graph of the SLM (with gradient checkpointing)
    so loss gradients flow back into the projector weights.
    """
    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
        in_dim: int = 384,
        hidden_dim: int = 1024,
        freeze_slm: bool = True,
        use_gradient_checkpointing: bool = True,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: Optional[str] = None
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path

        # 1. Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            padding_side="right"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2. Load Base Language Model
        print(f"[PhysicalSLMWrapper] Loading base SLM: {model_name_or_path} (dtype={torch_dtype})...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True
        )

        out_dim = self.llm.config.hidden_size
        print(f"[PhysicalSLMWrapper] SLM hidden size: {out_dim}")

        # 3. Freeze Base SLM parameters strictly
        if freeze_slm:
            print("[PhysicalSLMWrapper] Freezing base SLM weights (strictly frozen)...")
            for param in self.llm.parameters():
                param.requires_grad = False

        # 4. Optional Gradient Checkpointing on SLM
        if use_gradient_checkpointing:
            print("[PhysicalSLMWrapper] Enabling gradient checkpointing on SLM to minimize VRAM activation memory...")
            self.llm.gradient_checkpointing_enable()

        # 5. Initialize Trainable Projector
        target_device = next(self.llm.parameters()).device
        self.projector = PhysicalToLLMProjector(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            dropout=0.1
        )
        # Ensure projector matches device and computation dtype
        self.projector.to(device=target_device, dtype=torch_dtype)

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def forward(
        self,
        prefix_input_ids: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        physical_tokens: torch.Tensor,
        suffix_input_ids: torch.Tensor,
        suffix_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assembles inputs_embeds by inserting projected physical tokens between prefix and suffix:
          [Prefix Embeddings] + [Projected Physical Tokens] + [Suffix Embeddings]
        
        prefix_input_ids: (B, L_pre)
        physical_tokens: (B, N_phys, 384)
        suffix_input_ids: (B, L_suf)
        labels: Optional (B, Total_L) - Target loss computation strictly on assistant response
        """
        embed_fn = self.get_input_embeddings()

        # 1. Embed text tokens
        prefix_embeds = embed_fn(prefix_input_ids) # (B, L_pre, d_model)
        suffix_embeds = embed_fn(suffix_input_ids) # (B, L_suf, d_model)

        # 2. Project physical tokens
        # Ensure physical tokens match projector dtype and device
        phys_t = physical_tokens.to(dtype=self.projector.net[0].weight.dtype, device=prefix_embeds.device)
        phys_embeds = self.projector(phys_t)

        # 3. Concatenate embeddings along sequence dimension
        inputs_embeds = torch.cat([prefix_embeds, phys_embeds, suffix_embeds], dim=1)

        # 4. Construct unified attention mask
        B, N_phys, _ = phys_embeds.shape
        phys_attention_mask = torch.ones((B, N_phys), dtype=prefix_attention_mask.dtype, device=prefix_attention_mask.device)
        attention_mask = torch.cat([prefix_attention_mask, phys_attention_mask, suffix_attention_mask], dim=1)

        # 5. Forward pass through base SLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

        return outputs.loss, outputs.logits

    def generate_response(
        self,
        prefix_input_ids: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        physical_tokens: Optional[torch.Tensor],
        suffix_input_ids: torch.Tensor,
        suffix_attention_mask: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.2,
        top_p: float = 0.9
    ) -> List[str]:
        """
        Autoregressive generation for evaluation and inference.
        """
        self.eval()
        with torch.no_grad():
            embed_fn = self.get_input_embeddings()
            prefix_embeds = embed_fn(prefix_input_ids)
            suffix_embeds = embed_fn(suffix_input_ids)

            if physical_tokens is None:
                inputs_embeds = torch.cat([prefix_embeds, suffix_embeds], dim=1)
                attention_mask = torch.cat([prefix_attention_mask, suffix_attention_mask], dim=1)
            else:
                phys_t = physical_tokens.to(dtype=self.projector.net[0].weight.dtype, device=prefix_embeds.device)
                phys_embeds = self.projector(phys_t)
                inputs_embeds = torch.cat([prefix_embeds, phys_embeds, suffix_embeds], dim=1)
                B, N_phys, _ = phys_embeds.shape
                phys_attention_mask = torch.ones((B, N_phys), dtype=prefix_attention_mask.dtype, device=prefix_attention_mask.device)
                attention_mask = torch.cat([prefix_attention_mask, phys_attention_mask, suffix_attention_mask], dim=1)

            generation_args = dict(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            if temperature > 0.0:
                generation_args.update(temperature=temperature, top_p=top_p)
            generated_ids = self.llm.generate(**generation_args)

            decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            return decoded
