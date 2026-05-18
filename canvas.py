"""
CANVAS - Code-switching ANchor Vector Anchor Steering.

Translation-free, calibration-free, decoding-time hidden-state interpolation
for code-switched (CS) QA. At each upper decoder layer, target-language token
positions are interpolated toward the source-language (EN) mean direction:

    h[TGT_pos] <- (1 - alpha) * h[TGT_pos] + alpha * en_mean_per_layer

The interpolation strength alpha can be fixed or adapted per example from the
measured alignment signal:

    alpha = clip(alpha_base + k * alignment, alpha_min, alpha_max)

where `alignment` is the cosine similarity between the question content mean
and the EN-TGT contrastive direction (computed during prefill).
"""

from __future__ import annotations
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch


# ----------------------------------------------------------------------
#  TOKEN-LEVEL LANGUAGE TAGGING (Unicode + Latin-script langid)
# ----------------------------------------------------------------------

LATIN_SCRIPT_TGT_LANGS = {"esp", "fra", "bbc"}
PAIR_TO_LINGUA_LANG = {
    "esp": "SPANISH",
    "fra": "FRENCH",
    "ben": "BENGALI",
    "hin": "HINDI",
    "kor": "KOREAN",
    "mar": "MARATHI",
    "urd": "URDU",
    "zho": "CHINESE",
    "bbc": "INDONESIAN",   # Toba Batak fallback
}

_LINGUA_DETECTOR_CACHE: dict = {}


def _is_latin(ch: str) -> bool:
    if not ch.isalpha():
        return False
    name = unicodedata.name(ch, "")
    return "LATIN" in name


def _script_of(text: str) -> str:
    """Return one of: TGT_NONLATIN, LATIN, MIXED, SHARED."""
    if not text or not text.strip():
        return "SHARED"
    has_latin = False
    has_cjk = has_hangul = has_deva = has_arab = has_bengali = False
    for ch in text:
        cp = ord(ch)
        if _is_latin(ch):
            has_latin = True
        elif (0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF):
            has_cjk = True
        elif (0xAC00 <= cp <= 0xD7A3) or (0x1100 <= cp <= 0x11FF):
            has_hangul = True
        elif 0x0900 <= cp <= 0x097F:
            has_deva = True
        elif (0x0600 <= cp <= 0x06FF) or (0x0750 <= cp <= 0x077F):
            has_arab = True
        elif 0x0980 <= cp <= 0x09FF:
            has_bengali = True
    nonlatin = has_cjk or has_hangul or has_deva or has_arab or has_bengali
    if nonlatin and not has_latin:
        return "TGT_NONLATIN"
    if has_latin and not nonlatin:
        return "LATIN"
    if has_latin and nonlatin:
        return "MIXED"
    return "SHARED"


def _get_lingua(tgt_pair_code: str):
    """Build or fetch a two-language detector for a Latin-script pair."""
    if tgt_pair_code not in _LINGUA_DETECTOR_CACHE:
        try:
            from lingua import Language, LanguageDetectorBuilder
        except ImportError:
            _LINGUA_DETECTOR_CACHE[tgt_pair_code] = None
            return None
        tgt_lang_name = PAIR_TO_LINGUA_LANG.get(tgt_pair_code, None)
        if tgt_lang_name is None:
            _LINGUA_DETECTOR_CACHE[tgt_pair_code] = None
            return None
        try:
            tgt_lang = getattr(Language, tgt_lang_name)
        except AttributeError:
            _LINGUA_DETECTOR_CACHE[tgt_pair_code] = None
            return None
        det = LanguageDetectorBuilder.from_languages(
            Language.ENGLISH, tgt_lang).build()
        _LINGUA_DETECTOR_CACHE[tgt_pair_code] = det
    return _LINGUA_DETECTOR_CACHE[tgt_pair_code]


def tag_tokens(token_strs: List[str],
               pair_code: Optional[str] = None,
               token_ids: Optional[List[int]] = None,
               tokenizer=None) -> List[str]:
    """Return one of {'EN', 'TGT', 'SHARED'} per token.

    Pure-script detection for non-Latin TGT (Korean, Hindi, Chinese, etc.).
    For Latin-script TGT (Spanish, French, Toba Batak) we use lingua to
    disambiguate EN vs TGT. BPE tokenizers (Qwen, Llama, Mistral) need
    `token_ids` + `tokenizer` to decode byte-encoded surfaces correctly.
    """
    tags: List[str] = []
    tgt_pair = (pair_code or "").split("-")[0] if pair_code else None
    use_langid = tgt_pair in LATIN_SCRIPT_TGT_LANGS

    detector = _get_lingua(tgt_pair) if use_langid else None
    target_enum = None
    if detector is not None:
        from lingua import Language
        tgt_lang_name = PAIR_TO_LINGUA_LANG.get(tgt_pair)
        target_enum = getattr(Language, tgt_lang_name, None) if tgt_lang_name else None

    if token_ids is not None and tokenizer is not None:
        decoded_per_id = []
        for tid in token_ids:
            try:
                decoded_per_id.append(
                    tokenizer.decode([tid], skip_special_tokens=False))
            except Exception:
                decoded_per_id.append("")
        surface_iter = decoded_per_id
    else:
        surface_iter = [t.lstrip("▁").lstrip("Ġ").strip()
                        for t in token_strs]

    for surface in surface_iter:
        sc = _script_of(surface)
        if sc == "TGT_NONLATIN":
            tags.append("TGT")
        elif sc == "LATIN":
            cleaned = surface.strip()
            if detector is not None and target_enum is not None and len(cleaned) >= 2:
                lang = detector.detect_language_of(cleaned)
                tags.append("TGT" if lang == target_enum else "EN")
            else:
                tags.append("EN")
        elif sc == "MIXED":
            tags.append("TGT")
        else:
            tags.append("SHARED")
    return tags


# ----------------------------------------------------------------------
#  STEERING ENGINE
# ----------------------------------------------------------------------

@dataclass
class CANVASConfig:
    """CANVAS hyperparameters.

    Control layers L_ctrl = [transition_frac * L, ..., L - 1] (paper: 0.70 ->
    final 30% of layers).

    Fixed alpha:
        alpha = interp_alpha (in [0, 1]). 0 = no steering, 1 = full replace.

    Adaptive alpha (paper default, negative-slope rule):
        alpha = clip(interp_alpha - interp_alpha_k * gamma,
                     interp_alpha_min, interp_alpha_max)
        where gamma is the canvas-alignment score. gamma > 0 means the CS
        representation is already source-canvas aligned (decrease alpha);
        gamma < 0 means it is target-oriented (increase alpha).
        Paper defaults: alpha = clip(0.45 - 1.5 * gamma, [0.05, 0.75]).
    """
    transition_frac:        float = 0.70
    min_tokens_per_side:    int   = 1
    interp_alpha:           float = 0.45
    interp_alpha_adaptive:  bool  = True
    interp_alpha_k:         float = 1.5
    interp_alpha_min:       float = 0.05
    interp_alpha_max:       float = 0.75


def _get_decoder_layers(model):
    """Return the ModuleList of decoder layers for common HF architectures."""
    m = model
    while hasattr(m, "model") and not hasattr(m, "layers"):
        m = m.model
    if hasattr(m, "layers"):
        return m.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError(
        f"Cannot locate decoder layers on {type(model).__name__}; "
        "extend _get_decoder_layers() for this architecture.")


class CANVAS:
    """Wraps a (model, tokenizer) into a steered decoder."""

    def __init__(self, model, tok, cfg: Optional[CANVASConfig] = None):
        self.model = model
        self.tok = tok
        self.cfg = cfg or CANVASConfig()
        self.layers = _get_decoder_layers(model)
        self.n_layers = len(self.layers)
        start = int(round(self.n_layers * self.cfg.transition_frac))
        start = min(max(0, start), self.n_layers - 1)
        self.upper_layer_idx = list(range(start, self.n_layers))
        self.upper_start = self.upper_layer_idx[0]
        self.dtype = next(model.parameters()).dtype
        self._hooks: list = []
        # Per-question state
        self.en_mean_per_layer: Optional[List[torch.Tensor]] = None
        self.last_alignment: float = 0.0
        self.last_eff_alpha: float = 0.0
        self.last_n_en: int = 0
        self.last_n_tgt: int = 0

    def _compute_steering(self, hidden_states_tuple,
                          lang_tags: List[str], device) -> None:
        """Compute en_mean per upper layer and the alignment signal."""
        T_tok = len(lang_tags)
        en_mask = torch.tensor([t == "EN" for t in lang_tags],
                               device=device, dtype=torch.bool)
        tgt_mask = torch.tensor([t == "TGT" for t in lang_tags],
                                device=device, dtype=torch.bool)
        n_en, n_tgt = int(en_mask.sum()), int(tgt_mask.sum())
        self.last_n_en, self.last_n_tgt = n_en, n_tgt

        if (n_en < self.cfg.min_tokens_per_side
                or n_tgt < self.cfg.min_tokens_per_side):
            self.en_mean_per_layer = None
            self.last_alignment = 0.0
            return

        en_mean_per_layer: List[torch.Tensor] = []
        cos_per_layer: List[float] = []
        for layer in self.upper_layer_idx:
            h = hidden_states_tuple[layer + 1][0]  # [T, d]
            r_en  = h[en_mask].float().mean(dim=0)
            r_tgt = h[tgt_mask].float().mean(dim=0)
            v = r_en - r_tgt
            v_norm = v / (v.norm() + 1e-8)
            en_mean_per_layer.append(r_en.to(self.dtype))
            content_mask = en_mask | tgt_mask
            r_full = h[content_mask].float().mean(dim=0)
            r_full_n = r_full / (r_full.norm() + 1e-8)
            cos_per_layer.append((r_full_n @ v_norm).item())

        self.en_mean_per_layer = en_mean_per_layer
        self.last_alignment = float(sum(cos_per_layer) / len(cos_per_layer))

    def _attach_interp_hooks(self,
                             tgt_indices: List[int],
                             alpha: float) -> None:
        """Post-hooks that interpolate TGT positions toward en_mean at each upper layer.

            h[TGT_pos] <- (1 - alpha) * h[TGT_pos] + alpha * en_mean
        """
        if self.en_mean_per_layer is None or not tgt_indices:
            return
        alpha = min(1.0, max(0.0, alpha))

        for slot_i, layer in enumerate(self.upper_layer_idx):
            en_mean = self.en_mean_per_layer[slot_i]

            def make_hook(en_mean_, alpha_, indices_):
                def post_hook(module, args, output):
                    h = output[0] if isinstance(output, tuple) else output
                    B, T, d = h.shape
                    if T <= 1:                       # decode step
                        return output
                    valid = [i for i in indices_ if i < T]
                    if not valid:
                        return output
                    h_new = h.clone()
                    en_rep = en_mean_.to(h.dtype).to(h.device)
                    idx_t = torch.tensor(valid, device=h.device)
                    h_new[0, idx_t, :] = (
                        (1 - alpha_) * h_new[0, idx_t, :]
                        + alpha_ * en_rep.unsqueeze(0)
                    )
                    if isinstance(output, tuple):
                        return (h_new,) + output[1:]
                    return h_new
                return post_hook

            handle = self.layers[layer].register_forward_hook(
                make_hook(en_mean, alpha, tgt_indices))
            self._hooks.append(handle)

    def _detach_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    @torch.inference_mode()
    def generate(self,
                 input_ids: torch.Tensor,
                 attention_mask: torch.Tensor,
                 max_new_tokens: int,
                 eos_ids: List[int],
                 lang_tags: List[str]) -> Tuple[List[int], dict]:
        """Two-pass decode:
          1) clean prefill to compute en_mean and alignment
          2) steered prefill (TGT positions interpolated toward en_mean)
          3) greedy decode from the steered KV cache (no decode-time hooks)
        """
        device = input_ids.device

        # Pass 1: clean prefill
        out1 = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        self._compute_steering(out1.hidden_states, lang_tags, device)

        # Effective alpha (paper: alpha = clip(base - k * gamma, [min, max]))
        if self.cfg.interp_alpha_adaptive and self.last_alignment != 0.0:
            raw = (self.cfg.interp_alpha
                   - self.cfg.interp_alpha_k * self.last_alignment)
            eff_alpha = max(self.cfg.interp_alpha_min,
                            min(self.cfg.interp_alpha_max, raw))
        else:
            eff_alpha = self.cfg.interp_alpha
        self.last_eff_alpha = float(eff_alpha)

        # Pass 2: steered prefill
        if self.en_mean_per_layer is not None:
            tgt_indices = [i for i, t in enumerate(lang_tags) if t == "TGT"]
            self._attach_interp_hooks(tgt_indices, alpha=eff_alpha)
            try:
                out2 = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                )
            finally:
                self._detach_hooks()
            past = out2.past_key_values
            next_logits = out2.logits[:, -1, :]
        else:
            past = out1.past_key_values
            next_logits = out1.logits[:, -1, :]

        # Pass 3: greedy decode from steered KV cache
        generated: List[int] = []
        cur_mask = attention_mask

        nid = int(next_logits.argmax(-1).item())
        if nid in eos_ids:
            return generated, self._diagnostics()
        generated.append(nid)
        cid = torch.tensor([[nid]], dtype=torch.long, device=device)
        if cur_mask is not None:
            cur_mask = torch.cat(
                [cur_mask,
                 torch.ones(1, 1, dtype=cur_mask.dtype, device=device)],
                dim=1)

        for _ in range(max_new_tokens - 1):
            step = self.model(
                input_ids=cid, attention_mask=cur_mask,
                past_key_values=past, use_cache=True,
                return_dict=True,
            )
            past = step.past_key_values
            logits = step.logits[:, -1, :]
            nid = int(logits.argmax(-1).item())
            if nid in eos_ids:
                break
            generated.append(nid)
            cid = torch.tensor([[nid]], dtype=torch.long, device=device)
            if cur_mask is not None:
                cur_mask = torch.cat(
                    [cur_mask,
                     torch.ones(1, 1, dtype=cur_mask.dtype, device=device)],
                    dim=1)

        return generated, self._diagnostics()

    def _diagnostics(self) -> dict:
        return {
            "alignment":   self.last_alignment,
            "eff_alpha":   self.last_eff_alpha,
            "n_en":        self.last_n_en,
            "n_tgt":       self.last_n_tgt,
            "n_upper":     len(self.upper_layer_idx),
            "layer_start": self.upper_start,
        }


# ----------------------------------------------------------------------
#  HIGH-LEVEL CONVENIENCE WRAPPER
# ----------------------------------------------------------------------

def steered_generate(model, tok,
                     prompt_text: str,
                     q_cs: str,
                     pair_code: Optional[str],
                     max_len: int,
                     max_new: int,
                     cfg: Optional[CANVASConfig] = None) -> Tuple[str, dict]:
    """Tokenize a chat-templated prompt, tag tokens by language, run CANVAS.

    Parameters
    ----------
    prompt_text : full chat-templated prompt (system + question + gen prompt)
    q_cs        : raw CS question (used to locate question span inside prompt
                  so the system prompt is NOT tagged)
    pair_code   : e.g. 'kor-eng' (determines TGT language)
    cfg         : CANVASConfig
    """
    inputs = tok(prompt_text, return_tensors="pt", truncation=True,
                 max_length=max_len, add_special_tokens=False)
    input_ids = inputs["input_ids"]
    attn = inputs.get("attention_mask")
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    if attn is not None:
        attn = attn.to(device)

    token_ids_list = input_ids[0].tolist()
    token_strs = tok.convert_ids_to_tokens(token_ids_list)

    cs_lo = prompt_text.find(q_cs)
    if cs_lo == -1:
        lang_tags = tag_tokens(token_strs, pair_code,
                               token_ids=token_ids_list, tokenizer=tok)
    else:
        cs_hi = cs_lo + len(q_cs)
        try:
            offsets = tok(prompt_text, return_offsets_mapping=True,
                          add_special_tokens=False, truncation=True,
                          max_length=max_len)["offset_mapping"]
        except Exception:
            offsets = None

        if offsets is not None and len(offsets) == len(token_strs):
            in_span_idx = [i for i, (a, b) in enumerate(offsets)
                           if a < cs_hi and b > cs_lo]
            in_span_strs = [token_strs[i] for i in in_span_idx]
            in_span_ids  = [token_ids_list[i] for i in in_span_idx]
            in_span_tags = tag_tokens(in_span_strs, pair_code,
                                      token_ids=in_span_ids, tokenizer=tok)
            lang_tags = ["SHARED"] * len(token_strs)
            for j, i in enumerate(in_span_idx):
                lang_tags[i] = in_span_tags[j]
        else:
            lang_tags = tag_tokens(token_strs, pair_code,
                                   token_ids=token_ids_list, tokenizer=tok)

    if len(lang_tags) != input_ids.size(1):
        lang_tags = lang_tags[: input_ids.size(1)]
        while len(lang_tags) < input_ids.size(1):
            lang_tags.append("SHARED")

    eos = set()
    if isinstance(tok.eos_token_id, list):
        eos.update(tok.eos_token_id)
    elif tok.eos_token_id is not None:
        eos.add(tok.eos_token_id)
    for ts, tid in tok.get_added_vocab().items():
        if "end" in ts.lower():
            eos.add(tid)

    canvas = CANVAS(model, tok, cfg)
    gen_ids, diag = canvas.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=max_new,
        eos_ids=list(eos),
        lang_tags=lang_tags,
    )
    text = tok.decode(gen_ids, skip_special_tokens=True).strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    diag["n_total_tokens"] = input_ids.size(1)
    diag["lang_tag_counts"] = {
        "EN":     sum(1 for t in lang_tags if t == "EN"),
        "TGT":    sum(1 for t in lang_tags if t == "TGT"),
        "SHARED": sum(1 for t in lang_tags if t == "SHARED"),
    }
    return text, diag
