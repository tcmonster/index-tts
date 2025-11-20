"""
Utilities for managing prompt-related caches around IndexTTS2 without touching
official source files.

This module keeps lightweight LRU caches for speaker/emotion reference audio,
provides helpers to pre-compute embeddings during warmup, and exposes methods
to inject cached tensors back into the IndexTTS2 instance before inference.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import Dict, Optional

import torch
import torchaudio


def _clone_tensor(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if value is None:
        return None
    return value.detach().cpu().clone()


def _to_device(value: Optional[torch.Tensor], device: torch.device) -> Optional[torch.Tensor]:
    if value is None:
        return None
    return value.to(device)


def _audio_cache_key(audio_path: str) -> Optional[tuple]:
    try:
        stat = os.stat(audio_path)
    except FileNotFoundError:
        return None
    return os.path.abspath(audio_path), stat.st_mtime_ns, stat.st_size


class PromptCache:
    """Simple LRU caches for speaker and emotion reference encodings."""

    def __init__(self, max_speakers: int = 8, max_emotions: int = 8):
        self.max_speakers = max_speakers
        self.max_emotions = max_emotions
        self._speaker_cache: "OrderedDict[tuple, Dict[str, torch.Tensor]]" = OrderedDict()
        self._emotion_cache: "OrderedDict[tuple, Dict[str, torch.Tensor]]" = OrderedDict()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Public API used by app.py
    # ------------------------------------------------------------------ #
    def inject_speaker(self, tts, audio_path: Optional[str]) -> bool:
        """Copy cached tensors (if any) onto the TTS instance before inference."""
        if not audio_path:
            return False
        key = _audio_cache_key(audio_path)
        if key is None:
            return False
        with self._lock:
            payload = self._speaker_cache.get(key)
            if payload is None:
                return False
            self._speaker_cache.move_to_end(key)
        tts.cache_spk_audio_prompt = audio_path
        tts.cache_spk_cond = _to_device(payload["spk_cond"], tts.device)
        tts.cache_s2mel_style = _to_device(payload["style"], tts.device)
        tts.cache_s2mel_prompt = _to_device(payload["prompt_condition"], tts.device)
        tts.cache_mel = _to_device(payload["ref_mel"], tts.device)
        return True

    def inject_emotion(self, tts, audio_path: Optional[str]) -> bool:
        if not audio_path:
            return False
        key = _audio_cache_key(audio_path)
        if key is None:
            return False
        with self._lock:
            payload = self._emotion_cache.get(key)
            if payload is None:
                return False
            self._emotion_cache.move_to_end(key)
        tts.cache_emo_audio_prompt = audio_path
        tts.cache_emo_cond = _to_device(payload["emo_cond"], tts.device)
        return True

    def capture_from_tts(self, tts) -> None:
        """Persist the latest tensors produced by IndexTTS2 into the cache."""
        self._capture_speaker(tts)
        self._capture_emotion(tts)

    def ensure_speaker(self, tts, audio_path: Optional[str], verbose: bool = False) -> None:
        """Guarantee that the given speaker audio has cached tensors."""
        if not audio_path:
            return
        if self.inject_speaker(tts, audio_path):
            return
        payload = build_speaker_cache(tts, audio_path, verbose=verbose)
        if payload is None:
            return
        self._store_speaker(audio_path, payload)

    def ensure_emotion(self, tts, audio_path: Optional[str], verbose: bool = False) -> None:
        if not audio_path:
            return
        if self.inject_emotion(tts, audio_path):
            return
        payload = build_emotion_cache(tts, audio_path, verbose=verbose)
        if payload is None:
            return
        self._store_emotion(audio_path, payload)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _capture_speaker(self, tts) -> None:
        audio_path = getattr(tts, "cache_spk_audio_prompt", None)
        if not audio_path or getattr(tts, "cache_spk_cond", None) is None:
            return
        payload = {
            "spk_cond": _clone_tensor(tts.cache_spk_cond),
            "style": _clone_tensor(tts.cache_s2mel_style),
            "prompt_condition": _clone_tensor(tts.cache_s2mel_prompt),
            "ref_mel": _clone_tensor(tts.cache_mel),
        }
        self._store_speaker(audio_path, payload)

    def _capture_emotion(self, tts) -> None:
        audio_path = getattr(tts, "cache_emo_audio_prompt", None)
        if not audio_path or getattr(tts, "cache_emo_cond", None) is None:
            return
        payload = {"emo_cond": _clone_tensor(tts.cache_emo_cond)}
        self._store_emotion(audio_path, payload)

    def _store_speaker(self, audio_path: str, payload: Dict[str, torch.Tensor]) -> None:
        key = _audio_cache_key(audio_path)
        if key is None:
            return
        with self._lock:
            self._speaker_cache[key] = payload
            self._speaker_cache.move_to_end(key)
            while len(self._speaker_cache) > self.max_speakers:
                self._speaker_cache.popitem(last=False)

    def _store_emotion(self, audio_path: str, payload: Dict[str, torch.Tensor]) -> None:
        key = _audio_cache_key(audio_path)
        if key is None:
            return
        with self._lock:
            self._emotion_cache[key] = payload
            self._emotion_cache.move_to_end(key)
            while len(self._emotion_cache) > self.max_emotions:
                self._emotion_cache.popitem(last=False)


# ---------------------------------------------------------------------- #
# Warmup helpers (run during startup)
# ---------------------------------------------------------------------- #

@torch.no_grad()
def build_speaker_cache(tts, audio_path: str, verbose: bool = False) -> Optional[Dict[str, torch.Tensor]]:
    if not os.path.isfile(audio_path):
        if verbose:
            print(f"[prompt_cache] Skip speaker warmup, file missing: {audio_path}")
        return None

    audio, sr = tts._load_and_cut_audio(audio_path, 15, verbose)
    audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
    audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)

    inputs = tts.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
    input_features = inputs["input_features"].to(tts.device)
    attention_mask = inputs["attention_mask"].to(tts.device)
    spk_cond_emb = tts.get_emb(input_features, attention_mask)

    _, S_ref = tts.semantic_codec.quantize(spk_cond_emb)
    ref_mel = tts.mel_fn(audio_22k.to(spk_cond_emb.device).float())
    ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)
    feat = torchaudio.compliance.kaldi.fbank(
        audio_16k.to(ref_mel.device),
        num_mel_bins=80,
        dither=0,
        sample_frequency=16000,
    )
    feat = feat - feat.mean(dim=0, keepdim=True)
    style = tts.campplus_model(feat.unsqueeze(0))
    prompt_condition = tts.s2mel.models["length_regulator"](
        S_ref,
        ylens=ref_target_lengths,
        n_quantizers=3,
        f0=None,
    )[0]

    payload = {
        "spk_cond": _clone_tensor(spk_cond_emb),
        "style": _clone_tensor(style),
        "prompt_condition": _clone_tensor(prompt_condition),
        "ref_mel": _clone_tensor(ref_mel),
    }
    return payload


@torch.no_grad()
def build_emotion_cache(tts, audio_path: str, verbose: bool = False) -> Optional[Dict[str, torch.Tensor]]:
    if not os.path.isfile(audio_path):
        if verbose:
            print(f"[prompt_cache] Skip emotion warmup, file missing: {audio_path}")
        return None

    emo_audio, _ = tts._load_and_cut_audio(audio_path, 15, verbose, sr=16000)
    emo_inputs = tts.extract_features(emo_audio, sampling_rate=16000, return_tensors="pt")
    emo_input_features = emo_inputs["input_features"].to(tts.device)
    emo_attention_mask = emo_inputs["attention_mask"].to(tts.device)
    emo_cond_emb = tts.get_emb(emo_input_features, emo_attention_mask)

    return {"emo_cond": _clone_tensor(emo_cond_emb)}


def warmup_roles(tts, roles: Dict[str, Dict], prompt_cache: PromptCache, resolve_path, verbose: bool = False) -> None:
    """
    Pre-fetch speaker/emotion embeddings for all roles so the first batch request
    does not need to re-encode them.
    """
    for role_name, cfg in roles.items():
        speaker_audio = resolve_path(cfg["speaker_audio"])
        emotion_audio_cfg = cfg.get("emotion_audio")
        emotion_audio = resolve_path(emotion_audio_cfg) if emotion_audio_cfg else speaker_audio
        if verbose:
            print(f"[prompt_cache] Warming role '{role_name}' speaker: {speaker_audio}")
        prompt_cache.ensure_speaker(tts, speaker_audio, verbose=verbose)
        if verbose:
            print(f"[prompt_cache] Warming role '{role_name}' emotion: {emotion_audio}")
        prompt_cache.ensure_emotion(tts, emotion_audio, verbose=verbose)
