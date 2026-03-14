"""
Human voice removal — replicating BirdCLEF 2025 2nd-place approach.

Reference: VSydorskyy/BirdCLEF_2025_2nd_place / scripts/curate_dataset.py
  - Stage 1: power-based dB scan to find the vocalization window
  - Stage 2: Silero-VAD on that window with threshold=0.5
  - Decision: if speech ≥ 2 s AND speech starts after `speech_start_th` (8 s)
              → TRUNCATE audio to end just before speech begins
              (keeps audio length consistent; returns a start/end sample window)

Key difference from naive zero-filling:
  Truncation removes speech contamination without inserting silent gaps that
  could confuse Perch's temporal convolutions.

Usage:
    from src.audio.human_filter import SpeechFilter

    filt = SpeechFilter()
    start, end = filt.find_clean_window(audio_32k, sr=32000)
    clean_audio = audio_32k[start:end]
"""

from __future__ import annotations

import numpy as np
import torch
import torchaudio
from typing import Tuple

_VAD_SR = 16_000       # Silero-VAD requires exactly 16 kHz


class SpeechFilter:
    """
    Two-stage human speech detector mirroring the 2nd-place BirdCLEF 2025 solution.

    Parameters
    ----------
    sr : int
        Sample rate of input audio (default 32000).
    threshold : float
        Silero-VAD confidence threshold (2nd place used 0.5).
    speech_db_th : float
        Power threshold in dB below which a chunk is treated as silence
        when scanning for the vocalization start (default -50 dB).
    chunk_len : float
        Chunk length in seconds for the power scan (default 0.1 s).
    speech_min_duration : float
        A detected speech segment must be ≥ this many seconds to trigger
        truncation (default 2.0 s).
    speech_start_th : float
        If speech starts before this many seconds from the audio start it
        is treated as a false positive and ignored (default 8.0 s).
    speech_merge_th : float
        Merge consecutive speech segments separated by less than this many
        seconds into one (default 0.3 s).
    """

    def __init__(
        self,
        sr: int = 32_000,
        threshold: float = 0.5,
        speech_db_th: float = -50.0,
        chunk_len: float = 0.1,
        speech_min_duration: float = 2.0,
        speech_start_th: float = 8.0,
        speech_merge_th: float = 0.3,
    ):
        self.sr = sr
        self.threshold = threshold
        self.speech_db_th = speech_db_th
        self.chunk_len = chunk_len
        self.speech_min_duration = speech_min_duration
        self.speech_start_th = speech_start_th
        self.speech_merge_th = speech_merge_th

        self._vad_model = None
        self._get_ts = None

    # ------------------------------------------------------------------
    def _load_vad(self):
        if self._vad_model is not None:
            return
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
            verbose=False,
        )
        self._vad_model = model
        self._get_ts = utils[0]   # get_speech_timestamps

    # ------------------------------------------------------------------
    def _power_scan(self, audio: np.ndarray) -> Tuple[int, int]:
        """
        Stage 1: scan power in dB chunks to find the first and last chunk
        that exceed `speech_db_th`.  Returns (start_sample, end_sample).
        """
        chunk_samples = max(1, int(self.chunk_len * self.sr))
        n_chunks = len(audio) // chunk_samples
        if n_chunks == 0:
            return 0, len(audio)

        chunks = audio[: n_chunks * chunk_samples].reshape(n_chunks, chunk_samples)
        power = np.mean(chunks ** 2, axis=1)
        # avoid log(0)
        power = np.maximum(power, 1e-12)
        power_db = 10.0 * np.log10(power)

        active = np.where(power_db >= self.speech_db_th)[0]
        if len(active) == 0:
            return 0, len(audio)

        start_sample = int(active[0]) * chunk_samples
        end_sample   = int(active[-1] + 1) * chunk_samples
        return start_sample, end_sample

    # ------------------------------------------------------------------
    def _merge_segments(self, segments: list) -> list:
        """Merge segments whose gap is shorter than `speech_merge_th`."""
        if not segments:
            return segments
        merge_samples = int(self.speech_merge_th * self._vad_sr_used)
        merged = [dict(segments[0])]
        for seg in segments[1:]:
            if seg["start"] - merged[-1]["end"] <= merge_samples:
                merged[-1]["end"] = seg["end"]
            else:
                merged.append(dict(seg))
        return merged

    # ------------------------------------------------------------------
    def find_clean_window(
        self, audio: np.ndarray, sr: int | None = None
    ) -> Tuple[int, int]:
        """
        Return (start_sample, end_sample) of the longest clean (speech-free)
        window.  If no speech is found the full audio range is returned.

        Parameters
        ----------
        audio : np.ndarray  float32 audio at `sr` Hz
        sr    : int          sample rate (uses self.sr if None)
        """
        if sr is None:
            sr = self.sr

        self._load_vad()

        # Stage 1: power-based vocalization window
        win_start, win_end = self._power_scan(audio)

        # Resample window to 16 kHz for Silero
        window = audio[win_start:win_end]
        tensor = torch.from_numpy(window).float().unsqueeze(0)
        if sr != _VAD_SR:
            tensor_16k = torchaudio.functional.resample(
                tensor, orig_freq=sr, new_freq=_VAD_SR
            )
        else:
            tensor_16k = tensor
        audio_16k = tensor_16k.squeeze(0)
        self._vad_sr_used = _VAD_SR

        # Stage 2: Silero-VAD
        self._vad_model.reset_states()
        try:
            speech_ts = self._get_ts(
                audio_16k,
                self._vad_model,
                sampling_rate=_VAD_SR,
                threshold=self.threshold,
            )
        except Exception:
            return 0, len(audio)

        if not speech_ts:
            return 0, len(audio)

        # Merge nearby segments
        merged = self._merge_segments(speech_ts)

        # Scale timestamps back to original sample rate
        scale = sr / _VAD_SR
        speech_start_th_samples = int(self.speech_start_th * sr)
        speech_min_samples      = int(self.speech_min_duration * sr)

        # Find the first qualifying speech segment
        for seg in merged:
            seg_start_orig = win_start + int(seg["start"] * scale)
            seg_end_orig   = win_start + int(seg["end"]   * scale)
            duration       = seg_end_orig - seg_start_orig

            # Ignore: too short OR starts too early (likely FP)
            if duration < speech_min_samples:
                continue
            if seg_start_orig < speech_start_th_samples:
                continue

            # Truncate: return audio up to just before this speech segment
            return 0, seg_start_orig

        # No qualifying speech found → return full range
        return 0, len(audio)

    # ------------------------------------------------------------------
    def filter(self, audio: np.ndarray, sr: int | None = None) -> np.ndarray:
        """Convenience wrapper: returns the clean audio slice."""
        start, end = self.find_clean_window(audio, sr=sr)
        return audio[start:end]
