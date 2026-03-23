from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class AudioFrontendConfig:
    enabled: bool = False
    expect_client_aec: bool = True
    noise_suppression_enabled: bool = False
    noise_suppression_level: int = 1
    sample_rate: int = 16000
    frame_ms: int = 32


class AudioFrontend:
    """
    Unified audio frontend.

    Current implementation is a no-op passthrough. It exists so the pipeline can be
    wired end-to-end (decode once -> frontend -> VAD/ASR/voiceprint share the same PCM),
    then we can add AEC/NS implementations behind this interface.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.config = AudioFrontendConfig(
            enabled=bool(cfg.get("enabled", False)),
            expect_client_aec=bool(cfg.get("expect_client_aec", True)),
            noise_suppression_enabled=bool(cfg.get("noise_suppression_enabled", False)),
            noise_suppression_level=int(cfg.get("noise_suppression_level", 1)),
            sample_rate=int(cfg.get("sample_rate", 16000)),
            frame_ms=int(cfg.get("frame_ms", 32)),
        )
        # Stateful filters (lightweight, safe defaults).
        self._dc_y = 0.0
        self._ns_floor = 0.0

    def process_frame(self, pcm_bytes: bytes) -> bytes:
        """Process a single PCM frame (16-bit mono)."""
        if not self.config.enabled or not pcm_bytes:
            return pcm_bytes

        return self._process_pcm(pcm_bytes, is_sentence=False)

    def process_sentence(self, pcm_bytes: bytes) -> bytes:
        """Process a whole sentence buffer (16-bit mono)."""
        if not self.config.enabled or not pcm_bytes:
            return pcm_bytes

        return self._process_pcm(pcm_bytes, is_sentence=True)

    def reset(self) -> None:
        """Reset internal state (for stateful AEC/NS)."""
        self._dc_y = 0.0
        self._ns_floor = 0.0

    def _process_pcm(self, pcm_bytes: bytes, *, is_sentence: bool) -> bytes:
        """
        Minimal server-side audio frontend.

        Notes:
        - AEC is typically better on the client/device since server lacks far-end reference audio.
        - Here we implement only safe, lightweight processing that doesn't require reference signals.
        - Input/Output: 16-bit signed little-endian mono PCM.
        """
        # If the buffer length is odd, drop the last byte to keep int16 alignment.
        if len(pcm_bytes) < 2:
            return pcm_bytes
        if len(pcm_bytes) % 2 != 0:
            pcm_bytes = pcm_bytes[:-1]

        # Fast path: frontend enabled but NS disabled, still run DC removal (cheap).
        try:
            import array
        except Exception:
            return pcm_bytes

        samples = array.array("h")
        try:
            samples.frombytes(pcm_bytes)
        except Exception:
            return pcm_bytes

        # 1) DC removal: simple one-pole high-pass (leaky integrator on mean).
        # y[n] = x[n] - m[n],  m[n] = (1-a)*x[n] + a*m[n-1]
        # a close to 1 => slow mean tracking.
        a = 0.995
        m = self._dc_y
        for i in range(len(samples)):
            x = float(samples[i])
            m = (1.0 - a) * x + a * m
            y = x - m
            # clamp to int16
            if y > 32767.0:
                y = 32767.0
            elif y < -32768.0:
                y = -32768.0
            samples[i] = int(y)
        self._dc_y = m

        if not self.config.noise_suppression_enabled:
            return samples.tobytes()

        # 2) Lightweight "noise suppression": adaptive noise floor + soft gate.
        # This is NOT diarization; it's only to reduce steady background noise.
        lvl = int(self.config.noise_suppression_level)
        if lvl <= 0:
            return samples.tobytes()
        if lvl > 3:
            lvl = 3

        # Compute average absolute amplitude for this chunk.
        abs_sum = 0.0
        for s in samples:
            abs_sum += abs(int(s))
        avg_abs = abs_sum / max(1, len(samples))

        # Update noise floor with different speeds for frame/sentence.
        # Sentence buffers may include pauses; keep it conservative.
        alpha = 0.995 if not is_sentence else 0.998
        floor = self._ns_floor
        # Floor tracks low-energy segments; cap update to not jump too high.
        candidate = avg_abs
        if candidate < floor:
            floor = alpha * floor + (1.0 - alpha) * candidate
        else:
            # Slow rise
            floor = 0.999 * floor + 0.001 * candidate
        self._ns_floor = floor

        # Gate threshold and attenuation depend on level.
        # Higher level => higher threshold and stronger attenuation.
        thr = floor * (2.0 + 1.0 * lvl) + 50.0
        att = {1: 0.7, 2: 0.5, 3: 0.35}.get(lvl, 0.7)

        # Soft gate: below threshold attenuate, above keep.
        for i in range(len(samples)):
            x = float(samples[i])
            ax = abs(x)
            if ax < thr:
                y = x * att
            else:
                y = x
            if y > 32767.0:
                y = 32767.0
            elif y < -32768.0:
                y = -32768.0
            samples[i] = int(y)

        return samples.tobytes()
