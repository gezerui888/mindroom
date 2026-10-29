"""Tests for cascaded-call text-to-speech through the OpenAI Realtime API."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from livekit.agents import APIConnectOptions, APIError

from mindroom.matrix_rtc import realtime_tts
from mindroom.matrix_rtc.realtime_tts import RealtimeTTS
from mindroom.matrix_rtc.voice_agent import SpeechServiceOptions, _cascaded_tts
from mindroom.voice.openai_realtime_speech import RealtimeSpeechError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@pytest.mark.asyncio
async def test_realtime_tts_pushes_pcm_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    """Realtime PCM chunks become 24 kHz mono frames for the call."""
    calls: list[tuple[str, dict[str, object]]] = []

    async def stream(text: str, **kwargs: object) -> AsyncIterator[bytes]:
        calls.append((text, kwargs))
        yield b"\x01\x00" * 2400
        yield b"\x02\x00" * 2400

    monkeypatch.setattr(realtime_tts, "stream_realtime_speech", stream)
    tts = RealtimeTTS(model="gpt-realtime-2.1-mini", api_key="sk-test", voice="ash")

    frames = [event.frame async for event in tts.synthesize("Hello there.")]

    assert calls == [
        ("Hello there.", {"model": "gpt-realtime-2.1-mini", "voice": "ash", "api_key": "sk-test", "base_url": None}),
    ]
    audio = b"".join(bytes(frame.data) for frame in frames)
    spoken = b"\x01\x00" * 2400 + b"\x02\x00" * 2400
    assert audio[: len(spoken)] == spoken
    # LiveKit ends a segment with silence.
    assert not audio[len(spoken) :].strip(b"\x00")
    assert {(frame.sample_rate, frame.num_channels) for frame in frames} == {(24000, 1)}


@pytest.mark.asyncio
async def test_realtime_tts_retries_a_segment_that_never_started(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure before any audio is retried, as a failed speech-endpoint request is."""
    attempts: list[str] = []

    async def stream(text: str, **_kwargs: object) -> AsyncIterator[bytes]:
        attempts.append(text)
        msg = "connection refused"
        raise RealtimeSpeechError(msg)
        yield b""

    monkeypatch.setattr(realtime_tts, "stream_realtime_speech", stream)
    tts = RealtimeTTS(model="gpt-realtime-2.1-mini", api_key="sk-test")

    with pytest.raises(APIError, match="connection refused"):
        [
            event
            async for event in tts.synthesize("Hello.", conn_options=APIConnectOptions(max_retry=1, retry_interval=0))
        ]
    assert attempts == ["Hello.", "Hello."]


@pytest.mark.asyncio
async def test_realtime_tts_does_not_respeak_a_started_segment(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure after audio played ends the segment with an error instead of speaking it again."""
    attempts: list[str] = []

    async def stream(text: str, **_kwargs: object) -> AsyncIterator[bytes]:
        attempts.append(text)
        yield b"\x01\x00" * 240
        msg = "connection closed"
        raise RealtimeSpeechError(msg)

    monkeypatch.setattr(realtime_tts, "stream_realtime_speech", stream)
    tts = RealtimeTTS(model="gpt-realtime-2.1-mini", api_key="sk-test")

    with pytest.raises(APIError, match="connection closed"):
        [
            event
            async for event in tts.synthesize("Hello.", conn_options=APIConnectOptions(max_retry=1, retry_interval=0))
        ]
    assert attempts == ["Hello."]


def test_realtime_tts_rejects_unknown_options() -> None:
    """Options the Realtime voice does not take fail when the call starts, as an invalid TTS option does."""
    with pytest.raises(TypeError):
        RealtimeTTS(model="gpt-realtime-2.1-mini", api_key="sk-test", speed=1.2)  # ty: ignore[unknown-argument]


def test_cascaded_tts_uses_realtime_for_realtime_models() -> None:
    """A gpt-realtime TTS model speaks through the Realtime adapter."""
    tts = _cascaded_tts(
        SpeechServiceOptions(
            provider="openai",
            model="gpt-realtime-2.1-mini",
            api_key="sk-test",
            extra_kwargs={"voice": "ash"},
        ),
    )

    assert isinstance(tts, RealtimeTTS)
    assert tts.model == "gpt-realtime-2.1-mini"


def test_cascaded_tts_keeps_speech_endpoint_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Other TTS models still use LiveKit's speech-endpoint client."""
    built: list[dict[str, object]] = []
    monkeypatch.setattr("livekit.plugins.openai.TTS", lambda **kwargs: built.append(kwargs) or "endpoint-tts")

    tts = _cascaded_tts(
        SpeechServiceOptions(
            provider="openai_compatible",
            model="kokoro",
            api_key="sk-no-key-required",
            base_url="http://localhost:8880/v1",
            extra_kwargs={"voice": "af_heart"},
        ),
    )

    assert tts == "endpoint-tts"
    assert built == [
        {
            "voice": "af_heart",
            "model": "kokoro",
            "api_key": "sk-no-key-required",
            "base_url": "http://localhost:8880/v1",
        },
    ]
