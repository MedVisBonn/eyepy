"""Exact integer arithmetic for Windows acquisition timestamps."""

import datetime

WINDOWS_EPOCH = datetime.datetime(1601, 1, 1)


def decode_windows_time(ticks: int) -> str:
    """Decode to datetime's microsecond resolution without floating-point drift.

    The final 100 ns digit is not representable by datetime; native raw
    ticks must be retained separately when exact re-encoding is
    required.
    """
    return (WINDOWS_EPOCH + datetime.timedelta(microseconds=ticks // 10)).isoformat()


def encode_windows_time(value: str) -> int:
    """Encode every representable microsecond exactly as ten 100 ns ticks."""
    parsed = datetime.datetime.fromisoformat(value)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(datetime.UTC).replace(tzinfo=None)
    delta = parsed - WINDOWS_EPOCH
    return ((delta.days * 86400 + delta.seconds) * 1000000 + delta.microseconds) * 10
