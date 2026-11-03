import struct

import pytest

from eyepy.io.he.e2e_format import DateTime
from eyepy.io.utils import IntDate


@pytest.mark.parametrize('adapter', [DateTime, IntDate], ids=['e2e', 'vol'])
@pytest.mark.parametrize('value,ticks,decoded', [
    ('1970-01-01T00:00:00.123456', 116444736001234560, '1970-01-01T00:00:00.123456'),
    ('2000-01-01T00:00:00.123457', 125911584001234570, '2000-01-01T00:00:00.123457'),
    ('1970-01-01T01:00:00.123456+01:00', 116444736001234560, '1970-01-01T00:00:00.123456'),
])
def test_windows_acquisition_times_use_exact_microseconds(adapter, value, ticks, decoded):
    # Independent epoch vectors; the final 100 ns digit is intentionally
    # outside datetime's resolution and must not cause floating-point drift.
    assert adapter.build(value) == struct.pack('<Q', ticks)
    assert adapter.parse(struct.pack('<Q', ticks + 9)) == decoded
