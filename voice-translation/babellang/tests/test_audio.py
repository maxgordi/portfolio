import asyncio
import numpy as np
import pytest
from babellang.audio.capture import list_devices, find_device_by_name


def test_list_devices_returns_list():
    devices = list_devices()
    assert isinstance(devices, list)
    assert len(devices) > 0


def test_devices_have_required_fields():
    for dev in list_devices():
        assert "index" in dev
        assert "name" in dev
        assert "max_input_channels" in dev
        assert "max_output_channels" in dev


def test_find_nonexistent_device():
    result = find_device_by_name("__this_device_does_not_exist_xyz__")
    assert result is None


def test_find_device_case_insensitive():
    devices = list_devices()
    input_devices = [d for d in devices if d["max_input_channels"] > 0]
    if input_devices:
        name = input_devices[0]["name"]
        result = find_device_by_name(name.upper()[:5])
        assert result is not None
