"""Tests for the canonical SSE decoder seam (uniinfer.sse)."""
import pytest

from uniinfer.sse import DONE, sse_events


def test_basic_data():
    events = list(sse_events(['data: {"a": 1}', "", "data: [DONE]", ""]))
    assert events[0].data == '{"a": 1}'
    assert events[1].data == "[DONE]"


def test_multiline_data_is_joined():
    events = list(sse_events(["data: line1", "data: line2", ""]))
    assert events[0].data == "line1\nline2"


def test_event_name_field():
    events = list(sse_events(["event: response.created", "data: {}", ""]))
    assert events[0].event == "response.created"
    assert events[0].data == "{}"


def test_comments_and_blank_ignored():
    events = list(sse_events([": keep-alive", "", "data: x", ""]))
    assert len(events) == 1
    assert events[0].data == "x"


def test_no_space_after_colon():
    events = list(sse_events(["data:nospace", ""]))
    assert events[0].data == "nospace"


def test_crlf_lines():
    events = list(sse_events(["data: crlf\r", "\r", "data: two\r", "\r"]))
    assert [e.data for e in events] == ["crlf", "two"]


def test_id_field_preserved():
    events = list(sse_events(["id: 42", "data: d", ""]))
    assert events[0].id == "42"


def test_empty_stream():
    assert list(sse_events([])) == []


def test_data_prefix_variants_none_match_done_constant():
    assert DONE == "[DONE]"


def test_final_event_without_blank_line_is_flushed():
    events = list(sse_events(['data: {"last": true}']))
    assert len(events) == 1
    assert events[0].data == '{"last": true}'
