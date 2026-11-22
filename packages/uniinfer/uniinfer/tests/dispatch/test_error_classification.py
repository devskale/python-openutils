"""errors.map_provider_error classifies by exception CLASS NAME as well as
message text — str(httpx.ReadTimeout()) is often "", which surfaced as a naked
'<provider> error: ' with no cause (2026-10-02). A refactor dropping the
class-name branch would silently bring that back."""

import httpx
import pytest

from uniinfer.errors import map_provider_error


@pytest.mark.parametrize("exc", [
    httpx.ReadTimeout(""),
    httpx.ConnectTimeout(""),
    httpx.WriteTimeout(None),
    httpx.PoolTimeout(""),
])
def test_empty_str_timeouts_classify_by_class_name(exc):
    err = map_provider_error("prov", exc)
    assert type(err).__name__ == "TimeoutError"
    # the message is self-explanatory even when str(exc) was empty
    assert "timeout" in str(err).lower()
    assert len(str(err)) > len("prov timeout error: ")


def test_generic_exception_stays_provider_error():
    err = map_provider_error("prov", RuntimeError("boom"))
    assert type(err).__name__ == "ProviderError"
    assert "boom" in str(err)
