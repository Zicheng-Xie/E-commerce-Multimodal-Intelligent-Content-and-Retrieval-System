"""Only pre-content sandbox socket failures are eligible for the frozen retry."""

from __future__ import annotations

from recover_balanced_media import eligible


def test_no_http_denial_or_other_failure_is_retried() -> None:
    """Verify that no HTTP denial or other failure is retried."""
    assert eligible(dict(error_type="ConnectError", error="[WinError 10013] denied"))
    assert not eligible(dict(error_type="HTTPStatusError", error="HTTP420"))
    assert not eligible(dict(error_type="ReadTimeout", error="timeout"))
    assert not eligible(dict(error_type="ConnectError", error="10061 refused"))
