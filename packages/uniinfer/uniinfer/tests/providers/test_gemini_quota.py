"""Gemini quota extraction — structured details first (AIP-193), marker fallback.

No regex: machine data comes from google.rpc error details (ErrorInfo
metadata for the Generative Language API, QuotaFailure for Cloud-style
surfaces); the human message split is the last resort only.
"""
from uniinfer.providers.gemini import _parse_gemini_quota


class _Err(Exception):
    def __init__(self, message="", response_json=None):
        super().__init__(message)
        self.message = message
        self.response_json = response_json


class TestStructuredDetails:
    def test_errorinfo_metadata_gemini_api_style(self):
        err = _Err(
            response_json={
                "error": {
                    "code": 429,
                    "status": "RESOURCE_EXHAUSTED",
                    "details": [
                        {
                            "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                            "reason": "RATE_LIMIT_EXCEEDED",
                            "metadata": {
                                "quota_metric": "GenerateRequestsPerDayPerProjectForGenerativeLanguage",
                                "quota_limit": "20",
                            },
                        }
                    ],
                }
            }
        )
        info = _parse_gemini_quota(err)
        assert info["quota_metric"].startswith("GenerateRequests")
        assert info["quota_limit"] == 20

    def test_quotafailure_cloud_style(self):
        err = _Err(
            response_json={
                "error": {
                    "details": [
                        {
                            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                            "violations": [
                                {"quotaMetric": "RequestsPerMin", "quotaValue": "60"}
                            ],
                        }
                    ]
                }
            }
        )
        info = _parse_gemini_quota(err)
        assert info == {"quota_metric": "RequestsPerMin", "quota_limit": 60}

    def test_structured_wins_over_message(self):
        err = _Err(
            message="Quota exceeded for quota metric 'FromMessage' with limit '1'",
            response_json={
                "error": {
                    "details": [
                        {
                            "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                            "metadata": {"quota_metric": "FromDetails", "quota_limit": "99"},
                        }
                    ]
                }
            },
        )
        assert _parse_gemini_quota(err)["quota_metric"] == "FromDetails"


class TestMessageFallback:
    def test_classic_free_tier_message(self):
        err = _Err(
            "Quota exceeded for quota metric 'GenerateRequestsPerDayPerModel-FreeTier' with limit '20'"
        )
        info = _parse_gemini_quota(err)
        assert info["quota_metric"] == "GenerateRequestsPerDayPerModel-FreeTier"
        assert info["quota_limit"] == 20

    def test_no_quota_info(self):
        assert _parse_gemini_quota(_Err("irgendwas anderes")) == {}

    def test_non_numeric_limit_ignored(self):
        err = _Err("Quota exceeded for quota metric 'X' with limit 'unlimited'")
        info = _parse_gemini_quota(err)
        assert info == {"quota_metric": "X"}

    def test_marker_without_close_quote(self):
        err = _Err("abgeschnittene Meldung: quota metric 'GenerateRe")
        assert _parse_gemini_quota(err) == {}
