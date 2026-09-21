"""Unit tests for CFPB API client and data normalization.

Tests client instantiation, request pagination, normalization into canonical schema,
empty narrative handling, and network/HTTP error handling using mocked responses.
"""

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
import requests

from src.cfpb_api import CFPBClient, DEFAULT_API_URL, fetch_cfpb_data
from src.data_loader import get_complaints_data, load_dataset_from_api


@pytest.fixture
def sample_api_raw_hit():
    """Mock Elasticsearch hit structure returned by the CFPB complaints API."""
    return {
        "_index": "complaint-public-v2",
        "_id": "1234567",
        "_source": {
            "complaint_id": "1234567",
            "product": "Credit card or prepaid card",
            "sub_product": "General-purpose credit card",
            "issue": "Problem when making payments",
            "company": "JPMORGAN CHASE & CO.",
            "state": "NY",
            "date_received": "2024-01-15T12:00:00.000Z",
            "complaint_what_happened": "I was charged an unexpected late fee despite on-time payment.",
            "has_narrative": True,
        }
    }


class TestCFPBClientConfiguration:
    """Test suite for client setup and parameters."""

    def test_default_initialization(self):
        """Client initializes with default endpoint URL and headers."""
        client = CFPBClient()
        assert client.base_url == DEFAULT_API_URL
        assert client.timeout == 25.0
        assert "User-Agent" in client.session.headers

    def test_custom_initialization(self):
        """Client respects custom base_url and timeout."""
        custom_url = "https://example.com/api/"
        client = CFPBClient(base_url=custom_url, timeout=10.0)
        assert client.base_url == "https://example.com/api/"
        assert client.timeout == 10.0


class TestCFPBResponseNormalization:
    """Test suite for mapping raw CFPB hits into canonical project schema."""

    def test_normalize_hit_fields(self, sample_api_raw_hit):
        """normalize_hit maps product to category and complaint_what_happened to text."""
        client = CFPBClient()
        normalized = client.normalize_hit(sample_api_raw_hit)

        assert normalized["complaint_id"] == "1234567"
        assert normalized["category"] == "Credit card or prepaid card"
        assert normalized["text"] == "I was charged an unexpected late fee despite on-time payment."
        assert normalized["Product"] == "Credit card or prepaid card"
        assert normalized["Consumer Complaint"] == normalized["text"]
        assert normalized["company"] == "JPMORGAN CHASE & CO."

    def test_normalize_hit_without_narrative(self):
        """Hits lacking complaint_what_happened are given an empty string narrative."""
        client = CFPBClient()
        raw_hit = {
            "_id": "9999",
            "_source": {
                "complaint_id": "9999",
                "product": "Mortgage",
                "complaint_what_happened": None,
            }
        }
        normalized = client.normalize_hit(raw_hit)
        assert normalized["complaint_id"] == "9999"
        assert normalized["category"] == "Mortgage"
        assert normalized["text"] == ""


class TestFetchComplaintsPagination:
    """Test suite for multi-page retrieval and filtering."""

    @patch.object(CFPBClient, "fetch_raw_complaints")
    def test_fetch_complaints_single_page(self, mock_fetch_raw, sample_api_raw_hit):
        """Single page response is parsed into DataFrame conforming to schema."""
        mock_fetch_raw.return_value = {
            "hits": {
                "total": {"value": 1, "relation": "eq"},
                "hits": [sample_api_raw_hit]
            }
        }
        client = CFPBClient()
        df = client.fetch_complaints(max_records=10, page_size=10)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "text" in df.columns
        assert "category" in df.columns
        assert "complaint_id" in df.columns
        assert df.iloc[0]["complaint_id"] == "1234567"

    @patch.object(CFPBClient, "fetch_raw_complaints")
    def test_fetch_complaints_pagination(self, mock_fetch_raw):
        """Retrieving 4 records with page_size=2 queries fetch_raw_complaints twice."""
        page_1_hit = {"_id": "1", "_source": {"complaint_id": "1", "product": "P1", "complaint_what_happened": "Text 1"}}
        page_2_hit = {"_id": "2", "_source": {"complaint_id": "2", "product": "P2", "complaint_what_happened": "Text 2"}}

        mock_fetch_raw.side_effect = [
            {"hits": {"hits": [page_1_hit, page_1_hit]}},
            {"hits": {"hits": [page_2_hit, page_2_hit]}},
        ]

        client = CFPBClient()
        df = client.fetch_complaints(max_records=4, page_size=2)
        assert len(df) == 4
        assert mock_fetch_raw.call_count == 2

    @patch.object(CFPBClient, "fetch_raw_complaints")
    def test_drop_empty_narratives_filtering(self, mock_fetch_raw):
        """drop_empty_narratives=True discards records without narrative text."""
        hit_with_text = {"_id": "1", "_source": {"complaint_id": "1", "product": "P1", "complaint_what_happened": "Real text"}}
        hit_empty_text = {"_id": "2", "_source": {"complaint_id": "2", "product": "P2", "complaint_what_happened": ""}}

        mock_fetch_raw.return_value = {
            "hits": {"hits": [hit_with_text, hit_empty_text]}
        }

        client = CFPBClient()
        df = client.fetch_complaints(max_records=5, drop_empty_narratives=True)
        assert len(df) == 1
        assert df.iloc[0]["complaint_id"] == "1"

    def test_invalid_max_records_raises_value_error(self):
        """max_records <= 0 raises ValueError."""
        client = CFPBClient()
        with pytest.raises(ValueError, match="positive integer"):
            client.fetch_complaints(max_records=0)


class TestCFPBErrorHandling:
    """Test suite for HTTP error handling, timeouts, and network failures."""

    def test_http_error_handling(self):
        """HTTP 404 or 500 raises requests.exceptions.HTTPError."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
        mock_session.get.return_value = mock_response

        client = CFPBClient(session=mock_session)
        with pytest.raises(requests.exceptions.HTTPError):
            client.fetch_raw_complaints()

    def test_timeout_error_handling(self):
        """Network timeout raises requests.exceptions.Timeout."""
        mock_session = MagicMock()
        mock_session.get.side_effect = requests.exceptions.Timeout("Request timed out")

        client = CFPBClient(session=mock_session)
        with pytest.raises(requests.exceptions.Timeout):
            client.fetch_raw_complaints()

    def test_invalid_json_raises_value_error(self):
        """Non-JSON response raises ValueError."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.side_effect = ValueError("No JSON object could be decoded")
        mock_session.get.return_value = mock_response

        client = CFPBClient(session=mock_session)
        with pytest.raises(ValueError, match="JSON"):
            client.fetch_raw_complaints()

    @patch("src.cfpb_api.CFPBClient.fetch_complaints")
    def test_unified_data_loader_dispatcher(self, mock_fetch):
        """get_complaints_data(source='api') correctly delegates to API loader."""
        mock_fetch.return_value = pd.DataFrame([{"complaint_id": "API-1", "category": "Card", "text": "Test"}])
        df_api = get_complaints_data(source="api", max_records=10)
        assert len(df_api) == 1
        assert df_api.iloc[0]["complaint_id"] == "API-1"

    def test_unknown_source_raises_value_error(self):
        """get_complaints_data with unsupported source raises ValueError."""
        with pytest.raises(ValueError, match="Unknown data source"):
            get_complaints_data(source="unsupported_source")
