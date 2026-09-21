"""Official Consumer Financial Protection Bureau (CFPB) API Client.

Provides a robust, configurable interface to retrieve live consumer financial complaints
directly from the official CFPB Consumer Complaint Database API:
https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/

Normalizes API responses into the project's canonical schema:
- text: Complaint narrative ('complaint_what_happened')
- category: Financial product category ('product')
- complaint_id: Unique complaint identifier ('complaint_id')
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import pandas as pd
import requests

logger = logging.getLogger(__name__)

DEFAULT_API_URL: str = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
DEFAULT_TIMEOUT: float = 25.0
DEFAULT_USER_AGENT: str = "Python-urllib/3.14"


class CFPBClient:
    """HTTP client for querying the official CFPB Consumer Complaint Database API.

    Attributes
    ----------
    base_url : str
        The base endpoint URL for the CFPB complaints search API.
    timeout : float
        HTTP request timeout in seconds.
    session : requests.Session
        Reused HTTP session with optimized headers for API compatibility.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_API_URL,
        timeout: float = DEFAULT_TIMEOUT,
        user_agent: str = DEFAULT_USER_AGENT,
        session: Optional[requests.Session] = None
    ) -> None:
        """Initialize the CFPB API client.

        Parameters
        ----------
        base_url : str, default=DEFAULT_API_URL
            Base API endpoint URL.
        timeout : float, default=25.0
            Timeout in seconds for API network requests.
        user_agent : str, default='Python-urllib/3.14'
            User-Agent string ensuring compatibility with CFPB edge gateways.
        session : Optional[requests.Session], optional
            Pre-configured requests session (useful for unit testing and mocking).
        """
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept-Encoding": "identity",
            "Accept": "application/json",
        })

    def fetch_raw_complaints(
        self,
        size: int = 25,
        frm: int = 0,
        search_term: Optional[str] = None,
        has_narrative: bool = True,
        extra_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Query the CFPB API endpoint for a single page of complaint records.

        Parameters
        ----------
        size : int, default=25
            Number of complaint records to retrieve per request (max 100).
        frm : int, default=0
            Pagination offset (from-index).
        search_term : Optional[str], optional
            Optional keyword or phrase to search within complaint text.
        has_narrative : bool, default=True
            Whether to filter for complaints flagged with published narratives.
        extra_params : Optional[Dict[str, Any]], optional
            Additional query parameters forwarded to the API.

        Returns
        -------
        Dict[str, Any]
            Parsed JSON payload returned by the CFPB API.

        Raises
        ------
        ValueError
            If size <= 0, frm < 0, or the API response is not valid JSON.
        requests.exceptions.HTTPError
            If the server responds with a 4xx or 5xx HTTP error code.
        requests.exceptions.RequestException
            For network, DNS, or connection failures.
        """
        if size <= 0:
            raise ValueError(f"size must be a positive integer, got {size}.")
        if frm < 0:
            raise ValueError(f"frm (offset) cannot be negative, got {frm}.")

        params: Dict[str, Any] = {
            "size": min(size, 100),
            "frm": frm,
            "no_aggs": "true",  # Disable heavy aggregation facets for speed
        }

        if has_narrative:
            params["has_narrative"] = "true"

        if search_term:
            params["search_term"] = str(search_term).strip()

        if extra_params:
            params.update(extra_params)

        try:
            response = self.session.get(
                self.base_url,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as err:
            logger.error(f"CFPB API HTTP Error {response.status_code}: {err}")
            raise
        except requests.exceptions.Timeout as err:
            logger.error(f"CFPB API request timed out after {self.timeout}s: {err}")
            raise
        except requests.exceptions.RequestException as err:
            logger.error(f"CFPB API network error: {err}")
            raise

        try:
            data = response.json()
        except Exception as err:
            raise ValueError(f"Failed to parse JSON response from CFPB API: {err}")

        if not isinstance(data, dict):
            raise ValueError(f"Unexpected API response type: expected dict, got {type(data).__name__}.")

        return data

    def normalize_hit(self, hit: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a single CFPB API Elasticsearch hit to project schema.

        Parameters
        ----------
        hit : Dict[str, Any]
            Raw hit object from 'hits.hits' containing '_source'.

        Returns
        -------
        Dict[str, Any]
            Normalized dictionary containing 'complaint_id', 'category', 'text',
            and standardized aliases.
        """
        source = hit.get("_source", {})
        complaint_id = str(source.get("complaint_id") or hit.get("_id", ""))
        product = str(source.get("product") or "Unspecified")

        # Narrative field in CFPB schema is 'complaint_what_happened'
        raw_text = source.get("complaint_what_happened")
        narrative = str(raw_text).strip() if raw_text is not None else ""

        normalized = {
            "complaint_id": complaint_id,
            "category": product,
            "text": narrative,
            # Standardized schema aliases compatible with data_loader
            "Complaint ID": complaint_id,
            "Product": product,
            "Consumer Complaint": narrative,
            "issue": source.get("issue"),
            "sub_product": source.get("sub_product"),
            "company": source.get("company"),
            "date_received": source.get("date_received"),
            "state": source.get("state"),
        }
        return normalized

    def fetch_complaints(
        self,
        max_records: int = 50,
        page_size: int = 25,
        search_term: Optional[str] = None,
        has_narrative: bool = True,
        drop_empty_narratives: bool = False
    ) -> pd.DataFrame:
        """Retrieve and normalize complaint records from the CFPB API across pages.

        Parameters
        ----------
        max_records : int, default=50
            Maximum number of complaint records to retrieve.
        page_size : int, default=25
            Number of records to fetch per pagination request.
        search_term : Optional[str], optional
            Optional keyword or query string to filter complaints.
        has_narrative : bool, default=True
            Whether to request complaints with narratives.
        drop_empty_narratives : bool, default=False
            If True, discards records whose narrative text is empty.
            Note: As of August 2026, the live CFPB database ceased publishing new
            narratives. Older historical narratives exist in the archive.

        Returns
        -------
        pd.DataFrame
            DataFrame of normalized complaints conforming to canonical schema
            ('text', 'category', 'complaint_id').

        Raises
        ------
        ValueError
            If max_records <= 0.
        """
        if max_records <= 0:
            raise ValueError(f"max_records must be a positive integer, got {max_records}.")

        records: List[Dict[str, Any]] = []
        offset = 0

        while len(records) < max_records:
            current_size = min(page_size, max_records - len(records))
            raw_response = self.fetch_raw_complaints(
                size=current_size,
                frm=offset,
                search_term=search_term,
                has_narrative=has_narrative
            )

            hits = raw_response.get("hits", {}).get("hits", [])
            if not hits:
                logger.info(f"No further hits returned by CFPB API at offset {offset}.")
                break

            for hit in hits:
                norm = self.normalize_hit(hit)
                if drop_empty_narratives and not norm["text"].strip():
                    continue
                records.append(norm)
                if len(records) >= max_records:
                    break

            offset += len(hits)
            # Guard against infinite loop if API returns fewer hits than requested
            if len(hits) < current_size:
                break

        if not records:
            logger.warning("CFPB API returned 0 matching records.")
            return pd.DataFrame(columns=[
                "complaint_id", "category", "text", "Complaint ID", "Product",
                "Consumer Complaint", "issue", "sub_product", "company", "date_received", "state"
            ])

        df = pd.DataFrame(records)
        return df


def fetch_cfpb_data(
    max_records: int = 50,
    base_url: str = DEFAULT_API_URL,
    timeout: float = DEFAULT_TIMEOUT,
    search_term: Optional[str] = None,
    drop_empty_narratives: bool = False
) -> pd.DataFrame:
    """Convenience functional interface to fetch normalized CFPB complaints.

    Parameters
    ----------
    max_records : int, default=50
        Maximum records to fetch.
    base_url : str, default=DEFAULT_API_URL
        CFPB API base URL.
    timeout : float, default=25.0
        Request timeout.
    search_term : Optional[str], optional
        Search keyword.
    drop_empty_narratives : bool, default=False
        Whether to drop records with empty narratives.

    Returns
    -------
    pd.DataFrame
        Normalized complaints DataFrame.
    """
    client = CFPBClient(base_url=base_url, timeout=timeout)
    return client.fetch_complaints(
        max_records=max_records,
        search_term=search_term,
        drop_empty_narratives=drop_empty_narratives
    )
