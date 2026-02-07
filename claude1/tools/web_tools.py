"""Web tools: fetch pages, make HTTP requests, search the web."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote_plus

from claude1.config import MAX_TOOL_OUTPUT_CHARS
from claude1.tools.base import BaseTool

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None  # type: ignore[assignment]
    HTTPX_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BeautifulSoup = None  # type: ignore[assignment,misc]
    BS4_AVAILABLE = False


_HTTP_TIMEOUT = 30  # seconds
_DEFAULT_HEADERS = {
    "User-Agent": "Claude1/0.1 (coding-assistant)",
}


def _html_to_text(html: str) -> str:
    """Convert HTML to readable plain text via BeautifulSoup."""
    if not BS4_AVAILABLE:
        return html

    soup = BeautifulSoup(html, "html.parser")

    # Remove script and style elements
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)

    # Collapse multiple blank lines
    lines = [line.strip() for line in text.splitlines()]
    collapsed: list[str] = []
    prev_blank = False
    for line in lines:
        if not line:
            if not prev_blank:
                collapsed.append("")
            prev_blank = True
        else:
            collapsed.append(line)
            prev_blank = False

    return "\n".join(collapsed)


class WebFetchTool(BaseTool):
    """Fetch a URL and return its content as clean text."""

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return (
            "Fetch a web page or API endpoint and return its content. "
            "HTML is automatically converted to clean readable text. "
            "JSON responses are returned as-is. "
            f"Output is truncated at {MAX_TOOL_OUTPUT_CHARS} characters. "
            "Use for: reading documentation, fetching API responses, downloading data."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch",
                },
                "headers": {
                    "type": "object",
                    "description": "Optional HTTP headers as key-value pairs",
                },
            },
            "required": ["url"],
        }

    def execute(self, **kwargs: Any) -> str:
        if not HTTPX_AVAILABLE:
            return "Error: httpx is not installed. Run: pip install httpx"

        url = kwargs.get("url", "")
        headers = kwargs.get("headers") or {}

        if not url:
            return "Error: url is required"

        merged_headers = {**_DEFAULT_HEADERS, **headers}

        try:
            with httpx.Client(
                timeout=_HTTP_TIMEOUT,
                follow_redirects=True,
                headers=merged_headers,
            ) as client:
                response = client.get(url)

            content_type = response.headers.get("content-type", "")

            if "json" in content_type:
                body = response.text
            elif "html" in content_type:
                body = _html_to_text(response.text)
            else:
                body = response.text

            result = f"[{response.status_code}] {url}\n\n{body}"

            if len(result) > MAX_TOOL_OUTPUT_CHARS:
                result = result[:MAX_TOOL_OUTPUT_CHARS] + f"\n... [truncated at {MAX_TOOL_OUTPUT_CHARS} chars]"

            return result

        except httpx.TimeoutException:
            return f"Error: Request timed out after {_HTTP_TIMEOUT}s for {url}"
        except httpx.RequestError as e:
            return f"Error fetching {url}: {e}"
        except Exception as e:
            return f"Error: {e}"


class HttpRequestTool(BaseTool):
    """Generic HTTP client for any method."""

    @property
    def name(self) -> str:
        return "http_request"

    @property
    def description(self) -> str:
        return (
            "Make an HTTP request with any method (GET, POST, PUT, DELETE, PATCH). "
            "Returns status code, response headers, and body. "
            "Use for: calling APIs, submitting forms, webhooks, REST operations. "
            "Requires confirmation because HTTP requests can have side effects."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "description": "HTTP method: GET, POST, PUT, DELETE, PATCH",
                    "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                },
                "url": {
                    "type": "string",
                    "description": "The URL to send the request to",
                },
                "headers": {
                    "type": "object",
                    "description": "Optional HTTP headers as key-value pairs",
                },
                "body": {
                    "type": "string",
                    "description": "Raw request body (string)",
                },
                "json_body": {
                    "type": "object",
                    "description": "JSON request body (will be serialized and Content-Type set automatically)",
                },
                "timeout": {
                    "type": "integer",
                    "description": f"Request timeout in seconds (default: {_HTTP_TIMEOUT})",
                },
            },
            "required": ["method", "url"],
        }

    @property
    def requires_confirmation(self) -> bool:
        return True

    def execute(self, **kwargs: Any) -> str:
        if not HTTPX_AVAILABLE:
            return "Error: httpx is not installed. Run: pip install httpx"

        method = kwargs.get("method", "GET").upper()
        url = kwargs.get("url", "")
        headers = kwargs.get("headers") or {}
        body = kwargs.get("body")
        json_body = kwargs.get("json_body")
        timeout = kwargs.get("timeout", _HTTP_TIMEOUT)

        if not url:
            return "Error: url is required"

        merged_headers = {**_DEFAULT_HEADERS, **headers}

        try:
            with httpx.Client(
                timeout=timeout,
                follow_redirects=True,
                headers=merged_headers,
            ) as client:
                request_kwargs: dict[str, Any] = {}
                if json_body is not None:
                    request_kwargs["json"] = json_body
                elif body is not None:
                    request_kwargs["content"] = body

                response = client.request(method, url, **request_kwargs)

            # Format response
            resp_headers = dict(response.headers)
            result_parts = [
                f"Status: {response.status_code}",
                f"Headers: {resp_headers}",
                f"\nBody:\n{response.text}",
            ]
            result = "\n".join(result_parts)

            if len(result) > MAX_TOOL_OUTPUT_CHARS:
                result = result[:MAX_TOOL_OUTPUT_CHARS] + f"\n... [truncated at {MAX_TOOL_OUTPUT_CHARS} chars]"

            return result

        except httpx.TimeoutException:
            return f"Error: Request timed out after {timeout}s for {url}"
        except httpx.RequestError as e:
            return f"Error: {method} {url} failed: {e}"
        except Exception as e:
            return f"Error: {e}"


class WebSearchTool(BaseTool):
    """Search the web using DuckDuckGo HTML endpoint."""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web using DuckDuckGo and return results with titles, URLs, and snippets. "
            "No API key required. Use for: finding documentation, looking up error messages, "
            "discovering packages, researching solutions."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 5, max: 10)",
                },
            },
            "required": ["query"],
        }

    def execute(self, **kwargs: Any) -> str:
        if not HTTPX_AVAILABLE:
            return "Error: httpx is not installed. Run: pip install httpx"
        if not BS4_AVAILABLE:
            return "Error: beautifulsoup4 is not installed. Run: pip install beautifulsoup4"

        query = kwargs.get("query", "")
        max_results = min(kwargs.get("max_results", 5), 10)

        if not query:
            return "Error: query is required"

        search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"

        try:
            with httpx.Client(
                timeout=_HTTP_TIMEOUT,
                follow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                },
            ) as client:
                response = client.get(search_url)

            soup = BeautifulSoup(response.text, "html.parser")
            results = []

            for result_div in soup.select(".result")[:max_results]:
                title_el = result_div.select_one(".result__title a, .result__a")
                snippet_el = result_div.select_one(".result__snippet")

                title = title_el.get_text(strip=True) if title_el else "No title"
                href = title_el.get("href", "") if title_el else ""
                snippet = snippet_el.get_text(strip=True) if snippet_el else "No snippet"

                results.append(f"Title: {title}\nURL: {href}\nSnippet: {snippet}")

            if not results:
                return f"No results found for: {query}"

            return f"Search results for: {query}\n\n" + "\n\n---\n\n".join(results)

        except httpx.TimeoutException:
            return f"Error: Search timed out for query: {query}"
        except httpx.RequestError as e:
            return f"Error searching: {e}"
        except Exception as e:
            return f"Error: {e}"
