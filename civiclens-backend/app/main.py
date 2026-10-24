import asyncio
import hashlib
import logging
import os
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from io import BytesIO
from urllib.parse import urljoin, urlparse

from typing import Optional

import requests
import urllib3
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pypdf import PdfReader

from app.schemas import (
    AuthResponse,
    DocumentAnalysisRequest,
    DocumentAnalysisResponse,
    LocalityCheckRequest,
    LoginRequest,
    PolicyAnalysisResponse,
    PolicyImpactItem,
    SignupRequest,
    UserProfileBase,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from app import db as database


# ------------------------------------------
# CONFIGURATION
# ------------------------------------------
load_dotenv()

MODEL_ID = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

GCC_HOST = "chennaicorporation.gov.in"
GCC_NEWS_URL = "https://chennaicorporation.gov.in/gcc/latest-news/"

# Network / extraction limits (kept small for speed).
CONNECT_TIMEOUT = 5
READ_TIMEOUT = 10
MAX_DOWNLOAD_BYTES = 5 * 1024 * 1024
MAX_PDF_PAGES = 5
MAX_CHARS_PER_NOTICE = 3000

# Cache settings.
NOTICE_CACHE_TTL = 30 * 60      # 30 minutes
ANALYSIS_CACHE_TTL = 30 * 60    # 30 minutes
ANALYSIS_CACHE_MAX = 500

# Link titles that are site navigation, not announcements.
NAV_TITLES = {
    "home", "about", "about us", "contact", "contact us", "login",
    "sitemap", "site map", "feedback", "privacy policy", "terms of use",
    "disclaimer", "skip to main content", "read more", "more", "view all",
}

HTTP = requests.Session()
HTTP.headers.update({"User-Agent": "CivicLens/1.0 (Civic information application)"})

_notice_cache: dict[int, tuple[float, list[dict]]] = {}
_notice_lock = asyncio.Lock()
_analysis_cache: dict[str, tuple[float, str]] = {}

# Keeps references to background save tasks so they are not garbage collected.
_background_tasks: set[asyncio.Task] = set()


# ------------------------------------------
# APP SETUP
# ------------------------------------------

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Warm the notice cache in the background so the first click is fast."""

    async def warm():
        try:
            await get_notices_cached(5)
            logger.info("Notice cache warmed.")
        except Exception:
            logger.warning("Cache warm-up failed; will retry on first request.")

    task = asyncio.create_task(warm())
    yield
    task.cancel()


app = FastAPI(
    title="CivicLens Backend API",
    description="API for personalized civic policy analysis in Chennai.",
    version="1.0.0",
    lifespan=lifespan,
)

# Set FRONTEND_ORIGINS to comma-separated URLs during deployment.
# Local Streamlit origins are always allowed for development.
origins_env = os.getenv("FRONTEND_ORIGINS", "")
allowed_origins = [o.strip() for o in origins_env.split(",") if o.strip()]
allowed_origins += ["http://localhost:8501", "http://127.0.0.1:8501"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# ------------------------------------------
# GEMINI CLIENT
# ------------------------------------------

@lru_cache(maxsize=1)
def _build_client(api_key: str) -> genai.Client:
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=30_000),  # milliseconds
    )


def get_gemini_client() -> genai.Client:
    """Return a reusable Gemini client, created on first use."""
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="AI service is not configured.",
        )

    return _build_client(api_key)


def build_generation_config(
    system_instruction: Optional[str] = None,
) -> types.GenerateContentConfig:
    """Fast, short responses. Thinking is disabled only on 2.5 Flash models."""
    kwargs = {
        "temperature": 0.3,
        "max_output_tokens": 600,
    }

    if system_instruction:
        kwargs["system_instruction"] = system_instruction

    if "2.5-flash" in MODEL_ID:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

    return types.GenerateContentConfig(**kwargs)


# ------------------------------------------
# GCC NOTICE FETCHING
# ------------------------------------------

def fetch_gcc_notices(limit: int = 5) -> list[dict]:
    """Fetch announcement titles and links from the official GCC website."""

    try:
        response = HTTP.get(
            GCC_NEWS_URL,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        )
        response.raise_for_status()
    except requests.RequestException:
        logger.exception("Failed to fetch the official GCC news page.")
        raise HTTPException(
            status_code=502,
            detail="Unable to retrieve GCC announcements right now.",
        )

    soup = BeautifulSoup(response.text, "html.parser")

    # Ignore menus, headers and footers when looking for announcement links.
    for element in soup(["nav", "header", "footer", "script", "style"]):
        element.decompose()

    container = soup.find("main") or soup.find("article") or soup

    notices: list[dict] = []
    seen_urls: set[str] = set()

    for link in container.find_all("a", href=True):
        href = link["href"].strip()

        if href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue

        title = link.get_text(" ", strip=True)
        notice_url = urljoin(GCC_NEWS_URL, href).split("#")[0]

        if len(title) < 10 or title.lower() in NAV_TITLES:
            continue

        if urlparse(notice_url).hostname != GCC_HOST:
            continue

        if notice_url.rstrip("/") == GCC_NEWS_URL.rstrip("/"):
            continue

        if notice_url in seen_urls:
            continue

        seen_urls.add(notice_url)
        notices.append({"title": title, "url": notice_url})

        if len(notices) >= limit:
            break

    logger.info("Fetched %d GCC notices.", len(notices))
    return notices


# ------------------------------------------
# GCC ANNOUNCEMENT CONTENT EXTRACTION
# ------------------------------------------

def fetch_notice_content(notice: dict) -> dict:
    """Retrieve readable text from a GCC webpage or PDF."""

    notice_url = notice["url"]
    parsed_url = urlparse(notice_url)

    # Only retrieve content from the official GCC domain.
    if parsed_url.hostname != GCC_HOST:
        return {**notice, "content": "", "content_status": "Untrusted URL"}

    try:
        response = HTTP.get(
            notice_url,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        )
        response.raise_for_status()

        if len(response.content) > MAX_DOWNLOAD_BYTES:
            return {
                **notice,
                "content": "",
                "content_status": "Document exceeds size limit",
            }

        content_type = response.headers.get("Content-Type", "").lower()

        is_pdf = (
            "application/pdf" in content_type
            or parsed_url.path.lower().endswith(".pdf")
            or response.content.startswith(b"%PDF-")
        )

        if is_pdf:
            reader = PdfReader(BytesIO(response.content))
            text_parts = []

            for page in reader.pages[:MAX_PDF_PAGES]:
                page_text = page.extract_text() or ""
                if page_text:
                    text_parts.append(page_text)

            extracted_text = "\n".join(text_parts)
        else:
            soup = BeautifulSoup(response.text, "html.parser")

            for element in soup(
                ["script", "style", "nav", "footer", "header", "noscript"]
            ):
                element.decompose()

            body = soup.find("main") or soup.find("article") or soup
            extracted_text = body.get_text(separator=" ", strip=True)

        extracted_text = extracted_text[:MAX_CHARS_PER_NOTICE]

        if not extracted_text.strip():
            return {
                **notice,
                "content": "",
                "content_status": (
                    "No readable text found; the document "
                    "may be scanned or image-based."
                ),
            }

        return {
            **notice,
            "content": extracted_text,
            "content_status": "Text extracted successfully",
        }

    except Exception:
        logger.exception("Could not extract content from a GCC notice.")
        return {
            **notice,
            "content": "",
            "content_status": "Content extraction failed",
        }


async def get_notices_cached(limit: int) -> list[dict]:
    """Return notices with extracted text, cached since they are not user-specific."""

    async with _notice_lock:
        cached = _notice_cache.get(limit)
        if cached and time.time() - cached[0] < NOTICE_CACHE_TTL:
            return cached[1]

        notices = await asyncio.to_thread(fetch_gcc_notices, limit)

        if not notices:
            raise HTTPException(
                status_code=502,
                detail="No GCC announcement links could be found.",
            )

        extracted = await asyncio.gather(
            *[asyncio.to_thread(fetch_notice_content, n) for n in notices]
        )
        extracted = [n for n in extracted if n.get("content", "").strip()]

        if not extracted:
            raise HTTPException(
                status_code=502,
                detail=(
                    "GCC links were found, but their content "
                    "could not be extracted."
                ),
            )

        _notice_cache[limit] = (time.time(), extracted)
        return extracted


# ------------------------------------------
# GEMINI LOCALITY ANALYSIS
# ------------------------------------------

def run_gemini_locality_analysis(
    name: str,
    locality: str,
    role: str,
    description: str,
    notices: list[dict],
) -> str:
    """Explain retrieved GCC announcements for a specific citizen."""

    client = get_gemini_client()

    notice_text = "\n\n".join(
        (
            f"Title: {notice['title']}\n"
            f"Source: {notice['url']}\n"
            f"Extracted content:\n{notice['content']}"
        )
        for notice in notices
    )

    prompt = f"""
You are CivicLens, a civic information assistant for Chennai.

CITIZEN PROFILE:
- Name: {name}
- Locality: {locality}
- Role: {role}
- Daily context: {description}

ANNOUNCEMENTS RETRIEVED FROM THE OFFICIAL GCC WEBSITE:
{notice_text}

INSTRUCTIONS:
1. Keep the response under 200 words.
2. Explain only facts supported by the supplied announcement content.
3. Do not invent dates, deadlines, statistics or government decisions.
4. Explain which announcements may be relevant to the citizen.
5. Distinguish city-wide information from locality-specific information.
6. Do not claim an announcement affects the selected locality unless
   its content supports that conclusion.
7. If relevance cannot be established, state that clearly.
8. Include practical next steps only when supported by the sources.
9. Address the citizen respectfully.
10. Mention the relevant announcement title when possible.
"""

    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=build_generation_config(),
        )

        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("Gemini returned an empty response.")

        return text

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Gemini locality analysis failed.")
        raise HTTPException(
            status_code=502,
            detail="AI analysis is temporarily unavailable.",
        ) from exc


def analysis_cache_key(
    name: str, locality: str, role: str, description: str, notices: list[dict]
) -> str:
    # Key changes automatically when the notice set changes.
    notice_fingerprint = "|".join(n["url"] for n in notices)
    raw = f"{name}|{locality}|{role}|{description}|{notice_fingerprint}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def store_analysis(key: str, text: str) -> None:
    if len(_analysis_cache) >= ANALYSIS_CACHE_MAX:
        oldest = min(_analysis_cache, key=lambda k: _analysis_cache[k][0])
        _analysis_cache.pop(oldest, None)
    _analysis_cache[key] = (time.time(), text)


def extract_token(authorization: Optional[str]) -> Optional[str]:
    """Read the session token from an 'Authorization: Bearer <token>' header."""
    if not authorization:
        return None
    scheme, _, value = authorization.partition(" ")
    if scheme.lower() != "bearer" or not value.strip():
        return None
    return value.strip()


def save_in_background(
    token: Optional[str],
    locality: str,
    notices_found: int,
    analysis: str,
) -> None:
    """Log the check for the signed-in citizen without delaying the response."""
    if not token:
        return

    task = asyncio.create_task(
        asyncio.to_thread(
            database.save_check,
            token,
            locality,
            notices_found,
            analysis,
        )
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


# ------------------------------------------
# GOVERNMENT DOCUMENT ANALYSIS
# ------------------------------------------

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DOCUMENT_HOST_SUFFIXES = (".gov.in", ".nic.in")
DOCUMENT_READ_TIMEOUT = 20
MAX_DOC_PDF_PAGES = 10
MAX_CHARS_DOCUMENT = 8000

DOCUMENT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,text/html;q=0.9,*/*;q=0.8",
}

DOCUMENT_SYSTEM_INSTRUCTION = (
    "You are CivicLens Chennai. Explain official government documents "
    "in simple, everyday language.\n"
    "Rules:\n"
    "1. Keep your entire answer very short (under 150 words).\n"
    "2. Address the citizen directly by name.\n"
    "3. Use simple bullet points to explain: What this document is, "
    "How it affects them, and What action they need to take.\n"
    "4. Avoid complex legal jargon or long paragraphs.\n"
    "5. Use only facts found in the document. Do not invent dates, "
    "deadlines or decisions. If something is unclear, say so."
)


def is_allowed_document_host(hostname: Optional[str]) -> bool:
    """Only official government domains may be downloaded."""
    if not hostname:
        return False
    hostname = hostname.lower()
    return hostname in ("gov.in", "nic.in") or hostname.endswith(
        DOCUMENT_HOST_SUFFIXES
    )


def download_document(url: str) -> tuple[bytes, str]:
    """Download a government document; returns (content, content_type)."""

    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https") or not is_allowed_document_host(
        parsed.hostname
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Only official government links "
                "(ending in .gov.in or .nic.in) are supported."
            ),
        )

    try:
        try:
            response = HTTP.get(
                url,
                headers=DOCUMENT_HEADERS,
                timeout=(CONNECT_TIMEOUT, DOCUMENT_READ_TIMEOUT),
            )
        except requests.exceptions.SSLError:
            # Some state portals have misconfigured certificates.
            # The host was already verified as a government domain above.
            logger.warning("SSL error for %s; retrying without verification.",
                           parsed.hostname)
            response = HTTP.get(
                url,
                headers=DOCUMENT_HEADERS,
                timeout=(CONNECT_TIMEOUT, DOCUMENT_READ_TIMEOUT),
                verify=False,
            )

        response.raise_for_status()

    except requests.RequestException:
        logger.exception("Could not download the document.")
        raise HTTPException(
            status_code=502,
            detail="Could not download the document from that link.",
        )

    # A redirect must not lead outside government domains.
    if not is_allowed_document_host(urlparse(response.url).hostname):
        raise HTTPException(
            status_code=400,
            detail="That link redirects to a non-government website.",
        )

    if len(response.content) > MAX_DOWNLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="The document is too large to analyze (limit 5 MB).",
        )

    return response.content, response.headers.get("Content-Type", "").lower()


def extract_document_text(
    content: bytes, content_type: str, url: str
) -> tuple[str, bool]:
    """Return (text, is_pdf) for a downloaded PDF or web page."""

    is_pdf = (
        "application/pdf" in content_type
        or urlparse(url).path.lower().endswith(".pdf")
        or content.startswith(b"%PDF-")
    )

    text = ""

    if is_pdf:
        try:
            reader = PdfReader(BytesIO(content))
            parts = []
            for page in reader.pages[:MAX_DOC_PDF_PAGES]:
                page_text = page.extract_text() or ""
                if page_text:
                    parts.append(page_text)
            text = "\n".join(parts)
        except Exception:
            logger.exception("PDF text extraction failed.")
    else:
        soup = BeautifulSoup(content, "html.parser")
        for element in soup(
            ["script", "style", "nav", "footer", "header", "noscript"]
        ):
            element.decompose()
        body = soup.find("main") or soup.find("article") or soup
        text = body.get_text(separator=" ", strip=True)

    return text.strip()[:MAX_CHARS_DOCUMENT], is_pdf


def run_document_analysis(
    name: str,
    locality: str,
    role: str,
    description: str,
    document_url: str,
) -> str:
    """Download, read and explain a government document for one citizen."""

    content, content_type = download_document(document_url)
    text, is_pdf = extract_document_text(content, content_type, document_url)

    client = get_gemini_client()

    profile = (
        "CITIZEN PROFILE:\n"
        f"- Name: {name}\n"
        f"- Locality/Zone: {locality}\n"
        f"- Role: {role}\n"
        f"- Daily context: {description}\n"
    )

    if len(text) > 100:
        contents = f"{profile}\nGOVERNMENT DOCUMENT TEXT:\n{text}"
    elif is_pdf:
        # Scanned PDF: let Gemini read the file directly.
        contents = [
            types.Part.from_bytes(data=content, mime_type="application/pdf"),
            f"{profile}\nAnalyze the attached PDF and explain its impact "
            "on this citizen.",
        ]
    else:
        raise HTTPException(
            status_code=422,
            detail="No readable text was found at that link.",
        )

    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=contents,
            config=build_generation_config(DOCUMENT_SYSTEM_INSTRUCTION),
        )

        summary = (response.text or "").strip()
        if not summary:
            raise RuntimeError("Gemini returned an empty response.")

        return summary

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Gemini document analysis failed.")
        raise HTTPException(
            status_code=502,
            detail="AI analysis is temporarily unavailable.",
        ) from exc


# ------------------------------------------
# HEALTH CHECK
# ------------------------------------------

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "CivicLens Backend API",
    }


# ------------------------------------------
# LOCALITY ANALYSIS ENDPOINT
# ------------------------------------------

@app.post(
    "/api/v1/locality-check",
    response_model=PolicyAnalysisResponse,
)
async def check_locality(
    payload: LocalityCheckRequest,
    authorization: Optional[str] = Header(default=None),
):
    user = payload.user_profile

    name = user.name.strip()
    locality = user.locality.value
    role = user.role.value
    description = (user.user_description or "Standard daily routine").strip()
    limit = max(1, min(payload.limit or 5, 10))

    try:
        # 1-3. Notices (cached): links, content download, text extraction.
        notices = await get_notices_cached(limit)

        # 4. AI analysis (cached per profile + notice set).
        key = analysis_cache_key(name, locality, role, description, notices)
        cached = _analysis_cache.get(key)

        if cached and time.time() - cached[0] < ANALYSIS_CACHE_TTL:
            ai_analysis_text = cached[1]
        else:
            ai_analysis_text = await asyncio.to_thread(
                run_gemini_locality_analysis,
                name=name,
                locality=locality,
                role=role,
                description=description,
                notices=notices,
            )
            store_analysis(key, ai_analysis_text)

        # 5. Log this check for the signed-in citizen (background, never blocks).
        save_in_background(
            extract_token(authorization),
            locality,
            len(notices),
            ai_analysis_text,
        )

        # 6. Build the response.
        impact_items = [
            PolicyImpactItem(
                title=f"GCC Updates for {locality}",
                category="Civic Information",
                impact_summary=ai_analysis_text,
                source_url=GCC_NEWS_URL,
                confidence_score=0.5,
            )
        ]

        return PolicyAnalysisResponse(
            citizen_name=name,
            locality=locality,
            role=role,
            total_updates_found=len(notices),
            summary=(
                "GCC announcement content was retrieved and "
                "summarized. Open the original sources to "
                "verify complete details."
            ),
            impact_items=impact_items,
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Locality analysis request failed.")
        raise HTTPException(
            status_code=502,
            detail="Unable to complete the locality analysis. Please try again.",
        ) from exc


# ------------------------------------------
# DOCUMENT ANALYSIS ENDPOINT
# ------------------------------------------

@app.post(
    "/api/v1/analyze-document",
    response_model=DocumentAnalysisResponse,
)
async def analyze_document(payload: DocumentAnalysisRequest):
    user = payload.user_profile

    name = user.name.strip()
    locality = user.locality.value
    role = user.role.value
    description = (user.user_description or "Standard daily routine").strip()
    document_url = payload.document_url.strip()

    try:
        summary = await asyncio.to_thread(
            run_document_analysis,
            name,
            locality,
            role,
            description,
            document_url,
        )

        return DocumentAnalysisResponse(
            citizen_name=name,
            locality=locality,
            role=role,
            document_url=document_url,
            summary=summary,
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Document analysis request failed.")
        raise HTTPException(
            status_code=502,
            detail="Unable to analyze the document. Please try again.",
        ) from exc


# ------------------------------------------
# ACCOUNT ENDPOINTS (sign up, log in, profile)
# ------------------------------------------

@app.post(
    "/api/v1/auth/signup",
    response_model=AuthResponse,
    status_code=201,
)
async def signup(payload: SignupRequest):
    try:
        result = await asyncio.to_thread(
            database.signup,
            payload.email,
            payload.password,
            payload.name,
            payload.locality.value,
            payload.role.value,
            payload.user_description or "Standard daily routine",
        )
    except database.EmailExistsError:
        raise HTTPException(
            status_code=409,
            detail="An account with this email already exists. Please log in.",
        )
    except database.DatabaseUnavailable:
        raise HTTPException(
            status_code=503,
            detail="The database is not configured.",
        )
    except Exception as exc:
        logger.exception("Sign-up failed.")
        raise HTTPException(
            status_code=502,
            detail="Could not create the account. Please try again.",
        ) from exc

    return AuthResponse(**result)


@app.post("/api/v1/auth/login", response_model=AuthResponse)
async def login(payload: LoginRequest):
    try:
        result = await asyncio.to_thread(
            database.login,
            payload.email,
            payload.password,
        )
    except database.DatabaseUnavailable:
        raise HTTPException(
            status_code=503,
            detail="The database is not configured.",
        )
    except Exception as exc:
        logger.exception("Login failed.")
        raise HTTPException(
            status_code=502,
            detail="Could not log in. Please try again.",
        ) from exc

    if result is None:
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password.",
        )

    return AuthResponse(**result)


@app.put("/api/v1/profile", response_model=AuthResponse)
async def update_profile(
    payload: UserProfileBase,
    authorization: Optional[str] = Header(default=None),
):
    token = extract_token(authorization)

    if not token:
        raise HTTPException(
            status_code=401,
            detail="Session expired. Please log in again.",
        )

    try:
        result = await asyncio.to_thread(
            database.update_profile,
            token,
            payload.name.strip(),
            payload.locality.value,
            payload.role.value,
            (payload.user_description or "Standard daily routine").strip(),
        )
    except database.DatabaseUnavailable:
        raise HTTPException(
            status_code=503,
            detail="The database is not configured.",
        )
    except Exception as exc:
        logger.exception("Profile update failed.")
        raise HTTPException(
            status_code=502,
            detail="Could not update the profile. Please try again.",
        ) from exc

    if result is None:
        raise HTTPException(
            status_code=401,
            detail="Session expired. Please log in again.",
        )

    return AuthResponse(**result)