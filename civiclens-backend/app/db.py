import hashlib
import logging
import os
import secrets
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional

import bcrypt

logger = logging.getLogger(__name__)

DEFAULT_DESCRIPTION = "Standard daily routine"


# ------------------------------------------
# ERRORS
# ------------------------------------------

class DatabaseUnavailable(Exception):
    """Supabase is not configured or could not be reached."""


class EmailExistsError(Exception):
    """An account with this email already exists."""


# ------------------------------------------
# CLIENT
# ------------------------------------------

@lru_cache(maxsize=1)
def get_db():
    """Create the Supabase client once. Returns None if not configured."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        logger.warning("Supabase is not configured; data will not be saved.")
        return None

    try:
        from supabase import create_client

        return create_client(url, key)
    except Exception:
        logger.exception("Could not initialise the Supabase client.")
        return None


def _require_db():
    db = get_db()
    if db is None:
        raise DatabaseUnavailable()
    return db


# ------------------------------------------
# HELPERS
# ------------------------------------------

def _hash_password(password: str) -> str:
    # bcrypt only uses the first 72 bytes.
    return bcrypt.hashpw(
        password.encode("utf-8")[:72], bcrypt.gensalt()
    ).decode("utf-8")


def _verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(
            password.encode("utf-8")[:72], hashed.encode("utf-8")
        )
    except ValueError:
        return False


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _new_token() -> str:
    return secrets.token_urlsafe(32)


def _public(row: dict, token: Optional[str] = None) -> dict:
    """The account details that are safe to send to the frontend."""
    return {
        "citizen_id": str(row["id"]),
        "email": row["email"],
        "name": row["name"],
        "locality": row["locality"],
        "role": row["role"],
        "user_description": row.get("description") or DEFAULT_DESCRIPTION,
        "token": token,
    }


def _find_by_email(db, email: str) -> Optional[dict]:
    result = (
        db.table("citizens").select("*").eq("email", email).limit(1).execute()
    )
    return result.data[0] if result.data else None


def get_citizen_by_token(token: str) -> Optional[dict]:
    """Return the full citizen row for a session token, or None."""
    if not token:
        return None

    db = _require_db()
    result = (
        db.table("citizens")
        .select("*")
        .eq("session_token_hash", _hash_token(token))
        .limit(1)
        .execute()
    )
    return result.data[0] if result.data else None


# ------------------------------------------
# ACCOUNTS
# ------------------------------------------

def signup(
    email: str,
    password: str,
    name: str,
    locality: str,
    role: str,
    description: str,
) -> dict:
    """Create a citizen account. Raises EmailExistsError if taken."""
    db = _require_db()

    if _find_by_email(db, email):
        raise EmailExistsError()

    token = _new_token()

    try:
        result = (
            db.table("citizens")
            .insert(
                {
                    "email": email,
                    "password_hash": _hash_password(password),
                    "name": name,
                    "locality": locality,
                    "role": role,
                    "description": description or DEFAULT_DESCRIPTION,
                    "session_token_hash": _hash_token(token),
                }
            )
            .execute()
        )
    except Exception as exc:
        # Two sign-ups racing for the same email hit the unique index.
        if "23505" in str(exc) or "duplicate" in str(exc).lower():
            raise EmailExistsError() from exc
        raise

    return _public(result.data[0], token)


def login(email: str, password: str) -> Optional[dict]:
    """Verify credentials. Returns the account (with a new token) or None."""
    db = _require_db()

    row = _find_by_email(db, email)

    if (
        not row
        or not row.get("password_hash")
        or not _verify_password(password, row["password_hash"])
    ):
        return None

    token = _new_token()
    db.table("citizens").update(
        {"session_token_hash": _hash_token(token)}
    ).eq("id", row["id"]).execute()

    return _public(row, token)


def update_profile(
    token: str,
    name: str,
    locality: str,
    role: str,
    description: str,
) -> Optional[dict]:
    """Update the signed-in citizen's profile. None if the token is invalid."""
    db = _require_db()

    row = get_citizen_by_token(token)
    if not row:
        return None

    result = (
        db.table("citizens")
        .update(
            {
                "name": name,
                "locality": locality,
                "role": role,
                "description": description or DEFAULT_DESCRIPTION,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        .eq("id", row["id"])
        .execute()
    )

    return _public(result.data[0], token)


# ------------------------------------------
# LOCALITY CHECK HISTORY
# ------------------------------------------

def save_check(
    token: Optional[str],
    locality: str,
    notices_found: int,
    analysis: str,
) -> None:
    """Log a locality check for the signed-in citizen. Never raises."""
    if not token or get_db() is None:
        return

    try:
        row = get_citizen_by_token(token)
        if not row:
            return

        get_db().table("locality_checks").insert(
            {
                "citizen_id": row["id"],
                "locality": locality,
                "notices_found": notices_found,
                "analysis": analysis,
            }
        ).execute()

        logger.info("Saved locality check for %s.", row["email"])

    except Exception:
        logger.exception("Could not save the locality check to Supabase.")