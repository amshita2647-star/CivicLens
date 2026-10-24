import re
from enum import Enum
from typing import List, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _clean_email(value: str) -> str:
    value = value.strip().lower()
    if not EMAIL_PATTERN.match(value):
        raise ValueError("Enter a valid email address.")
    return value


# ==========================================
# LOCALITY AND ROLE ENUMS
# ==========================================

class ChennaiLocality(str, Enum):
    ADYAR = "Adyar (Zone 13)"
    VELACHERY = "Velachery (Zone 13)"
    TNAGAR = "T. Nagar (Zone 10)"
    ANNA_NAGAR = "Anna Nagar (Zone 8)"
    MYLAPORE = "Mylapore (Zone 9)"
    KODAMBAKKAM = "Kodambakkam (Zone 10)"
    GUINDY = "Guindy (Zone 13)"
    ROYAPURAM = "Royapuram (Zone 5)"
    GENERAL = "Other / General Chennai"


class CitizenRole(str, Enum):
    RENTER = "Renter"
    HOMEOWNER = "Homeowner"
    BUSINESS_OWNER = "Small Business Owner"
    STUDENT = "Student"
    PARENT = "Parent"
    SENIOR_CITIZEN = "Senior Citizen"


# ==========================================
# USER PROFILE
# ==========================================

class UserProfileBase(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(
        default="Citizen",
        min_length=1,
        max_length=100,
    )

    locality: ChennaiLocality = Field(
        default=ChennaiLocality.ADYAR,
    )

    role: CitizenRole = Field(
        default=CitizenRole.RENTER,
    )

    user_description: Optional[str] = Field(
        default="Standard daily routine",
        max_length=500,
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Name cannot be empty.")

        return value.strip()

    @field_validator("user_description")
    @classmethod
    def normalize_description(
        cls,
        value: Optional[str],
    ) -> Optional[str]:
        if value is None:
            return "Standard daily routine"

        value = value.strip()

        return value or "Standard daily routine"


# ==========================================
# ACCOUNTS
# ==========================================

class SignupRequest(UserProfileBase):
    email: str = Field(max_length=254)
    password: str = Field(min_length=8, max_length=72)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        return _clean_email(value)


class LoginRequest(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(min_length=1, max_length=200)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        return _clean_email(value)


class AuthResponse(BaseModel):
    citizen_id: str
    email: str
    name: str
    locality: str
    role: str
    user_description: str
    token: Optional[str] = None


# ==========================================
# LOCALITY ANALYSIS
# ==========================================

class LocalityCheckRequest(BaseModel):
    user_profile: UserProfileBase

    limit: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of notices to analyze",
    )


class PolicyImpactItem(BaseModel):
    title: str
    category: str
    impact_summary: str
    source_url: Optional[str] = None

    confidence_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
    )


class PolicyAnalysisResponse(BaseModel):
    citizen_name: str
    locality: str
    role: str
    total_updates_found: int
    summary: str
    impact_items: List[PolicyImpactItem]


# ==========================================
# DOCUMENT ANALYSIS
# ==========================================

class DocumentAnalysisRequest(BaseModel):
    user_profile: UserProfileBase

    document_url: str = Field(
        min_length=10,
        max_length=2000,
        description="Direct link to an official government notice or PDF",
    )


class DocumentAnalysisResponse(BaseModel):
    citizen_name: str
    locality: str
    role: str
    document_url: str
    summary: str