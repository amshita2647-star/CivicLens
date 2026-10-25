import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv(
    "CIVICLENS_BACKEND_URL",
    "https://civiclens-6cab.onrender.com",
).rstrip("/")

REQUEST_TIMEOUT = 120  # seconds; allows for a sleeping free-tier backend

LOCALITY_LIST = [
    "Adyar (Zone 13)",
    "Velachery (Zone 13)",
    "T. Nagar (Zone 10)",
    "Anna Nagar (Zone 8)",
    "Mylapore (Zone 9)",
    "Kodambakkam (Zone 10)",
    "Guindy (Zone 13)",
    "Royapuram (Zone 5)",
    "Other / General Chennai",
]

ROLE_LIST = [
    "Renter",
    "Homeowner",
    "Small Business Owner",
    "Student",
    "Parent",
    "Senior Citizen",
]


# ==========================================
# BACKEND CALLS (all logic happens in FastAPI)
# ==========================================

def call_backend(
    path: str,
    payload: dict,
    token: str = None,
    method: str = "POST",
) -> dict:
    """Call the FastAPI backend and return its JSON response."""

    headers = {"Authorization": f"Bearer {token}"} if token else {}

    response = requests.request(
        method,
        f"{BACKEND_URL}{path}",
        json=payload,
        headers=headers,
        timeout=REQUEST_TIMEOUT,
    )

    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text

        if isinstance(detail, list):  # FastAPI validation errors
            detail = "Some of the details entered are invalid."

        raise RuntimeError(detail or f"Backend error {response.status_code}")

    return response.json()


def build_profile(user: dict) -> dict:
    return {
        "name": user["name"],
        "locality": user["locality"],
        "role": user["role"],
        "user_description": user.get("description")
        or "Standard daily routine",
    }


def check_locality_with_backend(user: dict) -> dict:
    return call_backend(
        "/api/v1/locality-check",
        {"user_profile": build_profile(user), "limit": 5},
        token=st.session_state.get("token"),
    )


def analyze_document_with_backend(user: dict, document_url: str) -> dict:
    return call_backend(
        "/api/v1/analyze-document",
        {
            "user_profile": build_profile(user),
            "document_url": document_url,
        },
    )


def apply_account(account: dict) -> None:
    """Store the account returned by the backend in the session."""
    st.session_state.logged_in = True
    st.session_state.token = account.get("token") or st.session_state.get(
        "token"
    )
    st.session_state.user_data = {
        "email": account["email"],
        "name": account["name"],
        "locality": account["locality"],
        "role": account["role"],
        "description": account.get("user_description")
        or "Standard daily routine",
    }


def show_backend_error(error: Exception) -> None:
    """Show a friendly message for any backend failure."""

    if isinstance(error, requests.exceptions.Timeout):
        st.error("The backend took too long to respond. Please try again.")
    elif isinstance(error, requests.exceptions.ConnectionError):
        st.error(
            "Cannot connect to the CivicLens backend. "
            f"Make sure FastAPI is running at {BACKEND_URL}."
        )
    else:
        st.error(str(error))


# Page Configuration
st.set_page_config(
    page_title="CivicLens Chennai",
    page_icon="🏛️",
    layout="centered",
)

# Initialize Session State
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "token" not in st.session_state:
    st.session_state.token = None
if "user_data" not in st.session_state:
    st.session_state.user_data = {
        "email": "",
        "name": "",
        "locality": "Adyar (Zone 13)",
        "role": "Renter",
        "description": "I live on a residential street and commute by bike daily.",
    }

# Display Header Emblem
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    try:
        st.image("CivicLens_2.webp", use_container_width=True)
    except Exception:
        pass

st.title("CivicLens 🏛️")
st.markdown("---")

# ==========================================
# SCREEN 1: LOGIN / SIGNUP PAGE (If not logged in)
# ==========================================
if not st.session_state.logged_in:
    st.subheader("Welcome! Please log in to continue.")

    auth_mode = st.radio(
        "Choose Action:",
        ["Log In", "Sign Up / Create Account"],
        horizontal=True,
    )

    with st.form(key="auth_form"):
        email = st.text_input("Email Address:", placeholder="name@example.com")
        password = st.text_input("Password:", type="password")

        if auth_mode == "Sign Up / Create Account":
            name = st.text_input(
                "Full Name:", placeholder="e.g., Sundar Raman"
            )
            locality = st.selectbox("Locality / Zone:", LOCALITY_LIST)
            role = st.selectbox("Primary Role:", ROLE_LIST)
            user_des = st.text_area(
                "Daily Context / Routine (Optional):",
                placeholder=(
                    "e.g., Commute daily via OMR, have 2 school-going "
                    "children, run a small tea shop"
                ),
                help=(
                    "Helps Gemini personalize policy impact summaries "
                    "specifically for your routine."
                ),
            )

        submit_auth = st.form_submit_button(label="Submit")

    if submit_auth:
        is_signup = auth_mode == "Sign Up / Create Account"

        if not email.strip() or not password:
            st.error("Please fill in both email and password fields.")
        elif is_signup and len(password) < 8:
            st.error("Password must be at least 8 characters.")
        else:
            try:
                with st.spinner("Please wait..."):
                    if is_signup:
                        account = call_backend(
                            "/api/v1/auth/signup",
                            {
                                "email": email.strip(),
                                "password": password,
                                "name": name.strip() or "Citizen",
                                "locality": locality,
                                "role": role,
                                "user_description": user_des.strip()
                                or "Standard daily routine",
                            },
                        )
                    else:
                        account = call_backend(
                            "/api/v1/auth/login",
                            {
                                "email": email.strip(),
                                "password": password,
                            },
                        )

                apply_account(account)
                st.rerun()

            except Exception as e:
                show_backend_error(e)

# ==========================================
# SCREEN 2: MAIN APP DASHBOARD
# ==========================================

else:
    user = st.session_state.user_data

    # Top Bar: User Greeting and Logout
    col_user, col_logout = st.columns([3, 1])

    with col_user:
        st.write(f"Logged in as: **{user['name']}** (`{user['email']}`)")

    with col_logout:
        if st.button("Log Out"):
            st.session_state.logged_in = False
            st.session_state.token = None
            st.rerun()

    tab1, tab2 = st.tabs(["🏛️ CivicLens Dashboard", "👤 My Profile"])

    # ==========================================
    # TAB 1: CIVICLENS DASHBOARD
    # ==========================================

    with tab1:
        st.subheader("Your Civic Policy Dashboard")

        st.caption(
            f"Active Persona: **{user['role']}** | "
            f"Locality: **{user['locality']}**"
        )

        st.markdown("---")

        # --------------------------------------
        # SECTION 1: GOVERNMENT DOCUMENT ANALYSIS
        # --------------------------------------

        st.subheader("📄 Analyze a Government Document")

        st.caption(
            "Paste a direct link to an official government circular or PDF."
        )

        with st.form(key="pdf_analysis_form"):
            document_url = st.text_input(
                "Government Notice / PDF URL:",
                placeholder="https://example.gov.in/notice.pdf",
            )

            analyze_pdf_button = st.form_submit_button("Analyze Document 🚀")

        if analyze_pdf_button:
            if not document_url.strip():
                st.warning("Please enter a PDF URL.")
            else:
                try:
                    with st.spinner("Analyzing the document..."):
                        doc_result = analyze_document_with_backend(
                            user, document_url.strip()
                        )

                    st.markdown("---")
                    st.subheader("📋 Personalized Policy Impact")
                    st.success(doc_result["summary"])

                except Exception as e:
                    show_backend_error(e)

        # --------------------------------------
        # SECTION 2: LOCALITY UPDATES
        # --------------------------------------

        st.markdown("---")
        st.subheader("📍 Government Updates Near You")

        st.caption(
            f"Find recent GCC announcements relevant to "
            f"**{user['locality']}**."
        )

        with st.form(key="locality_check_form"):
            check_locality_button = st.form_submit_button(
                "Check My Locality 🔍"
            )

        if check_locality_button:
            try:
                with st.spinner(
                    f"Analyzing GCC updates for {user['locality']}..."
                ):
                    result = check_locality_with_backend(user)

                st.markdown("---")
                st.subheader(f"Updates for {result['locality']}")
                st.info(result["summary"])

                for item in result.get("impact_items", []):
                    st.markdown(f"### {item['title']}")
                    st.write(item["impact_summary"])

                    source_url = item.get("source_url")
                    if source_url:
                        st.markdown(f"[View official source]({source_url})")

                st.caption(
                    f"Announcements analyzed: "
                    f"{result.get('total_updates_found', 0)}"
                )

            except (KeyError, ValueError) as e:
                st.error(
                    f"The backend response was not in the expected format: {e}"
                )

            except Exception as e:
                show_backend_error(e)

    # ==========================================
    # TAB 2: MY PROFILE
    # ==========================================

    with tab2:
        st.subheader("My Profile")

        st.write("Update your details to personalize future policy analyses.")

        if st.session_state.pop("profile_saved", False):
            st.success("Profile updated successfully!")

        with st.form(key="profile_update_form"):
            updated_name = st.text_input(
                "Full Name:", value=user.get("name", "")
            )

            current_loc_idx = (
                LOCALITY_LIST.index(user["locality"])
                if user.get("locality") in LOCALITY_LIST
                else 0
            )

            updated_locality = st.selectbox(
                "Locality / Zone:", LOCALITY_LIST, index=current_loc_idx
            )

            current_role_idx = (
                ROLE_LIST.index(user["role"])
                if user.get("role") in ROLE_LIST
                else 0
            )

            updated_role = st.selectbox(
                "Primary Role:", ROLE_LIST, index=current_role_idx
            )

            updated_des = st.text_area(
                "Daily Context / Routine (Optional):",
                value=user.get("description", ""),
                placeholder=(
                    "e.g., I commute daily via OMR, "
                    "have school-going children, or run a small shop."
                ),
                help=(
                    "Describe your daily routine so CivicLens can "
                    "explain how government policies may affect you."
                ),
            )

            save_profile = st.form_submit_button("Update Profile")

        if save_profile:
            if not updated_name.strip():
                st.error("Please enter your name.")
            else:
                try:
                    with st.spinner("Saving your profile..."):
                        account = call_backend(
                            "/api/v1/profile",
                            {
                                "name": updated_name.strip(),
                                "locality": updated_locality,
                                "role": updated_role,
                                "user_description": updated_des.strip()
                                or "Standard daily routine",
                            },
                            token=st.session_state.get("token"),
                            method="PUT",
                        )

                    apply_account(account)
                    st.session_state.profile_saved = True
                    st.rerun()

                except Exception as e:
                    show_backend_error(e)