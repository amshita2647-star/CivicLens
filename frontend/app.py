import io
import os
from dotenv import load_dotenv
import tempfile
import time

from bs4 import BeautifulSoup
from google import genai
from google.genai import types
import pdfplumber
import requests
import streamlit as st
BACKEND_URL = os.getenv(
    "CIVICLENS_BACKEND_URL",
    "http://127.0.0.1:8000"
)


def check_locality_with_backend(
    name: str,
    locality: str,
    role: str,
    description: str,
) -> dict:
    """Send the citizen profile to FastAPI for locality analysis."""

    payload = {
        "user_profile": {
            "name": name,
            "locality": locality,
            "role": role,
            "user_description": description,
        },
        "limit": 5,
    }

    response = requests.post(
        f"{BACKEND_URL}/api/v1/locality-check",
        json=payload,
        timeout=240,
    )

    response.raise_for_status()
    return response.json()


# Page Configuration
st.set_page_config(page_title="CivicLens Chennai",
                   page_icon="🏛️", layout="centered")

# Initialize Session State
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_data" not in st.session_state:
    st.session_state.user_data = {
        "email": "",
        "name": "",
        "locality": "Adyar (Zone 13)",
        "role": "Renter",
        "description": "I live on a residential street and commute by bike daily.",
    }

# Initialize Gemini Client (Clean & Secure)
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")


def fetch_latest_gcc_pdf_links() -> list:
    """Scrapes news links and direct PDF URLs from the GCC portal."""
    url = "https://chennaicorporation.gov.in/gcc/latest-news/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        pdf_notices = []
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            title = a_tag.get_text(strip=True)

            if href.endswith(".pdf") or "document" in href.lower():
                full_url = (
                    href
                    if href.startswith("http")
                    else f"https://chennaicorporation.gov.in/gcc/latest-news/{href}"
                )
                if title and len(title) > 5:
                    pdf_notices.append({"title": title, "url": full_url})

        return pdf_notices[:5]
    except Exception as e:
        st.error(f"Failed to fetch live GCC notices: {e}")
        return []


def fetch_pdf_bytes(pdf_url: str) -> bytes:
    """Downloads PDF bytes from government portals with browser headers and SSL handling."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/pdf,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    try:
        response = requests.get(pdf_url, headers=headers, timeout=20)
        response.raise_for_status()
        return response.content
    except requests.exceptions.SSLError:
        # Fallback for state portals with misconfigured SSL certificates
        response = requests.get(pdf_url, headers=headers,
                                timeout=20, verify=False)
        response.raise_for_status()
        return response.content
    except Exception as e:
        raise Exception(f"Failed to download PDF from URL: {str(e)}")


def extract_text_locally(pdf_bytes: bytes) -> str:
    """Attempts local text extraction using pdfplumber."""
    try:
        pdf_file = io.BytesIO(pdf_bytes)
        extracted_text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
        return extracted_text.strip()
    except Exception:
        return ""


def analyze_policy_impact(pdf_bytes: bytes, name: str, locality: str, role: str, user_des: str) -> str:
    """Passes document text (or raw PDF file to Gemini if scanned) for analysis."""

    system_instruction = (
        "You are CivicLens Chennai. Explain official government documents in simple, everyday language.\n"
        "Rules:\n"
        "1. Keep your entire answer very short (under 150 words).\n"
        "2. Address the citizen directly by name.\n"
        "3. Use simple bullet points to explain: What this document is, How it affects them, and What action they need to take.\n"
        "4. Avoid complex legal jargon or long paragraphs."
    )

    # Use the model version recommended by your API endpoint
    MODEL_ID = "gemini-3.6-flash"

    # 1. Try local text extraction
    extracted_text = extract_text_locally(pdf_bytes)

    # 2. If text was successfully extracted, send text prompt
    if len(extracted_text) > 100:
        user_prompt = f"""
        CITIZEN PROFILE:
        - Name: {name}
        - Locality/Zone: {locality}
        - Role: {role}
        - Daily Context / Routine: {user_des}

        GOVERNMENT DOCUMENT TEXT:
        {extracted_text[:10000]}
        """

        response = client.models.generate_content(
            model=MODEL_ID,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
            ),
        )
        return response.text if response.text else "Unable to parse text content from this document."

    # 3. Fallback for scanned PDFs: Upload raw PDF directly to Gemini Multimodal
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_bytes)
            tmp_path = tmp_file.name

        try:
            uploaded_file = client.files.upload(file=tmp_path)

            # Ensure file processing is complete before calling generate_content
            while uploaded_file.state.name == "PROCESSING":
                time.sleep(2)
                uploaded_file = client.files.get(name=uploaded_file.name)

            if uploaded_file.state.name == "FAILED":
                raise Exception("File processing failed on Gemini server.")

            user_prompt = f"""
            CITIZEN PROFILE:
            - Name: {name}
            - Locality/Zone: {locality}
            - Role: {role}
            - Daily Context / Routine: {user_des}

            Analyze the attached PDF document and explain its impact on the citizen.
            """

            response = client.models.generate_content(
                model=MODEL_ID,
                contents=[uploaded_file, user_prompt],
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2,
                ),
            )

            if response.text:
                return response.text
            elif response.candidates and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text
            else:
                return "The document was processed, but no text response was generated."

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


def analyze_policy_impact_multilingual(pdf_bytes: bytes, name: str, locality: str, role: str, user_des: str, output_lang: str = "English") -> str:
    """Passes English or Tamil PDF (scanned or text) to Gemini with language-aware instructions."""

    # Explicit Multilingual System Instructions
    system_instruction = (
        f"You are CivicLens Chennai, an AI civic assistant capable of understanding both Tamil and English official documents.\n"
        f"Rules:\n"
        f"1. Read and analyze the document content (even if it is written or scanned in Tamil script).\n"
        f"2. Generate the entire summary strictly in **{output_lang}**.\n"
        f"3. Address the citizen directly by name ({name}).\n"
        f"4. Keep the summary under 150 words using simple, non-legal terminology.\n"
        f"5. Break down: What the update/G.O. is about, how it affects a {role} in {locality}, and any required action items or deadlines."
    )

    MODEL_ID = "gemini-2.5-flash"

    # 1. Try local text extraction first
    extracted_text = extract_text_locally(pdf_bytes)

    # 2. If text was extracted successfully (>100 chars), send prompt
    if len(extracted_text) > 100:
        user_prompt = f"""
        CITIZEN PROFILE:
        - Name: {name}
        - Locality: {locality}
        - Role: {role}
        - Daily Context / Routine: {user_des}

        DOCUMENT CONTENT (May be Tamil or English):
        {extracted_text[:6000]}
        """

        response = client.models.generate_content(
            model=MODEL_ID,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
            ),
        )
        return response.text if response.text else "Unable to generate summary."

    # 3. Fallback for scanned Tamil/English PDFs (Multimodal Upload)
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_bytes)
            tmp_path = tmp_file.name

        try:
            uploaded_file = client.files.upload(file=tmp_path)

            while uploaded_file.state.name == "PROCESSING":
                time.sleep(2)
                uploaded_file = client.files.get(name=uploaded_file.name)

            if uploaded_file.state.name == "FAILED":
                raise Exception("File processing failed on Gemini server.")

            user_prompt = f"""
            CITIZEN PROFILE:
            - Name: {name}
            - Locality: {locality}
            - Role: {role}
            - Daily Context / Routine: {user_des}

            Analyze this attached document (scanned Tamil or English image PDF) and explain its impact following the system instructions.
            """

            response = client.models.generate_content(
                model=MODEL_ID,
                contents=[uploaded_file, user_prompt],
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2,
                ),
            )

            if response.text:
                return response.text
            elif response.candidates and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text
            else:
                return "The document was processed, but no response was generated."

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


def summarize_locality_updates(notices: list, name: str, locality: str, role: str, user_des: str) -> str:
    """Uses Gemini to filter and summarize updates relevant to the citizen's zone and role."""

    system_instruction = (
        "You are CivicLens Chennai. Filter official city announcements and explain updates relevant "
        "to a specific citizen's locality and role in simple, direct language under 150 words.\n"
        "Format:\n"
        "• **Relevant Updates for Your Area**: Bullet points of key notices affecting their zone or city-wide policies.\n"
        "• **What You Need to Do**: Clear action items or deadlines (if any)."
    )

    formatted_notices = "\n- ".join(notices)

    user_prompt = f"""
    CITIZEN PROFILE:
    - Name: {name}
    - Locality: {locality}
    - Role: {role}
    - Daily Context / Routine: {user_des}

    RECENT GCC ANNOUNCEMENTS:
    - {formatted_notices}
    """

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
        ),
    )
    return response.text if response.text else "No specific updates found for your locality today."


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

    auth_mode = st.radio("Choose Action:", [
        "Log In", "Sign Up / Create Account"
    ], horizontal=True)

    with st.form(key="auth_form"):
        email = st.text_input("Email Address:", placeholder="name@example.com")
        password = st.text_input("Password:", type="password")

        if auth_mode == "Sign Up / Create Account":
            name = st.text_input(
                "Full Name:", placeholder="e.g., Sundar Raman")
            locality = st.selectbox("Locality / Zone:", [
                "Adyar (Zone 13)", "Velachery (Zone 13)", "T. Nagar (Zone 10)",
                "Anna Nagar (Zone 8)", "Mylapore (Zone 9)", "Kodambakkam (Zone 10)",
                "Guindy (Zone 13)", "Royapuram (Zone 5)", "Other / General Chennai"
            ])
            role = st.selectbox("Primary Role:", [
                "Renter", "Homeowner", "Small Business Owner", "Student", "Parent", "Senior Citizen"
            ])
            # ADDED: User description / routine context field
            user_des = st.text_area(
                "Daily Context / Routine (Optional):",
                placeholder="e.g., Commute daily via OMR, have 2 school-going children, run a small tea shop",
                help="Helps Gemini personalize policy impact summaries specifically for your routine."
            )

        submit_auth = st.form_submit_button(label="Submit")

    if submit_auth:
        if not email or not password:
            st.error("Please fill in both email and password fields.")
        else:
            st.session_state.logged_in = True
            st.session_state.user_data["email"] = email

            if auth_mode == "Sign Up / Create Account":
                st.session_state.user_data["name"] = name if name else "Citizen"
                st.session_state.user_data["locality"] = locality
                st.session_state.user_data["role"] = role
                # ADDED: Save user_des in session state
                st.session_state.user_data["description"] = user_des if user_des else "Standard daily routine"
            else:
                st.session_state.user_data["name"] = email.split(
                    "@")[0].capitalize()
                # ADDED: Default description if logging into existing account
                if "description" not in st.session_state.user_data or not st.session_state.user_data["description"]:
                    st.session_state.user_data["description"] = "Standard daily routine"

            st.success(f"Welcome back, {st.session_state.user_data['name']}!")
            st.rerun()
# ==========================================
# SCREEN 2: MAIN APP DASHBOARD
# ==========================================

else:
    user = st.session_state.user_data

    # Top Bar: User Greeting and Logout
    col_user, col_logout = st.columns([3, 1])

    with col_user:
        st.write(
            f"Logged in as: **{user['name']}** (`{user['email']}`)"
        )

    with col_logout:
        if st.button("Log Out"):
            st.session_state.logged_in = False
            st.rerun()

    # Main Navigation: No Nested Tabs
    tab1, tab2 = st.tabs(
        ["🏛️ CivicLens Dashboard", "👤 My Profile"]
    )

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
                placeholder="https://example.gov.in/notice.pdf"
            )

            analyze_pdf_button = st.form_submit_button(
                "Analyze Document 🚀"
            )

        if analyze_pdf_button:
            if not document_url.strip():
                st.warning("Please enter a PDF URL.")

            else:
                try:
                    with st.spinner(
                        "Downloading government document..."
                    ):
                        pdf_bytes = fetch_pdf_bytes(
                            document_url.strip()
                        )

                    with st.spinner(
                        "Analyzing document with Gemini AI..."
                    ):
                        summary_result = analyze_policy_impact(
                            pdf_bytes=pdf_bytes,
                            name=user["name"],
                            locality=user["locality"],
                            role=user["role"],
                            user_des=user.get(
                                "description",
                                "Standard daily routine"
                            )
                        )

                    st.markdown("---")
                    st.subheader("📋 Personalized Policy Impact")
                    st.success(summary_result)

                except Exception as e:
                    st.error(
                        f"Error analyzing the document: {e}"
                    )

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
                    result = check_locality_with_backend(
                        name=user["name"],
                        locality=user["locality"],
                        role=user["role"],
                        description=user.get(
                            "description",
                            "Standard daily routine"
                        ),
                    )

                st.markdown("---")
                st.subheader(f"Updates for {result['locality']}")
                st.info(result["summary"])

                for item in result.get("impact_items", []):
                    st.markdown(f"### {item['title']}")
                    st.write(item["impact_summary"])

                    source_url = item.get("source_url")
                    if source_url:
                        st.markdown(
                            f"[View official source]({source_url})"
                        )

                st.caption(
                    f"Announcements analyzed: "
                    f"{result.get('total_updates_found', 0)}"
                )

            except requests.exceptions.Timeout:
                st.error(
                    "The backend took too long to respond. "
                    "Please try again."
                )

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to the CivicLens backend. "
                    "Make sure FastAPI is running at "
                    f"{BACKEND_URL}."
                )

            except requests.exceptions.HTTPError as e:
                st.error(
                    f"The backend returned an HTTP error: {e}"
                )

            except (KeyError, ValueError) as e:
                st.error(
                    f"The backend response was not in the expected "
                    f"format: {e}"
                )

            except Exception as e:
                st.error(f"Unable to load locality updates: {e}")
    # ==========================================
    # TAB 2: MY PROFILE
    # ==========================================

    with tab2:
        st.subheader("My Profile")

        st.write(
            "Update your details to personalize future policy analyses."
        )

        locality_list = [
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

        role_list = [
            "Renter",
            "Homeowner",
            "Small Business Owner",
            "Student",
            "Parent",
            "Senior Citizen",
        ]

        # Profile form stays entirely inside My Profile
        with st.form(key="profile_update_form"):

            updated_name = st.text_input(
                "Full Name:",
                value=user.get("name", "")
            )

            current_loc_idx = (
                locality_list.index(user["locality"])
                if user.get("locality") in locality_list
                else 0
            )

            updated_locality = st.selectbox(
                "Locality / Zone:",
                locality_list,
                index=current_loc_idx
            )

            current_role_idx = (
                role_list.index(user["role"])
                if user.get("role") in role_list
                else 0
            )

            updated_role = st.selectbox(
                "Primary Role:",
                role_list,
                index=current_role_idx
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
                )
            )

            save_profile = st.form_submit_button(
                "Update Profile"
            )

        if save_profile:
            if not updated_name.strip():
                st.error("Please enter your name.")

            else:
                user["name"] = updated_name.strip()
                user["locality"] = updated_locality
                user["role"] = updated_role
                user["description"] = updated_des.strip()

                st.session_state.user_data = user

                st.success(
                    "Profile updated successfully!"
                )

                st.rerun()
