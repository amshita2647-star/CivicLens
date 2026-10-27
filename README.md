# CivicLens🏛️

**CivicLens** helps Chennai residents understand government announcements and how they may relate to their locality and daily routine. It uses AI to explain information from official Greater Chennai Corporation (GCC) notices.

## Features⚡

- Create an account, log in, and manage your profile.
- Get personalized analysis of recent GCC announcements.
- Analyze supported government document or webpage URLs.
- View explanations with links to the original sources.
- Store relevant account and analysis data using Supabase.

## Tech Stack🔖

- **Frontend:** Streamlit
- **Backend:** Python, FastAPI
- **AI:** Google Gemini API
- **Database:** Supabase
- **Content extraction:** Requests, BeautifulSoup4, pypdf

## How It Works👤

1. The user enters profile details such as locality, role, and daily routine.
2. The Streamlit frontend sends a request to the FastAPI backend.
3. The backend retrieves GCC notices or extracts content from a supported document URL.
4. Gemini analyzes the content and explains its possible relevance to the user.
5. CivicLens displays the analysis and source links.

## Project Structure📂

```text
CivicLens/
├── frontend/
│   ├── app.py
│   └── requirements.txt
├── civiclens-backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── db.py
│   │   └── schemas.py
│   └── requirements.txt
└── README.md
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/amshita2647-star/CivicLens.git
cd CivicLens
```

### 2. Install frontend dependencies

```bash
pip install -r frontend/requirements.txt
```

### 3. Install backend dependencies

```bash
pip install -r civiclens-backend/requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in `civiclens-backend/` and add the required configuration values, including your Gemini API key and Supabase credentials. Set `GEMINI_MODEL` to a supported Gemini model. Keep secrets private and never commit `.env` to GitHub.

### 5. Start the backend

From the project root:

```bash
cd civiclens-backend
uvicorn app.main:app --reload
```

### 6. Start the frontend

Open a second terminal from the project root:

```bash
cd frontend
streamlit run app.py
```

## Links

- **GitHub:** https://github.com/amshita2647-star/CivicLens
- **Backend API:** https://civiclens-6cab.onrender.com
- **API documentation:** https://civiclens-6cab.onrender.com/docs
- **GCC announcements:** https://chennaicorporation.gov.in/gcc/latest-news/

## Important Note

CivicLens provides AI-generated explanations to help users understand public notices. It does not replace official government instructions, and a notice should not be treated as proof of an impact on a specific street or property unless the source confirms it.
