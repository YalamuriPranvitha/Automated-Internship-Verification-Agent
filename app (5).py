import os
import re

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from web_verification import (
    verify_company,
    verify_company_website,
    verify_internship
)


# ============================================================
# API KEYS
# ============================================================

try:
    from google.colab import userdata

    GOOGLE_API_KEY = userdata.get("GOOGLE_API_KEY")
    SEARCH_API_KEY = userdata.get("SEARCH_API_KEY")

    if SEARCH_API_KEY:
        os.environ["SEARCH_API_KEY"] = SEARCH_API_KEY

except Exception:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")


if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not configured.")

if not SEARCH_API_KEY:
    print("WARNING: SEARCH_API_KEY is not configured.")


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="AI Internship Authenticity Checker",
    description="Agentic RAG system for internship verification",
    version="1.0"
)


# ============================================================
# GEMMA
# ============================================================

MODEL_NAME = os.getenv(
    "GEMMA_MODEL",
    "gemma-3-27b-it"
)

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)


# ============================================================
# KNOWLEDGE BASE
# ============================================================

knowledge_base = """

INTERNSHIP VERIFICATION GUIDELINES

Registration fees:
Students should carefully verify internship opportunities that request
registration fees, processing fees, security deposits, training fees,
or other payments.

Guaranteed employment:
Claims such as guaranteed placement, guaranteed job, guaranteed salary,
or 100 percent employment should be independently verified.

Company identity:
Students should verify the organization through its official website,
official careers page, company domain, and official communication
channels.

Email:
Students should compare the contact email domain with the company's
official domain.

Urgency:
Pressure to make immediate payments or provide documents immediately
should be treated as a warning sign requiring verification.

Missing information:
Important internship information includes company name, role,
duration, responsibilities, location, stipend, selection process,
contact information, and terms.

Sensitive information:
Students should avoid providing passwords, OTPs, PINs, CVV numbers,
or unnecessary sensitive information.

Official internship listing:
If an internship is claimed to be offered by a company, students should
check whether the role appears on an independently verified official
careers page.

Important limitation:
An AI system cannot definitively prove that an internship is fraudulent
or legitimate based only on available information.

The system should provide an evidence-based risk assessment.
"""


documents = [
    Document(
        page_content=knowledge_base,
        metadata={
            "source": "internship_verification_guidelines"
        }
    )
]


# ============================================================
# TEXT SPLITTER
# ============================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)


# ============================================================
# EMBEDDINGS
# ============================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)


# ============================================================
# FAISS VECTOR DATABASE
# ============================================================

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)


# ============================================================
# RAG TOOL
# ============================================================

@tool
def search_guidelines(query: str) -> str:
    """
    Search the internship verification knowledge base.
    """

    docs = retriever.invoke(query)

    if not docs:
        return "No relevant guidelines found."

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# ============================================================
# RISK ANALYSIS TOOL
# ============================================================

@tool
def analyze_risk_indicators(text: str) -> str:
    """
    Detect common warning indicators in an internship offer.
    """

    text = text.lower()

    indicators = []

    patterns = {
        "Registration fee": [
            "registration fee",
            "registration fees"
        ],

        "Payment request": [
            "pay",
            "payment",
            "deposit",
            "processing fee",
            "security deposit",
            "training fee"
        ],

        "Guaranteed employment": [
            "guaranteed job",
            "guaranteed placement",
            "100% placement",
            "guaranteed employment"
        ],

        "Urgency": [
            "pay immediately",
            "act now",
            "urgent",
            "limited seats",
            "today only"
        ],

        "Sensitive information": [
            "otp",
            "password",
            "upi pin",
            "bank password",
            "cvv"
        ],

        "Unrealistic claims": [
            "no interview required",
            "guaranteed salary",
            "earn millions"
        ]
    }

    for category, keywords in patterns.items():

        found = []

        for keyword in keywords:
            if keyword in text:
                found.append(keyword)

        if found:
            indicators.append(
                f"{category}: {', '.join(found)}"
            )

    if not indicators:
        return "No common warning indicators detected."

    return "\n".join(indicators)


# ============================================================
# EMAIL ANALYSIS TOOL
# ============================================================

@tool
def analyze_email(text: str) -> str:
    """
    Detect email addresses and identify generic email providers.
    """

    emails = re.findall(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        text
    )

    if not emails:
        return "No email address found."

    generic_domains = {
        "gmail.com",
        "yahoo.com",
        "outlook.com",
        "hotmail.com",
        "rediffmail.com"
    }

    results = []

    for email in emails:

        domain = email.split("@")[-1].lower()

        if domain in generic_domains:
            results.append(
                f"{email}: generic email provider."
            )
        else:
            results.append(
                f"{email}: custom domain detected."
            )

    return "\n".join(results)


# ============================================================
# WEB VERIFICATION TOOL
# ============================================================

@tool
def verify_internship_online(
    company_name: str,
    internship_title: str
) -> str:
    """
    Search the live web for the company and internship.
    """

    result = verify_internship(
        company_name,
        internship_title
    )

    return str(result)


# ============================================================
# COMPANY WEBSITE TOOL
# ============================================================

@tool
def search_company_online(company_name: str) -> str:
    """
    Search for the company's official website.
    """

    result = verify_company_website(
        company_name
    )

    return str(result)


# ============================================================
# AGENT PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_template(
"""
You are an AI Internship Verification Agent.

Your job is to analyze internship opportunities for engineering
students.

You have access to:
1. RAG verification guidelines
2. Risk analysis
3. Email analysis
4. Live web verification

Use the available evidence carefully.

IMPORTANT:
Never claim that an internship is definitely fraudulent or definitely
legitimate based only on this analysis.

You must distinguish between:

VERIFIED FACT:
Information directly supported by a retrieved source.

WARNING:
An observation that requires additional verification.

INFERENCE:
Your AI-based interpretation of the available evidence.

Internship submitted by student:

{internship}

Company:

{company}

Internship title:

{title}

RAG guidelines:

{guidelines}

Risk analysis:

{risk}

Email analysis:

{email}

Live web verification:

{web_results}

Create the final report using this format:

==================================================
AI INTERNSHIP VERIFICATION REPORT
==================================================

Company:
...

Internship:
...

Overall Risk:
LOW / MEDIUM / HIGH / INSUFFICIENT INFORMATION

--------------------------------------------------
1. WEB VERIFICATION
--------------------------------------------------

Official company information:
...

Official internship listing:
...

Company/internship match:
...

--------------------------------------------------
2. WARNING INDICATORS
--------------------------------------------------

- ...
- ...

--------------------------------------------------
3. POSITIVE INDICATORS
--------------------------------------------------

- ...
- ...

--------------------------------------------------
4. INFORMATION MISSING
--------------------------------------------------

- ...
- ...

--------------------------------------------------
5. EVIDENCE SUMMARY
--------------------------------------------------

VERIFIED:
- ...

WARNING:
- ...

UNVERIFIED:
- ...

--------------------------------------------------
6. RECOMMENDED ACTION
--------------------------------------------------

...

--------------------------------------------------
DISCLAIMER
--------------------------------------------------

This is an AI-assisted informational risk assessment.
It cannot definitively prove that an internship is genuine or
fraudulent. Students should independently verify the organization
through official communication channels before making payments or
sharing sensitive information.
"""
)


# ============================================================
# REQUEST MODEL
# ============================================================

class InternshipRequest(BaseModel):

    company_name: str

    internship_title: str

    internship_description: str


# ============================================================
# MAIN ANALYSIS FUNCTION
# ============================================================

def analyze_internship(
    company_name,
    internship_title,
    internship_description
):

    # ----------------------------------------
    # RAG
    # ----------------------------------------

    rag_query = (
        company_name
        + " "
        + internship_title
        + " "
        + internship_description
    )

    docs = retriever.invoke(rag_query)

    guidelines = "\n\n".join(
        doc.page_content
        for doc in docs
    )


    # ----------------------------------------
    # RISK TOOL
    # ----------------------------------------

    risk = analyze_risk_indicators.invoke(
        internship_description
    )


    # ----------------------------------------
    # EMAIL TOOL
    # ----------------------------------------

    email = analyze_email.invoke(
        internship_description
    )


    # ----------------------------------------
    # WEB VERIFICATION
    # ----------------------------------------

    web_results = verify_internship_online.invoke(
        {
            "company_name": company_name,
            "internship_title": internship_title
        }
    )


    # ----------------------------------------
    # GEMMA
    # ----------------------------------------

    messages = prompt.format_messages(

        internship=internship_description,

        company=company_name,

        title=internship_title,

        guidelines=guidelines,

        risk=risk,

        email=email,

        web_results=web_results
    )


    response = llm.invoke(messages)

    return response.content


# ============================================================
# API ROUTES
# ============================================================

@app.get("/")
def home():

    return {
        "message": "AI Internship Authenticity Checker is running",

        "model": MODEL_NAME,

        "features": [
            "FastAPI",
            "Gemma",
            "LangChain",
            "RAG",
            "FAISS",
            "Web Verification",
            "Risk Analysis",
            "Email Analysis"
        ]
    }


@app.post("/check-internship")
def check_internship(
    request: InternshipRequest
):

    result = analyze_internship(

        request.company_name,

        request.internship_title,

        request.internship_description
    )

    return {
        "company": request.company_name,

        "internship": request.internship_title,

        "assessment": result
    }
