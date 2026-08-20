import os
import requests


# ============================================================
# TAVILY SEARCH API KEY
# ============================================================

SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")


# ============================================================
# WEB SEARCH
# ============================================================

def search_web(query):

    if not SEARCH_API_KEY:

        return {
            "status": "error",
            "message":
            "SEARCH_API_KEY is not configured."
        }


    url = "https://api.tavily.com/search"


    payload = {

        "api_key": SEARCH_API_KEY,

        "query": query,

        "search_depth": "advanced",

        "max_results": 5,

        "include_answer": True
    }


    try:

        response = requests.post(

            url,

            json=payload,

            timeout=30
        )


        response.raise_for_status()


        data = response.json()


        results = []


        for item in data.get("results", []):

            results.append({

                "title":
                item.get("title", ""),

                "url":
                item.get("url", ""),

                "content":
                item.get("content", "")
            })


        return {

            "status": "success",

            "answer":
            data.get("answer", ""),

            "results":
            results
        }


    except Exception as e:

        return {

            "status": "error",

            "message": str(e)
        }


# ============================================================
# COMPANY SEARCH
# ============================================================

def verify_company(company_name):

    query = (
        f'"{company_name}" '
        f'official company website'
    )

    return search_web(query)


# ============================================================
# COMPANY WEBSITE SEARCH
# ============================================================

def verify_company_website(company_name):

    query = (

        f'"{company_name}" '

        f'official website careers contact'
    )

    return search_web(query)


# ============================================================
# INTERNSHIP SEARCH
# ============================================================

def verify_internship(
    company_name,
    internship_title
):

    query = (

        f'"{company_name}" '

        f'"{internship_title}" '

        f'internship careers official'
    )

    return search_web(query)


# ============================================================
# COMBINED VERIFICATION
# ============================================================

def verify_all(
    company_name,
    internship_title
):

    company_results = verify_company(
        company_name
    )


    website_results = verify_company_website(
        company_name
    )


    internship_results = verify_internship(

        company_name,

        internship_title
    )


    return {

        "company_search":
        company_results,

        "website_search":
        website_results,

        "internship_search":
        internship_results
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    result = verify_internship(

        "Microsoft",

        "Software Engineering Intern"
    )


    print(result)
