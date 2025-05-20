import os
import requests
from dotenv import load_dotenv

load_dotenv()


def scrape_linkedin_profile(linkedin_profile_url: str, mock: bool = False):
    """scrape information from LinkedIn profiles,
    Manually scrape the information from the LinkedIn profile"""

    if mock:
        linkedin_profile_url = "https://gist.githubusercontent.com/akhilbhat-prog/6689e54e34da3e65e91070b801385a49/raw/e49cac1df5beaec355f832e7de5d5d58e97a95d6/akhil-bhat.json"
        response = requests.get(
            linkedin_profile_url,
            timeout=10,
        )
    else:
        # ENROLL WITH COUPON CODE: EDENMARCO
        # For 20% Discount on all pricing
        api_endpoint = "https://api.scrapin.io/enrichment/profile"
        params = {
            "apikey": os.environ["SCRAPIN_API_KEY"],
            "linkedInUrl": linkedin_profile_url,
        }
        response = requests.get(
            api_endpoint,
            params=params,
            timeout=10,
        )

    data = response.json().get("person")
    data = {
        k: v
        for k, v in data.items()
        if v not in ([], "", "", None) and k not in ["certifications"]
    }

    return data


if __name__ == "__main__":
    print(
        scrape_linkedin_profile(
            linkedin_profile_url="https://www.linkedin.com/in/akhil-bhat/"
        ),
    )
