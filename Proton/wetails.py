import requests
from bs4 import BeautifulSoup

def extract_website_info(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
    except Exception as e:
        print(f"Error fetching the URL: {e}")
        return None

    # Parse the HTML content with BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract the title of the page
    title = soup.title.string if soup.title else "No title found"

    # Extract the meta description (if available)
    meta_description = soup.find("meta", attrs={"name": "description"})
    description = meta_description['content'] if meta_description and meta_description.has_attr('content') else "No description available"

    # Extract text from all paragraph elements
    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
    
    return {
        "title": title,
        "description": description,
        "paragraphs": paragraphs,
    }

def main():
    url = input("Enter a website URL: ")
    info = extract_website_info(url)
    if info:
        print("\nWebsite Information:")
        print("Title:", info["title"])
        print("Meta Description:", info["description"])
        print("\nExtracted Paragraphs:\n")
        for p in info["paragraphs"]:
            print(p)
            print("-" * 80)

if __name__ == "__main__":
    main()
