import requests
from bs4 import BeautifulSoup
import time

# Seed URLs — all major direction sub-pages
DIRECTION_URLS = [
    "https://about.gitlab.com/direction/",
    "https://about.gitlab.com/direction/dev/",
    "https://about.gitlab.com/direction/ops/",
    "https://about.gitlab.com/direction/sec/",
    "https://about.gitlab.com/direction/data-science/",
    "https://about.gitlab.com/direction/modelops/",
    "https://about.gitlab.com/direction/anti-abuse/",
    "https://about.gitlab.com/direction/enablement/",
    "https://about.gitlab.com/direction/manage/",
    "https://about.gitlab.com/direction/plan/",
    "https://about.gitlab.com/direction/create/",
    "https://about.gitlab.com/direction/verify/",
    "https://about.gitlab.com/direction/package/",
    "https://about.gitlab.com/direction/deploy/",
    "https://about.gitlab.com/direction/monitor/",
    "https://about.gitlab.com/direction/govern/",
]

visited = set()
all_text = []

def scrape_page(url):
    if url in visited:
        return
    visited.add(url)
    
    try:
        print(f"Scraping: {url}")
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            print(f"  Skipped (status {response.status_code})")
            return
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove nav, footer, scripts, styles
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        
        # Extract clean text
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        page_text = "\n".join(lines)
        
        all_text.append(f"\n\n=== SOURCE: {url} ===\n\n{page_text}")
        
        # Find links to other direction sub-pages
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if href.startswith("/direction/"):
                full_url = "https://about.gitlab.com" + href
                if full_url not in visited:
                    time.sleep(0.5)  # Be polite, don't hammer server
                    scrape_page(full_url)
                    
    except Exception as e:
        print(f"  Error scraping {url}: {e}")

# Start scraping
for url in DIRECTION_URLS:
    scrape_page(url)
    time.sleep(1)

# Save output
output_file = "gitlab_direction.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(all_text))

print(f"\nDone! Scraped {len(visited)} pages.")
print(f"Saved to: {output_file}")