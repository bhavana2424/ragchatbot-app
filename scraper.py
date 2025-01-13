import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import time

def scrape_full_depth(url, visited=None, depth=2, delay=1):
    """
    Perform full-depth scraping for a given URL.

    Parameters:
        url (str): The URL to start scraping from.
        visited (set): Set to track visited URLs (prevents duplicate scraping).
        depth (int): Maximum recursion depth.
        delay (int): Delay between requests to avoid server overload.

    Returns:
        dict: A dictionary where keys are URLs and values are the extracted content.
    """
    if visited is None:
        visited = set()

    if depth == 0 or url in visited:
        return {}

    # Mark the URL as visited
    visited.add(url)
    print(f"Scraping: {url} (Depth: {depth})")

    try:
        # Fetch the page content
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            print(f"Failed to fetch {url}: {response.status_code}")
            return {}

        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract visible text from the page
        texts = " ".join(soup.stripped_strings)

        # Find all internal links on the page
        base_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
        links = [
            urljoin(base_url, a['href'])
            for a in soup.find_all('a', href=True)
        ]
        # Filter to include only internal links
        internal_links = [
            link for link in links if link.startswith(base_url)
        ]

        # Recursively scrape each link
        data = {url: texts}
        for link in internal_links:
            data.update(scrape_full_depth(link, visited, depth - 1, delay))

        # Respect rate limiting
        time.sleep(delay)

        return data

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return {}

def save_scraped_data(seed_urls, output_file, depth=2, delay=1):
    """
    Scrape multiple seed URLs and save the data to a file.

    Parameters:
        seed_urls (list): List of URLs to start scraping from.
        output_file (str): Path to save the scraped data as JSON.
        depth (int): Maximum recursion depth.
        delay (int): Delay between requests.
    """
    all_data = {}
    visited = set()

    for url in seed_urls:
        all_data.update(scrape_full_depth(url, visited, depth, delay))

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)

    print(f"Scraping complete. Data saved to '{output_file}'.")


# Example usage
if __name__ == "__main__":
    seed_urls = [
        "https://eumaster4hpc.uni.lu/application/",
        "https://www.fib.upc.edu/en/studies/masters/eumaster4hpc/",
        "https://www.usi.ch/en/education/master/computational-science/structure-and-contents/high-performance-computing",
        "https://masterhpc.polimi.it/",
        "https://www.eumaster4hpc.tf.fau.eu/",
        "https://eurohpc-ju.europa.eu/about/discover-eurohpc-ju_en",
        "https://www.kth.se/en/studies/master/computer-science/msc-computer-science-1.419974",
        "https://www.uni.lu/en/"
    ]
    save_scraped_data(seed_urls, "scraped_data.json", depth=2, delay=2)
