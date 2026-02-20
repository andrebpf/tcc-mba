import requests
import pandas as pd
from datetime import datetime
import time
import html
from typing import List, Dict, Any, Optional, Tuple

from src.utils.logger import setup_logger

logger = setup_logger("mba_tcc.scraper")

def fetch_posts_from_api(term: str, page: int = 1) -> Tuple[List[Dict[str, Any]], int]:
    """
    Fetches posts from the WordPress API.
    Returns a tuple containing the list of posts and the total number of pages.
    """
    base_url = "https://www.infomoney.com.br/wp-json/wp/v2/posts"
    params = {
        "search": term,
        "page": page,
        "per_page": 100,
        "orderby": "date",
        "order": "desc"
    }
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        response = requests.get(base_url, params=params, headers=headers, timeout=15)
        
        if response.status_code == 400:
            # Bad Request usually means invalid page (e.g., page=9999)
            logger.info("Pagination limit reached (400 Bad Request).")
            return [], 0
            
        response.raise_for_status()
        
        total_pages = int(response.headers.get('X-WP-TotalPages', 0))
        return response.json(), total_pages
        
    except requests.RequestException as e:
        logger.error(f"API Request Error (page {page}): {e}")
        return [], 0

def get_news_from_period(
    term: str = "ibovespa", 
    start_date: datetime = datetime(2025, 1, 1),
    end_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Orchestrates the news collection from the API within the specified period.
    """
    all_news = []
    page = 1
    continue_scraping = True
    total_pages_known = 0

    log_msg = f"Starting API collection for '{term}' from {start_date.date()}"
    if end_date:
        log_msg += f" up to {end_date.date()}"
    logger.info(log_msg)

    while continue_scraping:
        if total_pages_known > 0:
            logger.info(f"Fetching API page {page} of {total_pages_known}...")
        else:
            logger.info(f"Fetching API page {page}...")
        
        posts, total_pages = fetch_posts_from_api(term, page)
        
        if not posts:
            break
            
        if total_pages_known == 0:
            total_pages_known = total_pages
            logger.info(f"Total pages available in API: {total_pages}")

        for post in posts:
            try:
                date_str = post.get("date")
                if not date_str:
                    continue
                    
                post_date = datetime.fromisoformat(date_str)
                
                # If post is newer than end_date, just skip it (continue to next post)
                if end_date and post_date > end_date:
                    continue

                # If post is older than start_date, we can stop scraping entirely
                if post_date < start_date:
                    logger.info(f"Start date reached ({post_date.date()}). Stopping collection.")
                    continue_scraping = False
                    break
                
                # Rendered title might have HTML entities
                title_raw = post.get("title", {}).get("rendered", "")
                title_clean = html.unescape(title_raw)
                
                all_news.append({
                    'date': post_date,
                    'title': title_clean,
                    'link': post.get("link")
                })
                
            except ValueError as e:
                logger.warning(f"Error processing post date {post.get('id')}: {e}")
            
        if continue_scraping:
            # Extra safety check: stop if we reach the reported total pages
            if total_pages > 0 and page >= total_pages:
                logger.info(f"Reached the last page ({page}/{total_pages}).")
                break

            page += 1
            # WP API is robust, but let's be gentle
            time.sleep(0.1) 
            
    df = pd.DataFrame(all_news)
    if not df.empty:
        df = df.sort_values(by='date', ascending=False)
        logger.info(f"Collection finished. Total of {len(df)} news collected.")
    else:
        logger.warning("Collection finished with no results.")
        
    return df

if __name__ == "__main__":
    # Quick test
    print("Testing API scraper...")
    df = get_news_from_period(term="ibovespa", start_date=datetime(2026, 1, 1)) 
    print(df.head())
    if not df.empty:
        df.to_csv("test_api_scraper.csv", index=False)
        print("Test file saved: test_api_scraper.csv")
