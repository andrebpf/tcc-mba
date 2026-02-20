import logging
from datetime import datetime
from typing import Optional

# Avoid circular dependencies by importing only if necessary or using getLogger directly
logger = logging.getLogger("mba_tcc.utils.date_parser")

def parse_date(date_str: str) -> Optional[datetime]:
    """
    Parses the date string from InfoMoney (DD/MM/YYYY) into a datetime object.
    Returns None in case of failure.
    """
    if not date_str:
        return None
        
    try:
        # Clean extra whitespace that might come from HTML
        clean_date_str = date_str.strip()
        return datetime.strptime(clean_date_str, '%d/%m/%Y')
    except ValueError:
        logger.warning(f"Unexpected date format: '{date_str}'")
        return None
