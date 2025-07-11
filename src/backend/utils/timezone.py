"""
Timezone utility module for consistent EST time handling across the application.
"""
from datetime import datetime
import pytz
from typing import Optional, Union

EST = pytz.timezone('US/Eastern')

def get_est_now() -> datetime:
    """Get current time in EST timezone."""
    return datetime.now(EST)

def get_est_time(dt: Optional[datetime] = None) -> datetime:
    """
    Convert a datetime to EST timezone.
    If no datetime provided, returns current EST time.
    """
    if dt is None:
        return get_est_now()
    
    if dt.tzinfo is None:
        # Assume UTC for naive datetime objects
        dt = pytz.UTC.localize(dt)
    
    return dt.astimezone(EST)

def localize_to_est(dt: datetime) -> datetime:
    """
    Localize a naive datetime to EST timezone.
    """
    if dt.tzinfo is not None:
        return dt.astimezone(EST)
    return EST.localize(dt)

def convert_from_utc_to_est(dt: datetime) -> datetime:
    """
    Convert UTC datetime to EST.
    """
    if dt.tzinfo is None:
        dt = pytz.UTC.localize(dt)
    return dt.astimezone(EST)

def format_est_time(dt: Optional[datetime] = None, fmt: str = '%Y-%m-%d %H:%M:%S') -> str:
    """
    Format datetime in EST timezone.
    """
    est_time = get_est_time(dt)
    return f"{est_time.strftime(fmt)} EST"

def get_est_timestamp() -> str:
    """
    Get current EST timestamp in ISO format.
    """
    return get_est_now().isoformat()

def convert_unix_to_est(timestamp: Union[int, float], unit: str = 'ms') -> datetime:
    """
    Convert Unix timestamp to EST datetime.
    
    Args:
        timestamp: Unix timestamp
        unit: 'ms' for milliseconds, 's' for seconds
    """
    if unit == 'ms':
        dt = datetime.utcfromtimestamp(timestamp / 1000)
    else:
        dt = datetime.utcfromtimestamp(timestamp)
    
    dt = pytz.UTC.localize(dt)
    return dt.astimezone(EST)

def ensure_est_timezone(df):
    """
    Ensure a pandas DataFrame's datetime index is in EST timezone.
    """
    if hasattr(df.index, 'tz'):
        if df.index.tz is None:
            # Assume UTC for naive timestamps
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('US/Eastern')
    return df