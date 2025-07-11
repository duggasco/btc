"""
Frontend timezone utilities for EST display formatting.
"""
from datetime import datetime
import pytz
import pandas as pd

EST = pytz.timezone('US/Eastern')

def format_datetime_est(dt, format_str='%Y-%m-%d %H:%M'):
    """Format datetime with EST timezone indicator."""
    if pd.isna(dt) or dt is None:
        return '-'
    
    # Handle pandas Timestamp
    if hasattr(dt, 'to_pydatetime'):
        dt = dt.to_pydatetime()
    
    # Handle string dates
    if isinstance(dt, str):
        try:
            dt = pd.to_datetime(dt)
        except:
            return dt
    
    # Ensure timezone aware
    if hasattr(dt, 'tzinfo') and dt.tzinfo is None:
        # Assume UTC for naive timestamps from API
        dt = pytz.UTC.localize(dt)
    
    # Convert to EST
    if hasattr(dt, 'astimezone'):
        est_dt = dt.astimezone(EST)
        return f"{est_dt.strftime(format_str)} EST"
    
    return str(dt)

def format_time_est(dt):
    """Format time only with EST indicator."""
    return format_datetime_est(dt, '%H:%M:%S')

def format_date_est(dt):
    """Format date only with EST indicator."""
    return format_datetime_est(dt, '%Y-%m-%d')

def add_est_suffix(time_str):
    """Add EST suffix to existing time string."""
    if time_str and not time_str.endswith(' EST'):
        return f"{time_str} EST"
    return time_str