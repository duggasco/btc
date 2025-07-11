"""
Auto-refresh manager for Streamlit pages
Handles non-blocking refresh with proper page transition detection
"""

import streamlit as st
import time
import logging
from typing import Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class AutoRefreshManager:
    """
    Manages auto-refresh functionality for Streamlit pages
    
    Features:
    - Non-blocking refresh mechanism
    - Page transition detection
    - Proper cleanup on page exit
    - Configurable refresh intervals
    """
    
    def __init__(self, page_name: str):
        """
        Initialize the auto-refresh manager
        
        Args:
            page_name: Name of the page using this manager
        """
        self.page_name = page_name
        self.session_key = f"{page_name}_refresh_state"
        self.last_refresh_key = f"{page_name}_last_refresh"
        self.active_key = f"{page_name}_refresh_active"
        
        # Initialize session state
        if self.session_key not in st.session_state:
            st.session_state[self.session_key] = {
                'enabled': False,
                'interval': 5,
                'last_refresh': None,
                'refresh_count': 0
            }
        
        # Mark this refresh manager as active
        st.session_state[self.active_key] = True
        
        # Clean up other page refresh managers
        self._cleanup_other_pages()
    
    def _cleanup_other_pages(self):
        """Clean up refresh state from other pages"""
        for key in list(st.session_state.keys()):
            if key.endswith('_refresh_active') and key != self.active_key:
                other_page = key.replace('_refresh_active', '')
                if st.session_state.get(key, False):
                    logger.info(f"Cleaning up refresh state for {other_page}")
                    st.session_state[key] = False
                    
                    # Disable refresh for that page
                    other_session_key = f"{other_page}_refresh_state"
                    if other_session_key in st.session_state:
                        st.session_state[other_session_key]['enabled'] = False
    
    def render_controls(self, sidebar: bool = True, 
                       default_interval: int = 5,
                       default_enabled: bool = True) -> Tuple[bool, int]:
        """
        Render auto-refresh controls
        
        Args:
            sidebar: Whether to render in sidebar
            default_interval: Default refresh interval in seconds
            default_enabled: Whether auto-refresh is enabled by default
            
        Returns:
            Tuple of (enabled, interval)
        """
        container = st.sidebar if sidebar else st
        
        with container:
            st.markdown("### Auto-Refresh Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Get current state
                current_state = st.session_state[self.session_key]
                
                # Auto-refresh toggle
                enabled = st.checkbox(
                    "Enable Auto-Refresh",
                    value=current_state.get('enabled', default_enabled),
                    key=f"{self.page_name}_auto_refresh_enabled",
                    help="Automatically refresh the page at regular intervals"
                )
                
                # Update state
                current_state['enabled'] = enabled
            
            with col2:
                # Refresh interval
                interval = st.number_input(
                    "Interval (seconds)",
                    min_value=1,
                    max_value=300,
                    value=current_state.get('interval', default_interval),
                    step=1,
                    key=f"{self.page_name}_refresh_interval",
                    disabled=not enabled
                )
                
                # Update state
                current_state['interval'] = interval
            
            # Display refresh status
            if enabled:
                if current_state.get('last_refresh'):
                    last_refresh = current_state['last_refresh']
                    time_since = (datetime.now() - last_refresh).total_seconds()
                    next_refresh = interval - time_since
                    
                    if next_refresh > 0:
                        st.info(f"Next refresh in {int(next_refresh)}s")
                    else:
                        st.info("Refreshing...")
                else:
                    st.info(f"First refresh in {interval}s")
                
                # Show refresh count
                refresh_count = current_state.get('refresh_count', 0)
                if refresh_count > 0:
                    st.caption(f"Refreshed {refresh_count} times")
            
            # Save state
            st.session_state[self.session_key] = current_state
            
            return enabled, interval
    
    def should_refresh(self) -> bool:
        """
        Check if the page should refresh
        
        Returns:
            True if refresh is needed, False otherwise
        """
        state = st.session_state.get(self.session_key, {})
        
        if not state.get('enabled', False):
            return False
        
        # Check if this manager is still active
        if not st.session_state.get(self.active_key, False):
            logger.info(f"Refresh manager for {self.page_name} is no longer active")
            return False
        
        last_refresh = state.get('last_refresh')
        interval = state.get('interval', 5)
        
        if last_refresh is None:
            return True
        
        time_since = (datetime.now() - last_refresh).total_seconds()
        return time_since >= interval
    
    def mark_refreshed(self):
        """Mark that a refresh has occurred"""
        state = st.session_state.get(self.session_key, {})
        state['last_refresh'] = datetime.now()
        state['refresh_count'] = state.get('refresh_count', 0) + 1
        st.session_state[self.session_key] = state
        logger.info(f"Page {self.page_name} refreshed (count: {state['refresh_count']})")
    
    def handle_auto_refresh(self):
        """
        Handle auto-refresh logic
        
        This should be called at the end of the page render.
        Uses st.rerun() for non-blocking refresh.
        """
        if self.should_refresh():
            self.mark_refreshed()
            # Use st.rerun() for non-blocking refresh
            st.rerun()
    
    def cleanup(self):
        """Clean up refresh state when leaving the page"""
        logger.info(f"Cleaning up auto-refresh for {self.page_name}")
        
        # Mark as inactive
        st.session_state[self.active_key] = False
        
        # Disable refresh
        if self.session_key in st.session_state:
            st.session_state[self.session_key]['enabled'] = False
    
    def get_status(self) -> dict:
        """
        Get current refresh status
        
        Returns:
            Dictionary with refresh status information
        """
        state = st.session_state.get(self.session_key, {})
        return {
            'page': self.page_name,
            'enabled': state.get('enabled', False),
            'interval': state.get('interval', 5),
            'last_refresh': state.get('last_refresh'),
            'refresh_count': state.get('refresh_count', 0),
            'active': st.session_state.get(self.active_key, False)
        }