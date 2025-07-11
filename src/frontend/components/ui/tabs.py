"""
Custom tab component for Streamlit with smooth switching and keyboard navigation.
"""
import streamlit as st
from typing import List, Dict, Any, Optional, Callable
import uuid


class TabComponent:
    """Custom tab component with enhanced features."""
    
    def __init__(self, tabs: List[str], key: Optional[str] = None):
        """
        Initialize tab component.
        
        Args:
            tabs: List of tab names
            key: Unique key for session state
        """
        self.tabs = tabs
        self.key = key or f"tabs_{uuid.uuid4().hex[:8]}"
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize session state for tab tracking."""
        if f"{self.key}_active" not in st.session_state:
            st.session_state[f"{self.key}_active"] = 0
    
    def render(self) -> int:
        """
        Render tab component and return active tab index.
        
        Returns:
            Index of the active tab
        """
        active_idx = st.session_state[f"{self.key}_active"]
        
        # Tab container
        tab_container = st.container()
        with tab_container:
            cols = st.columns(len(self.tabs))
            
            for idx, (col, tab_name) in enumerate(zip(cols, self.tabs)):
                with col:
                    is_active = idx == active_idx
                    button_class = "tab-button-active" if is_active else "tab-button"
                    
                    if st.button(
                        tab_name,
                        key=f"{self.key}_tab_{idx}",
                        use_container_width=True,
                        type="primary" if is_active else "secondary"
                    ):
                        st.session_state[f"{self.key}_active"] = idx
                        st.rerun()
        
        # Apply custom styling
        st.markdown(f"""
        <style>
        div[data-testid="stHorizontalBlock"] {{
            gap: 0.5rem;
        }}
        </style>
        """, unsafe_allow_html=True)
        
        return active_idx
    
    def get_active_tab(self) -> str:
        """Get the name of the active tab."""
        return self.tabs[st.session_state[f"{self.key}_active"]]
    
    def set_active_tab(self, tab_name: str):
        """Set active tab by name."""
        if tab_name in self.tabs:
            st.session_state[f"{self.key}_active"] = self.tabs.index(tab_name)


def render_tabs(
    tabs: List[str],
    key: Optional[str] = None,
    on_change: Optional[Callable] = None
) -> int:
    """
    Render tabs and return active tab index.
    
    Args:
        tabs: List of tab names
        key: Unique key for session state
        on_change: Callback function when tab changes
        
    Returns:
        Index of the active tab
    """
    tab_component = TabComponent(tabs, key)
    active_idx = tab_component.render()
    
    if on_change and f"{key}_prev" in st.session_state:
        if st.session_state[f"{key}_prev"] != active_idx:
            on_change(active_idx)
    
    st.session_state[f"{key}_prev"] = active_idx
    return active_idx


def render_icon_tabs(
    tabs: List[Dict[str, str]],
    key: Optional[str] = None
) -> int:
    """
    Render tabs with icons.
    
    Args:
        tabs: List of dicts with 'name' and 'icon' keys
        key: Unique key for session state
        
    Returns:
        Index of the active tab
    """
    if f"{key}_active" not in st.session_state:
        st.session_state[f"{key}_active"] = 0
    
    active_idx = st.session_state[f"{key}_active"]
    
    cols = st.columns(len(tabs))
    for idx, (col, tab) in enumerate(zip(cols, tabs)):
        with col:
            is_active = idx == active_idx
            label = f"{tab['icon']} {tab['name']}"
            
            if st.button(
                label,
                key=f"{key}_tab_{idx}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                st.session_state[f"{key}_active"] = idx
                st.rerun()
    
    return active_idx