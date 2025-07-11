"""
Shared page styling module for consistent theming across all pages
"""
import streamlit as st
from pathlib import Path

def apply_page_config(page_title: str, page_icon: str = "₿"):
    """
    Apply consistent page configuration to all pages
    
    Args:
        page_title: Title for the page
        page_icon: Icon for the page (default: Bitcoin symbol)
    """
    import config
    
    st.set_page_config(
        page_title=f"{page_title} - BTC Trading System",
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": config.GITHUB_REPO_URL,
            "Report a bug": f"{config.GITHUB_REPO_URL}/issues",
            "About": "# BTC Trading System\nAI-powered Bitcoin trading with 50+ indicators and real-time analysis"
        }
    )

def inject_custom_css():
    """Load and inject custom CSS files"""
    css_dir = Path(__file__).parent.parent / "styles"
    css_files = ["professional_theme.css"]
    
    combined_css = ""
    for css_file in css_files:
        css_path = css_dir / css_file
        if css_path.exists():
            with open(css_path, 'r') as f:
                combined_css += f.read() + "\n"
    
    if combined_css:
        st.markdown(f"<style>{combined_css}</style>", unsafe_allow_html=True)
    
    # Add additional page-specific styles
    st.markdown("""
    <style>
    /* Page layout styles */
    .global-header {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
    }
    
    .global-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }
    
    .feature-list {
        list-style: none;
        padding: 0;
        margin: 1rem 0;
    }
    
    .feature-list li {
        padding: 0.25rem 0;
        font-size: 0.875rem;
        color: var(--text-secondary);
    }
    
    .feature-list li:before {
        content: "→ ";
        color: var(--accent-primary);
        font-weight: bold;
    }
    
    .footer {
        text-align: center;
        padding: 2rem 0;
    }
    
    /* Paper Trading specific styles */
    .tip-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        text-align: center;
        transition: all 0.2s ease;
    }
    
    .tip-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .tip-card h4 {
        color: var(--text-primary);
        margin: 0.5rem 0;
        font-size: 1.1rem;
    }
    
    .suggestion-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        transition: all 0.2s ease;
    }
    
    .suggestion-card:hover {
        border-color: var(--accent-primary);
        background: var(--bg-hover);
    }
    
    .order-form {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 1.5rem;
    }
    
    .position-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.2s ease;
    }
    
    .position-card:hover {
        border-color: var(--accent-primary);
        background: var(--bg-hover);
    }
    </style>
    """, unsafe_allow_html=True)

def render_page_header(title: str, subtitle: str = None):
    """
    Render a consistent page header
    
    Args:
        title: Main page title
        subtitle: Optional subtitle
    """
    header_html = f"""
    <div class="global-header">
        <h1>{title}</h1>
    """
    
    if subtitle:
        header_html += f'<p class="text-secondary">{subtitle}</p>'
    
    header_html += "</div>"
    
    st.markdown(header_html, unsafe_allow_html=True)

def setup_page(page_name: str, page_title: str, page_subtitle: str = None):
    """
    Complete page setup with all styling and common elements
    
    Args:
        page_name: Name for page identification (no longer used for navigation)
        page_title: Title for the page
        page_subtitle: Optional subtitle for the page
    
    Returns:
        api_client: Initialized API client for the page to use
    """
    import config
    from components.api_client import APIClient
    
    # Apply page configuration
    apply_page_config(page_title)
    
    # Inject custom CSS
    inject_custom_css()
    
    # Initialize session state
    if "api_client" not in st.session_state:
        st.session_state.api_client = APIClient(base_url=config.API_BASE_URL)
    
    # Render page header
    render_page_header(page_title, page_subtitle)
    
    return st.session_state.api_client