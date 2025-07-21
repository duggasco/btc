"""Dashboard Grid Layout Component"""
import streamlit as st

def render_dashboard_header(title, status_indicators=None):
    """Render dashboard header with title and status indicators"""
    header_html = f"""
    <div class="dashboard-header">
        <h1>{title}</h1>
        <div class="status-indicators">
    """
    
    if status_indicators:
        for indicator in status_indicators:
            status_class = "online" if indicator["status"] == "online" else "offline"
            header_html += f"""
            <div class="status-indicator">
                <div class="status-dot {status_class}"></div>
                <span>{indicator["label"]}</span>
            </div>
            """
    
    header_html += """
        </div>
    </div>
    """
    
    st.markdown(header_html, unsafe_allow_html=True)

def create_dashboard_grid():
    """Create the main dashboard grid layout"""
    st.markdown("""
    <style>
    .dashboard-container {
        display: grid;
        grid-template-columns: 1fr 350px;
        grid-template-rows: auto 1fr 350px;
        gap: 12px;
        min-height: calc(100vh - 150px);
    }
    
    .chart-section {
        grid-column: 1;
        grid-row: 2;
        min-height: 400px;
    }
    
    .signals-section {
        grid-column: 2;
        grid-row: 2 / -1;
    }
    
    .bottom-section {
        grid-column: 1;
        grid-row: 3;
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
    }
    
    @media (max-width: 1024px) {
        .dashboard-container {
            grid-template-columns: 1fr;
            grid-template-rows: auto;
        }
        
        .signals-section {
            grid-column: 1;
            grid-row: auto;
            max-height: 400px;
        }
        
        .bottom-section {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Return column containers for content
    col1 = st.container()
    col2 = st.container()
    col3 = st.container()
    
    return col1, col2, col3

def create_grid_card(title, content_func, card_class=""):
    """Create a card within the grid"""
    card_html = f"""
    <div class="card {card_class}">
        <div class="card-header">
            <h3 class="card-title">{title}</h3>
        </div>
        <div class="card-body">
    """
    st.markdown(card_html, unsafe_allow_html=True)
    
    # Execute content function
    content_func()
    
    st.markdown("</div></div>", unsafe_allow_html=True)