"""
Layout utilities for responsive grid systems and spacing.
"""
import streamlit as st
from typing import List, Optional, Union, Dict, Any, Callable


def create_grid(
    items: List[Any],
    columns: int = 3,
    gap: str = "medium",
    render_func: Optional[Callable] = None
) -> None:
    """
    Create a responsive grid layout.
    
    Args:
        items: List of items to display
        columns: Number of columns
        gap: Gap size (small, medium, large)
        render_func: Function to render each item
    """
    gap_map = {
        "small": "0.5rem",
        "medium": "1rem",
        "large": "1.5rem"
    }
    
    # Apply custom CSS for grid gap
    st.markdown(f"""
    <style>
    .stColumns > div {{
        padding: {gap_map.get(gap, "1rem")};
    }}
    </style>
    """, unsafe_allow_html=True)
    
    # Create grid
    rows = (len(items) + columns - 1) // columns
    
    for row in range(rows):
        cols = st.columns(columns)
        for col_idx in range(columns):
            item_idx = row * columns + col_idx
            if item_idx < len(items):
                with cols[col_idx]:
                    if render_func:
                        render_func(items[item_idx])
                    else:
                        st.write(items[item_idx])


def create_section(
    title: str,
    content: Optional[Any] = None,
    subtitle: Optional[str] = None,
    icon: Optional[str] = None,
    expandable: bool = False,
    expanded: bool = True,
    help_text: Optional[str] = None,
    actions: Optional[List[Dict[str, Any]]] = None
) -> st.container:
    """
    Create a section container with consistent styling.
    
    Args:
        title: Section title
        content: Section content
        subtitle: Section subtitle
        icon: Icon to display
        expandable: Whether section is expandable
        expanded: Initial expansion state
        help_text: Help tooltip
        actions: List of action buttons
        
    Returns:
        Container for additional content
    """
    if expandable:
        container = st.expander(
            f"{icon} {title}" if icon else title,
            expanded=expanded
        )
    else:
        # Section header
        col1, col2 = st.columns([10, 2])
        with col1:
            if icon:
                st.markdown(f"## {icon} {title}")
            else:
                st.markdown(f"## {title}")
            
            if subtitle:
                st.markdown(f"*{subtitle}*")
        
        with col2:
            if actions:
                for action in actions:
                    if st.button(
                        action.get("label", "Action"),
                        key=action.get("key"),
                        help=action.get("help"),
                        type=action.get("type", "secondary")
                    ):
                        if "callback" in action:
                            action["callback"]()
        
        if help_text:
            st.info(help_text)
        
        container = st.container()
    
    # Add content if provided
    with container:
        if content is not None:
            if callable(content):
                content()
            else:
                st.write(content)
    
    return container


def create_columns_with_gap(
    spec: List[Union[int, float]],
    gap: str = "medium",
    vertical_align: str = "top"
) -> List[st.container]:
    """
    Create columns with custom gap and alignment.
    
    Args:
        spec: Column width specification
        gap: Gap size (small, medium, large)
        vertical_align: Vertical alignment (top, middle, bottom)
        
    Returns:
        List of column containers
    """
    gap_map = {
        "small": "0.5rem",
        "medium": "1rem",
        "large": "1.5rem"
    }
    
    align_map = {
        "top": "flex-start",
        "middle": "center",
        "bottom": "flex-end"
    }
    
    # Apply custom styling
    st.markdown(f"""
    <style>
    .stColumns {{
        gap: {gap_map.get(gap, "1rem")};
        align-items: {align_map.get(vertical_align, "flex-start")};
    }}
    </style>
    """, unsafe_allow_html=True)
    
    return st.columns(spec)


def add_vertical_space(size: Union[int, str] = 1) -> None:
    """
    Add vertical spacing.
    
    Args:
        size: Number of line breaks or CSS size (e.g., "2rem")
    """
    if isinstance(size, int):
        for _ in range(size):
            st.write("")
    else:
        st.markdown(f'<div style="height: {size}"></div>', unsafe_allow_html=True)


def create_metric_row(
    metrics: List[Dict[str, Any]],
    columns: Optional[int] = None
) -> None:
    """
    Create a row of metrics with consistent styling.
    
    Args:
        metrics: List of metric dictionaries
        columns: Number of columns (auto if None)
    """
    num_metrics = len(metrics)
    num_columns = columns or num_metrics
    
    # Create columns
    cols = st.columns(num_columns)
    
    # Distribute metrics across columns
    for idx, metric in enumerate(metrics):
        col_idx = idx % num_columns
        with cols[col_idx]:
            st.metric(
                label=metric.get("label", ""),
                value=metric.get("value", ""),
                delta=metric.get("delta"),
                delta_color=metric.get("delta_color", "normal"),
                help=metric.get("help")
            )


def create_sidebar_section(
    title: str,
    content: Optional[Any] = None,
    icon: Optional[str] = None,
    expanded: bool = True
) -> st.container:
    """
    Create a sidebar section with consistent styling.
    
    Args:
        title: Section title
        content: Section content
        icon: Icon to display
        expanded: Whether to expand by default
        
    Returns:
        Container for additional content
    """
    with st.sidebar:
        with st.expander(f"{icon} {title}" if icon else title, expanded=expanded):
            container = st.container()
            if content is not None:
                with container:
                    if callable(content):
                        content()
                    else:
                        st.write(content)
            return container


def create_responsive_columns(
    mobile: List[int] = [1],
    tablet: List[int] = [1, 1],
    desktop: List[int] = [1, 2, 1]
) -> List[st.container]:
    """
    Create responsive columns that adapt to screen size.
    Note: Streamlit doesn't have true responsive design,
    this is a placeholder for future enhancement.
    
    Args:
        mobile: Column spec for mobile
        tablet: Column spec for tablet
        desktop: Column spec for desktop
        
    Returns:
        List of column containers
    """
    # For now, just use desktop spec
    return st.columns(desktop)


def create_card_grid(
    cards: List[Dict[str, Any]],
    columns: int = 3,
    card_height: Optional[str] = None
) -> None:
    """
    Create a grid of cards with consistent styling.
    
    Args:
        cards: List of card configurations
        columns: Number of columns
        card_height: Fixed card height (CSS value)
    """
    rows = (len(cards) + columns - 1) // columns
    
    for row in range(rows):
        cols = st.columns(columns)
        for col_idx in range(columns):
            card_idx = row * columns + col_idx
            if card_idx < len(cards):
                card = cards[card_idx]
                with cols[col_idx]:
                    with st.container():
                        # Card styling
                        if card_height:
                            st.markdown(f'<div style="height: {card_height}; overflow: auto;">', unsafe_allow_html=True)
                        
                        # Card content
                        if "title" in card:
                            st.markdown(f"### {card['title']}")
                        
                        if "metric" in card:
                            st.metric(
                                label=card["metric"]["label"],
                                value=card["metric"]["value"],
                                delta=card["metric"].get("delta")
                            )
                        
                        if "content" in card:
                            st.write(card["content"])
                        
                        if "chart" in card:
                            st.plotly_chart(card["chart"], use_container_width=True)
                        
                        if card_height:
                            st.markdown('</div>', unsafe_allow_html=True)


def create_page_header(
    title: str,
    subtitle: Optional[str] = None,
    icon: Optional[str] = None,
    breadcrumbs: Optional[List[str]] = None,
    actions: Optional[List[Dict[str, Any]]] = None
) -> None:
    """
    Create a consistent page header.
    
    Args:
        title: Page title
        subtitle: Page subtitle
        icon: Page icon
        breadcrumbs: Breadcrumb navigation
        actions: Action buttons
    """
    # Breadcrumbs
    if breadcrumbs:
        breadcrumb_text = " > ".join(breadcrumbs)
        st.markdown(f"<small>{breadcrumb_text}</small>", unsafe_allow_html=True)
    
    # Header with actions
    col1, col2 = st.columns([10, 2])
    
    with col1:
        if icon:
            st.markdown(f"# {icon} {title}")
        else:
            st.markdown(f"# {title}")
        
        if subtitle:
            st.markdown(f"*{subtitle}*")
    
    with col2:
        if actions:
            for action in actions:
                if st.button(
                    action.get("label", "Action"),
                    key=action.get("key"),
                    help=action.get("help"),
                    type=action.get("type", "secondary"),
                    use_container_width=True
                ):
                    if "callback" in action:
                        action["callback"]()
    
    # Separator
    st.markdown("---")