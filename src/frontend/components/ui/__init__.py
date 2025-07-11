"""
Reusable UI components for the Bitcoin Trading System frontend.
"""

# Tab components
from .tabs import (
    TabComponent,
    render_tabs,
    render_icon_tabs
)

# Card components
from .cards import (
    render_metric_card,
    render_info_card,
    render_alert_card,
    render_chart_card,
    render_stat_cards
)

# Table components
from .tables import (
    DataTable,
    render_condensed_table,
    render_comparison_table,
    render_editable_table
)

# Form components
from .forms import (
    FormValidator,
    render_input_group,
    render_preset_selector,
    render_form_section,
    render_dynamic_form,
    render_quick_settings
)

# Layout utilities
from .layout import (
    create_grid,
    create_section,
    create_columns_with_gap,
    add_vertical_space,
    create_metric_row,
    create_sidebar_section,
    create_responsive_columns,
    create_card_grid,
    create_page_header
)

__all__ = [
    # Tabs
    'TabComponent',
    'render_tabs',
    'render_icon_tabs',
    
    # Cards
    'render_metric_card',
    'render_info_card',
    'render_alert_card',
    'render_chart_card',
    'render_stat_cards',
    
    # Tables
    'DataTable',
    'render_condensed_table',
    'render_comparison_table',
    'render_editable_table',
    
    # Forms
    'FormValidator',
    'render_input_group',
    'render_preset_selector',
    'render_form_section',
    'render_dynamic_form',
    'render_quick_settings',
    
    # Layout
    'create_grid',
    'create_section',
    'create_columns_with_gap',
    'add_vertical_space',
    'create_metric_row',
    'create_sidebar_section',
    'create_responsive_columns',
    'create_card_grid',
    'create_page_header'
]