"""Data Table Display Component"""
import streamlit as st
import pandas as pd

def render_data_table(df, columns_config=None, height=400, key=None):
    """
    Render a styled data table
    
    Args:
        df: Pandas DataFrame
        columns_config: Dictionary of column configurations
        height: Table height in pixels
        key: Unique key for the component
    """
    # Apply custom styling
    st.markdown("""
    <style>
    div[data-testid="stDataFrame"] {
        background: var(--bg-secondary);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
    }
    
    div[data-testid="stDataFrame"] table {
        background: transparent;
    }
    
    div[data-testid="stDataFrame"] th {
        background: var(--bg-tertiary) !important;
        color: var(--text-secondary) !important;
        font-size: 11px !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 500 !important;
    }
    
    div[data-testid="stDataFrame"] td {
        color: var(--text-primary) !important;
        font-size: 12px !important;
        font-variant-numeric: tabular-nums;
    }
    
    div[data-testid="stDataFrame"] tr:hover {
        background: rgba(255, 255, 255, 0.02) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display dataframe
    st.dataframe(
        df,
        column_config=columns_config,
        height=height,
        use_container_width=True,
        key=key
    )

def format_table_columns(df, column_formats):
    """
    Format DataFrame columns
    
    Args:
        df: Pandas DataFrame
        column_formats: Dictionary mapping column names to format functions
    
    Returns:
        Formatted DataFrame
    """
    df_formatted = df.copy()
    
    for col, format_func in column_formats.items():
        if col in df_formatted.columns:
            df_formatted[col] = df_formatted[col].apply(format_func)
    
    return df_formatted

def create_trade_table(trades_df):
    """Create a formatted trades table"""
    if trades_df.empty:
        st.info("No trades to display")
        return
    
    # Format columns
    column_formats = {
        'price': lambda x: f"${x:,.2f}",
        'amount': lambda x: f"{x:.8f}",
        'value': lambda x: f"${x:,.2f}",
        'profit_loss': lambda x: f"${x:,.2f}",
        'profit_loss_pct': lambda x: f"{x:.2f}%"
    }
    
    # Apply formatting
    formatted_df = format_table_columns(trades_df, column_formats)
    
    # Column configuration for Streamlit
    columns_config = {
        "timestamp": st.column_config.DatetimeColumn(
            "Time",
            format="DD/MM HH:mm"
        ),
        "side": st.column_config.TextColumn(
            "Side",
            width="small"
        ),
        "price": st.column_config.TextColumn(
            "Price",
            width="medium"
        ),
        "amount": st.column_config.TextColumn(
            "Amount",
            width="medium"
        ),
        "profit_loss_pct": st.column_config.TextColumn(
            "P/L %",
            width="small"
        )
    }
    
    render_data_table(formatted_df, columns_config)

def create_position_table(positions_df):
    """Create a formatted positions table"""
    if positions_df.empty:
        st.info("No open positions")
        return
    
    # Format columns
    column_formats = {
        'entry_price': lambda x: f"${x:,.2f}",
        'current_price': lambda x: f"${x:,.2f}",
        'amount': lambda x: f"{x:.8f}",
        'value': lambda x: f"${x:,.2f}",
        'unrealized_pnl': lambda x: f"${x:,.2f}",
        'pnl_percentage': lambda x: f"{x:.2f}%"
    }
    
    formatted_df = format_table_columns(positions_df, column_formats)
    render_data_table(formatted_df)