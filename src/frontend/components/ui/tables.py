"""
Enhanced table components with sorting, pagination, and export functionality.
"""
import streamlit as st
import pandas as pd
from typing import Optional, List, Dict, Any, Callable
import io
import base64


class DataTable:
    """Enhanced data table with advanced features."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        key: Optional[str] = None,
        page_size: int = 10,
        sortable: bool = True,
        exportable: bool = True,
        searchable: bool = True
    ):
        """
        Initialize data table.
        
        Args:
            data: DataFrame to display
            key: Unique key for session state
            page_size: Number of rows per page
            sortable: Enable column sorting
            exportable: Enable export functionality
            searchable: Enable search functionality
        """
        self.data = data.copy()
        self.key = key or "datatable"
        self.page_size = page_size
        self.sortable = sortable
        self.exportable = exportable
        self.searchable = searchable
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize session state for table controls."""
        if f"{self.key}_page" not in st.session_state:
            st.session_state[f"{self.key}_page"] = 0
        if f"{self.key}_sort_column" not in st.session_state:
            st.session_state[f"{self.key}_sort_column"] = None
        if f"{self.key}_sort_ascending" not in st.session_state:
            st.session_state[f"{self.key}_sort_ascending"] = True
        if f"{self.key}_search" not in st.session_state:
            st.session_state[f"{self.key}_search"] = ""
    
    def render(self):
        """Render the data table with all features."""
        # Search bar
        if self.searchable:
            search_term = st.text_input(
                "Search",
                value=st.session_state[f"{self.key}_search"],
                key=f"{self.key}_search_input",
                placeholder="Search in table..."
            )
            st.session_state[f"{self.key}_search"] = search_term
            
            # Filter data based on search
            if search_term:
                mask = self.data.astype(str).apply(
                    lambda x: x.str.contains(search_term, case=False, na=False)
                ).any(axis=1)
                filtered_data = self.data[mask]
            else:
                filtered_data = self.data
        else:
            filtered_data = self.data
        
        # Apply sorting if enabled
        if self.sortable and st.session_state[f"{self.key}_sort_column"]:
            filtered_data = filtered_data.sort_values(
                by=st.session_state[f"{self.key}_sort_column"],
                ascending=st.session_state[f"{self.key}_sort_ascending"]
            )
        
        # Calculate pagination
        total_rows = len(filtered_data)
        total_pages = (total_rows + self.page_size - 1) // self.page_size
        current_page = min(st.session_state[f"{self.key}_page"], total_pages - 1)
        current_page = max(0, current_page)
        
        # Get current page data
        start_idx = current_page * self.page_size
        end_idx = min(start_idx + self.page_size, total_rows)
        page_data = filtered_data.iloc[start_idx:end_idx]
        
        # Controls row
        col1, col2, col3 = st.columns([2, 3, 2])
        
        with col1:
            st.markdown(f"Showing {start_idx + 1}-{end_idx} of {total_rows} rows")
        
        with col2:
            # Pagination controls
            if total_pages > 1:
                cols = st.columns(5)
                with cols[0]:
                    if st.button("◀◀", key=f"{self.key}_first", disabled=current_page == 0):
                        st.session_state[f"{self.key}_page"] = 0
                        st.rerun()
                with cols[1]:
                    if st.button("◀", key=f"{self.key}_prev", disabled=current_page == 0):
                        st.session_state[f"{self.key}_page"] = current_page - 1
                        st.rerun()
                with cols[2]:
                    st.markdown(f"<center>Page {current_page + 1} of {total_pages}</center>", unsafe_allow_html=True)
                with cols[3]:
                    if st.button("▶", key=f"{self.key}_next", disabled=current_page >= total_pages - 1):
                        st.session_state[f"{self.key}_page"] = current_page + 1
                        st.rerun()
                with cols[4]:
                    if st.button("▶▶", key=f"{self.key}_last", disabled=current_page >= total_pages - 1):
                        st.session_state[f"{self.key}_page"] = total_pages - 1
                        st.rerun()
        
        with col3:
            # Export button
            if self.exportable:
                export_format = st.selectbox(
                    "Export",
                    ["CSV", "Excel", "JSON"],
                    key=f"{self.key}_export_format",
                    label_visibility="collapsed"
                )
                if st.button("Export", key=f"{self.key}_export"):
                    self._export_data(filtered_data, export_format)
        
        # Render table with sortable headers
        if self.sortable:
            # Create sortable column headers
            cols = st.columns(len(page_data.columns))
            for idx, (col, column_name) in enumerate(zip(cols, page_data.columns)):
                with col:
                    sort_icon = ""
                    if st.session_state[f"{self.key}_sort_column"] == column_name:
                        sort_icon = " ↑" if st.session_state[f"{self.key}_sort_ascending"] else " ↓"
                    
                    if st.button(
                        f"{column_name}{sort_icon}",
                        key=f"{self.key}_sort_{idx}",
                        use_container_width=True
                    ):
                        if st.session_state[f"{self.key}_sort_column"] == column_name:
                            st.session_state[f"{self.key}_sort_ascending"] = not st.session_state[f"{self.key}_sort_ascending"]
                        else:
                            st.session_state[f"{self.key}_sort_column"] = column_name
                            st.session_state[f"{self.key}_sort_ascending"] = True
                        st.rerun()
        
        # Display the table
        st.dataframe(
            page_data,
            use_container_width=True,
            hide_index=True
        )
    
    def _export_data(self, data: pd.DataFrame, format: str):
        """Export data in the specified format."""
        if format == "CSV":
            csv = data.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">Download CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        elif format == "Excel":
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                data.to_excel(writer, index=False, sheet_name='Data')
            excel_data = output.getvalue()
            b64 = base64.b64encode(excel_data).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="data.xlsx">Download Excel</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        elif format == "JSON":
            json_str = data.to_json(orient='records', indent=2)
            b64 = base64.b64encode(json_str.encode()).decode()
            href = f'<a href="data:file/json;base64,{b64}" download="data.json">Download JSON</a>'
            st.markdown(href, unsafe_allow_html=True)


def render_condensed_table(
    data: pd.DataFrame,
    title: Optional[str] = None,
    max_rows: int = 5,
    highlight_columns: Optional[List[str]] = None,
    format_dict: Optional[Dict[str, str]] = None,
    key: Optional[str] = None
):
    """
    Render a condensed data table for summary views.
    
    Args:
        data: DataFrame to display
        title: Table title
        max_rows: Maximum rows to show
        highlight_columns: Columns to highlight
        format_dict: Column formatting (e.g., {"price": "${:.2f}"})
        key: Unique key
    """
    if title:
        st.markdown(f"### {title}")
    
    # Apply formatting if specified
    if format_dict:
        for col, fmt in format_dict.items():
            if col in data.columns:
                data[col] = data[col].apply(lambda x: fmt.format(x) if pd.notna(x) else "")
    
    # Apply highlighting
    def highlight_cols(row):
        return ['background-color: rgba(255, 255, 0, 0.1)' if col in highlight_columns else '' 
                for col in row.index]
    
    styled_data = data.head(max_rows)
    if highlight_columns:
        styled_data = styled_data.style.apply(highlight_cols, axis=1)
    
    st.dataframe(
        styled_data,
        use_container_width=True,
        hide_index=True
    )
    
    if len(data) > max_rows:
        st.caption(f"Showing {max_rows} of {len(data)} rows")


def render_comparison_table(
    data: pd.DataFrame,
    baseline_column: str,
    comparison_columns: List[str],
    metrics: List[str],
    title: Optional[str] = None,
    format_dict: Optional[Dict[str, str]] = None,
    key: Optional[str] = None
):
    """
    Render a comparison table with baseline and deltas.
    
    Args:
        data: DataFrame with comparison data
        baseline_column: Name of baseline column
        comparison_columns: Names of columns to compare
        metrics: Metrics to compare
        title: Table title
        format_dict: Metric formatting
        key: Unique key
    """
    if title:
        st.markdown(f"### {title}")
    
    # Create comparison table
    comparison_data = []
    for metric in metrics:
        row = {"Metric": metric}
        baseline_value = data.loc[metric, baseline_column]
        row[baseline_column] = baseline_value
        
        for col in comparison_columns:
            value = data.loc[metric, col]
            delta = value - baseline_value
            delta_pct = (delta / baseline_value * 100) if baseline_value != 0 else 0
            
            fmt = format_dict.get(metric, "{:.2f}") if format_dict else "{:.2f}"
            row[col] = f"{fmt.format(value)} ({delta_pct:+.1f}%)"
        
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )


def render_editable_table(
    data: pd.DataFrame,
    key: str,
    on_change: Optional[Callable] = None,
    column_config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Render an editable table with change tracking.
    
    Args:
        data: DataFrame to edit
        key: Unique key for tracking changes
        on_change: Callback when data changes
        column_config: Column configuration for data editor
        
    Returns:
        Edited DataFrame
    """
    edited_data = st.data_editor(
        data,
        use_container_width=True,
        hide_index=True,
        key=key,
        column_config=column_config
    )
    
    # Track changes
    if f"{key}_prev" in st.session_state:
        if not edited_data.equals(st.session_state[f"{key}_prev"]):
            if on_change:
                on_change(edited_data)
    
    st.session_state[f"{key}_prev"] = edited_data.copy()
    return edited_data