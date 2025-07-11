"""
Form components with inline validation and smart defaults.
"""
import streamlit as st
from typing import Optional, List, Dict, Any, Callable, Union, Tuple
from datetime import datetime, date, time
import re


class FormValidator:
    """Form validation utilities."""
    
    @staticmethod
    def validate_email(email: str) -> Tuple[bool, Optional[str]]:
        """Validate email address."""
        pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        if re.match(pattern, email):
            return True, None
        return False, "Invalid email format"
    
    @staticmethod
    def validate_number(value: Union[int, float], min_val: Optional[float] = None, 
                       max_val: Optional[float] = None) -> Tuple[bool, Optional[str]]:
        """Validate numeric value."""
        if min_val is not None and value < min_val:
            return False, f"Value must be at least {min_val}"
        if max_val is not None and value > max_val:
            return False, f"Value must be at most {max_val}"
        return True, None
    
    @staticmethod
    def validate_required(value: Any) -> Tuple[bool, Optional[str]]:
        """Validate required field."""
        if value is None or (isinstance(value, str) and not value.strip()):
            return False, "This field is required"
        return True, None


def render_input_group(
    label: str,
    input_type: str = "text",
    key: Optional[str] = None,
    default: Any = None,
    help_text: Optional[str] = None,
    placeholder: Optional[str] = None,
    required: bool = False,
    validation: Optional[Callable] = None,
    inline: bool = True,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Render an input group with validation.
    
    Args:
        label: Input label
        input_type: Type of input (text, number, email, etc.)
        key: Unique key
        default: Default value
        help_text: Help text
        placeholder: Placeholder text
        required: Whether field is required
        validation: Custom validation function
        inline: Use inline layout
        prefix: Text prefix
        suffix: Text suffix
        **kwargs: Additional arguments for input
        
    Returns:
        Input value
    """
    # Create layout
    if inline and (prefix or suffix):
        cols = []
        if prefix:
            cols.append(1)
        cols.append(3)
        if suffix:
            cols.append(1)
        layout_cols = st.columns(cols)
        col_idx = 0
        
        if prefix:
            with layout_cols[col_idx]:
                st.markdown(f"<div style='padding-top: 30px'>{prefix}</div>", unsafe_allow_html=True)
            col_idx += 1
        
        with layout_cols[col_idx]:
            input_container = st.container()
        
        if suffix:
            with layout_cols[col_idx + 1]:
                st.markdown(f"<div style='padding-top: 30px'>{suffix}</div>", unsafe_allow_html=True)
    else:
        input_container = st.container()
    
    # Render input
    with input_container:
        if input_type == "text":
            value = st.text_input(
                label,
                value=default or "",
                key=key,
                help=help_text,
                placeholder=placeholder,
                **kwargs
            )
        elif input_type == "number":
            value = st.number_input(
                label,
                value=default if default is not None else 0.0,
                key=key,
                help=help_text,
                **kwargs
            )
        elif input_type == "email":
            value = st.text_input(
                label,
                value=default or "",
                key=key,
                help=help_text,
                placeholder=placeholder or "email@example.com",
                **kwargs
            )
        elif input_type == "password":
            value = st.text_input(
                label,
                value=default or "",
                key=key,
                help=help_text,
                type="password",
                **kwargs
            )
        elif input_type == "textarea":
            value = st.text_area(
                label,
                value=default or "",
                key=key,
                help=help_text,
                placeholder=placeholder,
                **kwargs
            )
        elif input_type == "select":
            options = kwargs.pop("options", [])
            value = st.selectbox(
                label,
                options=options,
                index=options.index(default) if default in options else 0,
                key=key,
                help=help_text,
                **kwargs
            )
        elif input_type == "multiselect":
            options = kwargs.pop("options", [])
            value = st.multiselect(
                label,
                options=options,
                default=default or [],
                key=key,
                help=help_text,
                **kwargs
            )
        elif input_type == "date":
            value = st.date_input(
                label,
                value=default or date.today(),
                key=key,
                help=help_text,
                **kwargs
            )
        elif input_type == "time":
            value = st.time_input(
                label,
                value=default or time(0, 0),
                key=key,
                help=help_text,
                **kwargs
            )
        elif input_type == "checkbox":
            value = st.checkbox(
                label,
                value=default or False,
                key=key,
                help=help_text,
                **kwargs
            )
        elif input_type == "radio":
            options = kwargs.pop("options", [])
            value = st.radio(
                label,
                options=options,
                index=options.index(default) if default in options else 0,
                key=key,
                help=help_text,
                **kwargs
            )
        elif input_type == "slider":
            value = st.slider(
                label,
                value=default,
                key=key,
                help=help_text,
                **kwargs
            )
        else:
            value = None
    
    # Validation
    error_key = f"{key}_error" if key else "input_error"
    
    if required and not value:
        st.error("This field is required")
    elif validation and value:
        is_valid, error_msg = validation(value)
        if not is_valid:
            st.error(error_msg)
    elif input_type == "email" and value:
        is_valid, error_msg = FormValidator.validate_email(value)
        if not is_valid:
            st.error(error_msg)
    
    return value


def render_preset_selector(
    label: str,
    presets: Dict[str, Dict[str, Any]],
    key: str,
    on_select: Optional[Callable] = None
) -> Optional[str]:
    """
    Render a preset selector with quick options.
    
    Args:
        label: Selector label
        presets: Dictionary of preset configurations
        key: Unique key
        on_select: Callback when preset is selected
        
    Returns:
        Selected preset name
    """
    preset_names = ["Custom"] + list(presets.keys())
    selected = st.selectbox(
        label,
        options=preset_names,
        key=key
    )
    
    if selected != "Custom" and selected in presets:
        if on_select:
            on_select(presets[selected])
        
        # Display preset details
        with st.expander("Preset Details", expanded=False):
            preset_data = presets[selected]
            for param, value in preset_data.items():
                st.markdown(f"**{param}:** {value}")
    
    return selected if selected != "Custom" else None


def render_form_section(
    title: str,
    fields: List[Dict[str, Any]],
    columns: int = 1,
    key_prefix: Optional[str] = None
) -> Dict[str, Any]:
    """
    Render a form section with multiple fields.
    
    Args:
        title: Section title
        fields: List of field configurations
        columns: Number of columns for layout
        key_prefix: Prefix for field keys
        
    Returns:
        Dictionary of field values
    """
    st.markdown(f"### {title}")
    
    values = {}
    
    # Group fields by rows
    rows = (len(fields) + columns - 1) // columns
    
    for row in range(rows):
        cols = st.columns(columns)
        for col_idx in range(columns):
            field_idx = row * columns + col_idx
            if field_idx < len(fields):
                field = fields[field_idx]
                with cols[col_idx]:
                    field_key = f"{key_prefix}_{field['name']}" if key_prefix else field['name']
                    values[field['name']] = render_input_group(
                        label=field.get('label', field['name']),
                        input_type=field.get('type', 'text'),
                        key=field_key,
                        default=field.get('default'),
                        help_text=field.get('help'),
                        placeholder=field.get('placeholder'),
                        required=field.get('required', False),
                        validation=field.get('validation'),
                        **field.get('kwargs', {})
                    )
    
    return values


def render_dynamic_form(
    sections: List[Dict[str, Any]],
    key: str,
    submit_label: str = "Submit",
    on_submit: Optional[Callable] = None,
    show_reset: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Render a dynamic form with multiple sections.
    
    Args:
        sections: List of section configurations
        key: Form key
        submit_label: Submit button label
        on_submit: Submit callback
        show_reset: Show reset button
        
    Returns:
        Form data if submitted, None otherwise
    """
    form_data = {}
    
    with st.form(key=key):
        for section in sections:
            section_values = render_form_section(
                title=section['title'],
                fields=section['fields'],
                columns=section.get('columns', 1),
                key_prefix=f"{key}_{section['name']}"
            )
            form_data[section['name']] = section_values
        
        # Form actions
        col1, col2 = st.columns([1, 4])
        with col1:
            submitted = st.form_submit_button(submit_label, type="primary")
        with col2:
            if show_reset:
                reset = st.form_submit_button("Reset")
                if reset:
                    st.rerun()
        
        if submitted:
            if on_submit:
                result = on_submit(form_data)
                if result:
                    return form_data
            else:
                return form_data
    
    return None


def render_quick_settings(
    settings: Dict[str, Any],
    key: str,
    columns: int = 2
) -> Dict[str, Any]:
    """
    Render quick settings toggles.
    
    Args:
        settings: Dictionary of setting configurations
        key: Base key
        columns: Number of columns
        
    Returns:
        Dictionary of setting values
    """
    values = {}
    setting_items = list(settings.items())
    rows = (len(setting_items) + columns - 1) // columns
    
    for row in range(rows):
        cols = st.columns(columns)
        for col_idx in range(columns):
            setting_idx = row * columns + col_idx
            if setting_idx < len(setting_items):
                name, config = setting_items[setting_idx]
                with cols[col_idx]:
                    if config['type'] == 'toggle':
                        values[name] = st.checkbox(
                            config['label'],
                            value=config.get('default', False),
                            key=f"{key}_{name}",
                            help=config.get('help')
                        )
                    elif config['type'] == 'select':
                        values[name] = st.selectbox(
                            config['label'],
                            options=config['options'],
                            index=config.get('default_index', 0),
                            key=f"{key}_{name}",
                            help=config.get('help')
                        )
                    elif config['type'] == 'number':
                        values[name] = st.number_input(
                            config['label'],
                            value=config.get('default', 0),
                            min_value=config.get('min'),
                            max_value=config.get('max'),
                            step=config.get('step'),
                            key=f"{key}_{name}",
                            help=config.get('help')
                        )
    
    return values