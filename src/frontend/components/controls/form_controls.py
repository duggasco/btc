"""Form Control Components"""
import streamlit as st

def create_input_group(label, input_type="text", value=None, placeholder="", 
                      help_text=None, key=None, **kwargs):
    """
    Create a styled input group
    
    Args:
        label: Input label
        input_type: Type of input (text, number, select, etc.)
        value: Default value
        placeholder: Placeholder text
        help_text: Help text below input
        key: Unique key
        **kwargs: Additional arguments for the input
    """
    st.markdown(f"""
    <div class="input-group">
        <label class="input-label">{label}</label>
    </div>
    """, unsafe_allow_html=True)
    
    if input_type == "text":
        result = st.text_input(
            "", 
            value=value or "", 
            placeholder=placeholder,
            key=key,
            label_visibility="collapsed",
            **kwargs
        )
    elif input_type == "number":
        result = st.number_input(
            "",
            value=value or 0.0,
            key=key,
            label_visibility="collapsed",
            **kwargs
        )
    elif input_type == "select":
        options = kwargs.pop('options', [])
        result = st.selectbox(
            "",
            options=options,
            index=options.index(value) if value in options else 0,
            key=key,
            label_visibility="collapsed",
            **kwargs
        )
    elif input_type == "multiselect":
        options = kwargs.pop('options', [])
        result = st.multiselect(
            "",
            options=options,
            default=value or [],
            key=key,
            label_visibility="collapsed",
            **kwargs
        )
    elif input_type == "slider":
        result = st.slider(
            "",
            value=value or kwargs.get('min_value', 0),
            key=key,
            label_visibility="collapsed",
            **kwargs
        )
    elif input_type == "checkbox":
        result = st.checkbox(
            label,
            value=value or False,
            key=key,
            **kwargs
        )
    
    if help_text and input_type != "checkbox":
        st.caption(help_text)
    
    return result

def create_button(label, variant="primary", size="md", full_width=False, 
                 icon=None, key=None, disabled=False):
    """
    Create a styled button
    
    Args:
        label: Button label
        variant: Button variant (primary, secondary, success, danger)
        size: Button size (sm, md, lg)
        full_width: Whether button should be full width
        icon: Icon to display
        key: Unique key
        disabled: Whether button is disabled
    """
    button_class = f"btn btn-{variant} btn-{size}"
    if full_width:
        button_class += " btn-full"
    
    # Use Streamlit button with custom styling
    clicked = st.button(
        label,
        key=key,
        disabled=disabled,
        use_container_width=full_width
    )
    
    # Apply custom styling
    st.markdown(f"""
    <style>
    div[data-testid="stButton"] > button {{
        background: var(--accent-{variant if variant != 'secondary' else 'primary'});
        color: white;
        border: none;
        padding: {'4px 8px' if size == 'sm' else '8px 16px' if size == 'md' else '12px 24px'};
        font-size: {'12px' if size == 'sm' else '13px' if size == 'md' else '14px'};
        font-weight: 500;
        border-radius: var(--radius-sm);
        transition: all var(--transition-fast);
    }}
    
    div[data-testid="stButton"] > button:hover {{
        opacity: 0.9;
        transform: translateY(-1px);
    }}
    </style>
    """, unsafe_allow_html=True)
    
    return clicked

def create_form_section(title, description=None):
    """Create a form section with title and description"""
    st.markdown(f"""
    <div class="form-section">
        <h3 class="form-section-title">{title}</h3>
    """, unsafe_allow_html=True)
    
    if description:
        st.markdown(f"""
        <p class="form-section-description">{description}</p>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def create_trading_form():
    """Create a trading order form"""
    with st.form("trading_form"):
        # Order Type and Side in a single row
        order_type = create_input_group(
            "Order Type",
            input_type="select",
            options=["Market", "Limit", "Stop"],
            key="order_type"
        )
        
        side = create_input_group(
            "Side",
            input_type="select",
            options=["Buy", "Sell"],
            key="order_side"
        )
        
        # Amount
        amount = create_input_group(
            "Amount (BTC)",
            input_type="number",
            min_value=0.0001,
            max_value=10.0,
            step=0.0001,
            key="order_amount"
        )
        
        # Conditional price field
        price = None
        if order_type == "Limit":
            price = create_input_group(
                "Limit Price ($)",
                input_type="number",
                min_value=1000.0,
                max_value=100000.0,
                step=10.0,
                key="order_price"
            )
        
        # Buttons
        submit = st.form_submit_button(
            "Place Order",
            use_container_width=True
        )
        
        return {
            "submit": submit,
            "order_type": order_type,
            "amount": amount,
            "side": side,
            "price": price if order_type == "Limit" else None
        }