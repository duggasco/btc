# BTC Trading System CSS Framework

## Overview
A professional dark-themed CSS framework designed specifically for the Bitcoin trading system frontend. Built with a mobile-first approach and optimized for financial data visualization.

## File Structure
```
assets/css/
├── index.css         # Main entry point - import this in Streamlit
├── main.css          # Core design system and utilities
├── components.css    # Reusable component styles
├── responsive.css    # Breakpoints and responsive design
└── streamlit.css     # Streamlit-specific overrides
```

## Design System

### Color Palette
- **Primary Background**: `#0a0e1a` - Deep dark blue
- **Secondary Background**: `#121826` - Slightly lighter
- **Card Background**: `#1e2940` - Component backgrounds
- **Text Primary**: `#e2e8f0` - High contrast white
- **Text Secondary**: `#94a3b8` - Muted gray
- **Accent Blue**: `#3b82f6` - Primary actions
- **Success Green**: `#10b981` - Positive values
- **Error Red**: `#ef4444` - Negative values

### Typography
- **Font Stack**: Inter, system fonts fallback
- **Base Size**: 14px (0.875rem) - Compact professional look
- **Scale**: xs(12px) → sm(13px) → base(14px) → lg(16px) → xl(18px)
- **Monospace**: For numbers and timestamps

### Spacing System
- Compact spacing for data-dense interfaces
- Scale: xs(4px) → sm(8px) → md(12px) → lg(16px) → xl(24px)

## Usage in Streamlit

### 1. Import in your Streamlit app
```python
import streamlit as st

# Load custom CSS
def load_css():
    with open('assets/css/index.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Call in your app
load_css()
```

### 2. Apply component classes
```python
# Metric card
st.markdown('''
<div class="metric-card">
    <div class="metric-card__label">BTC Price</div>
    <div class="metric-card__value">$98,765.43</div>
    <div class="metric-card__change metric-card__change--positive">
        ↑ 2.34%
    </div>
</div>
''', unsafe_allow_html=True)

# Data table with custom styling
st.markdown('<div class="data-table">', unsafe_allow_html=True)
st.dataframe(df)
st.markdown('</div>', unsafe_allow_html=True)
```

### 3. Utility classes
```python
# Spacing
st.markdown('<div class="p-lg mb-xl">Content</div>', unsafe_allow_html=True)

# Grid layout
st.markdown('''
<div class="grid grid-cols-3 gap-md">
    <div>Column 1</div>
    <div>Column 2</div>
    <div>Column 3</div>
</div>
''', unsafe_allow_html=True)

# Text styling
st.markdown('<p class="text-secondary text-sm">Subtitle</p>', unsafe_allow_html=True)
```

## Component Classes

### Metric Cards
```html
<div class="metric-card">
    <div class="metric-card__label">Label</div>
    <div class="metric-card__value">Value</div>
    <div class="metric-card__change metric-card__change--positive">Change</div>
</div>
```

### Buttons
```html
<button class="btn btn--primary">Primary Action</button>
<button class="btn btn--secondary">Secondary</button>
<button class="btn btn--success">Buy</button>
<button class="btn btn--error">Sell</button>
<button class="btn btn--ghost">Cancel</button>
```

### Forms
```html
<div class="form-group">
    <label class="form-label">Amount</label>
    <input type="text" class="form-input" placeholder="0.00">
    <div class="form-helper">Enter amount in BTC</div>
</div>
```

### Tables
```html
<div class="data-table">
    <table>
        <thead>
            <tr>
                <th>Time</th>
                <th>Price</th>
                <th>Change</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>12:34:56</td>
                <td class="data-table__number">98,765.43</td>
                <td><span class="badge badge--success">+2.34%</span></td>
            </tr>
        </tbody>
    </table>
</div>
```

### Alerts
```html
<div class="alert alert--info">
    <div class="alert__icon">ℹ️</div>
    <div>Information message</div>
</div>
```

## Responsive Design

### Breakpoints
- Mobile: < 640px (default)
- Tablet: >= 640px (prefix: `sm:`)
- Desktop: >= 1024px (prefix: `lg:`)
- Wide: >= 1280px (prefix: `xl:`)

### Responsive Classes
```html
<!-- Hidden on mobile, visible on desktop -->
<div class="hidden lg:block">Desktop only</div>

<!-- Different grid columns by screen size -->
<div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4">
    <!-- Items -->
</div>
```

## Trading-Specific Classes

### Price Changes
```html
<span class="price-up">↑ $1,234.56</span>
<span class="price-down">↓ $1,234.56</span>
```

### Signal Strength
```html
<span class="signal-strong">Strong Buy</span>
<span class="signal-moderate">Hold</span>
<span class="signal-weak">Weak Sell</span>
```

### Connection Status
```html
<div class="connection-status connection-status--connected">
    <span class="status-dot status-dot--online"></span>
    Connected
</div>
```

### Risk Indicators
```html
<div class="risk-indicator risk-indicator--medium">
    <div class="risk-indicator__bar"></div>
    <div class="risk-indicator__bar risk-indicator__bar--active"></div>
    <div class="risk-indicator__bar risk-indicator__bar--active"></div>
    <div class="risk-indicator__bar"></div>
    <div class="risk-indicator__bar"></div>
</div>
```

## Performance Optimizations

1. **CSS Variables**: Dynamic theming without recompilation
2. **Mobile-First**: Styles optimized for smaller screens first
3. **Minimal Specificity**: Easy to override when needed
4. **Hardware Acceleration**: Transforms and opacity for animations
5. **Reduced Motion**: Respects user preferences

## Browser Support
- Chrome/Edge: Latest 2 versions
- Firefox: Latest 2 versions
- Safari: Latest 2 versions
- Mobile browsers: iOS Safari 14+, Chrome Android

## Customization

### Changing Colors
Edit CSS variables in `:root` in main.css:
```css
:root {
  --primary-bg: #0a0e1a;  /* Change this */
  --accent-blue: #3b82f6;  /* Change this */
}
```

### Adding New Components
1. Add component styles to components.css
2. Follow BEM naming convention
3. Use CSS variables for colors and spacing
4. Test responsive behavior

### Dark/Light Theme
Currently optimized for dark theme only. For light theme support, create theme-specific CSS variables.