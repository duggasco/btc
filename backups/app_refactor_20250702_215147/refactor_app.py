#!/usr/bin/env python3
import re
import sys

def remove_function(content, func_name):
    """Remove a function definition and its body from the content"""
    # Pattern to match function definition and its entire body
    # This handles both simple and complex function bodies
    pattern = rf'^def {func_name}\s*\([^)]*\)\s*:.*?(?=^def\s|\Z)'
    
    # First try with MULTILINE and DOTALL flags
    new_content = re.sub(pattern, '', content, flags=re.MULTILINE | re.DOTALL)
    
    if new_content == content:
        # If that didn't work, try a more aggressive approach
        lines = content.split('\n')
        new_lines = []
        in_function = False
        skip_next_empty = False
        indent_level = 0
        
        for i, line in enumerate(lines):
            # Check if this is the start of our target function
            if re.match(rf'^\s*def {func_name}\s*\(', line):
                in_function = True
                # Get the indent level
                indent_level = len(line) - len(line.lstrip())
                skip_next_empty = True
                continue
            
            # If we're in the function, check if we've exited it
            if in_function:
                # Check if this line is at the same or lower indent level (and not empty)
                if line.strip() and (len(line) - len(line.lstrip())) <= indent_level:
                    # We've exited the function
                    in_function = False
                    # Don't skip this line - it's the next function/code
                    new_lines.append(line)
                elif not line.strip() and skip_next_empty:
                    # Skip empty lines immediately after removing a function
                    skip_next_empty = False
                    continue
                else:
                    # Skip this line - it's part of the function
                    continue
            else:
                new_lines.append(line)
        
        new_content = '\n'.join(new_lines)
    
    return new_content

def remove_page_routing(content, page_name):
    """Remove page routing elif statement"""
    # Pattern to match elif page == "PageName": and the function call on the next line
    pattern = rf'^\s*elif page == ["\']{page_name}["\']:\s*\n\s*show_\w+\(\)\s*\n?'
    new_content = re.sub(pattern, '', content, flags=re.MULTILINE)
    return new_content

def main():
    # Read the file
    with open('src/frontend/app.py', 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Functions to remove
    functions_to_remove = [
        'show_trading',
        'show_portfolio', 
        'show_paper_trading',
        'show_limits',
        'show_paper_trading_portfolio',
        'show_risk'
    ]
    
    # Remove functions
    for func in functions_to_remove:
        print(f"Removing function: {func}")
        content = remove_function(content, func)
    
    # Remove page routing
    pages_to_remove = ['Trading', 'Portfolio', 'Paper Trading', 'Limits']
    for page in pages_to_remove:
        print(f"Removing routing for: {page}")
        content = remove_page_routing(content, page)
    
    # Clean up any double blank lines
    content = re.sub(r'\n\n\n+', '\n\n', content)
    
    # Write the modified content
    if content != original_content:
        with open('src/frontend/app.py', 'w') as f:
            f.write(content)
        print("✓ Refactoring complete")
        
        # Report what was removed
        removed_lines = original_content.count('\n') - content.count('\n')
        print(f"✓ Removed approximately {removed_lines} lines")
    else:
        print("⚠️  No changes were made")

if __name__ == '__main__':
    main()
