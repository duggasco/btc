#!/bin/bash

# Trading Hub Indentation Error Fix
# Fixes Python indentation issues caused by automatic text replacement

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Fixing Trading Hub Indentation Error...${NC}"

# Find frontend directory
if [ -d "src/frontend" ] && [ -f "src/frontend/app.py" ]; then
    FRONTEND_DIR="src/frontend"
elif [ -d "src" ] && [ -f "src/app.py" ]; then
    FRONTEND_DIR="src"
else
    echo -e "${RED}ERROR: Cannot find frontend directory!${NC}"
    exit 1
fi

APP_FILE="$FRONTEND_DIR/app.py"

# Create backup
echo -e "${YELLOW}Creating backup...${NC}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
cp "$APP_FILE" "$APP_FILE.indent_backup_$TIMESTAMP"
echo -e "${GREEN}✓ Backup created: $APP_FILE.indent_backup_$TIMESTAMP${NC}"

# Show the problematic area first
echo -e "${YELLOW}Checking area around line 3561...${NC}"
sed -n '3555,3570p' "$APP_FILE" | nl -v3555

# Fix the indentation using Python
echo -e "${YELLOW}Applying indentation fix...${NC}"

python3 << 'EOF'
import sys

# Read the file
with open('src/frontend/app.py', 'r') as f:
    lines = f.readlines()

print(f"Total lines in file: {len(lines)}")

# Find problematic areas and fix them
fixed_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    
    # Check for try: statements with no indented content
    if line.strip() == 'try:':
        fixed_lines.append(line)
        i += 1
        
        # Check if next line is properly indented
        if i < len(lines):
            next_line = lines[i]
            if next_line.strip() and not next_line.startswith('    ') and not next_line.startswith('\t'):
                # Next line is not indented, add proper indentation
                current_indent = len(line) - len(line.lstrip())
                proper_indent = ' ' * (current_indent + 4)
                
                # Add indented content
                while i < len(lines) and lines[i].strip():
                    content_line = lines[i]
                    if not content_line.startswith(proper_indent) and content_line.strip():
                        # Fix indentation
                        content_line = proper_indent + content_line.lstrip()
                    fixed_lines.append(content_line)
                    i += 1
            else:
                # Next line is already properly indented or empty
                fixed_lines.append(next_line)
                i += 1
    else:
        fixed_lines.append(line)
        i += 1

# Write the fixed content
with open('src/frontend/app.py', 'w') as f:
    f.writelines(fixed_lines)

print("✓ Applied indentation fixes")
EOF

# Alternative manual fix for common indentation issues
echo -e "${YELLOW}Applying additional fixes...${NC}"

# Fix common indentation patterns that cause issues
python3 << 'EOF'
import re

with open('src/frontend/app.py', 'r') as f:
    content = f.read()

# Fix try: blocks that are missing indented content
def fix_try_blocks(content):
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        if line.strip() == 'try:':
            fixed_lines.append(line)
            current_indent = len(line) - len(line.lstrip())
            proper_indent = ' ' * (current_indent + 4)
            
            # Look ahead to see if next lines need indentation
            j = i + 1
            added_content = False
            
            while j < len(lines):
                next_line = lines[j]
                
                # If it's empty, skip
                if not next_line.strip():
                    fixed_lines.append(next_line)
                    j += 1
                    continue
                
                # If it starts with except:, finally:, or is properly indented, we're good
                if (next_line.strip().startswith('except') or 
                    next_line.strip().startswith('finally') or
                    next_line.startswith(proper_indent)):
                    break
                
                # If it's not indented and has content, indent it
                if next_line.strip() and not next_line.startswith('    '):
                    fixed_lines.append(proper_indent + next_line.lstrip())
                    added_content = True
                else:
                    fixed_lines.append(next_line)
                    added_content = True
                
                j += 1
                break  # Only fix the immediate next line
            
            i = j - 1  # Will be incremented at end of loop
        else:
            fixed_lines.append(line)
        
        i += 1
    
    return '\n'.join(fixed_lines)

# Apply the fix
fixed_content = fix_try_blocks(content)

# Write back
with open('src/frontend/app.py', 'w') as f:
    f.write(fixed_content)

print("✓ Fixed try block indentation")
EOF

# Check if the specific error line still exists
echo -e "${YELLOW}Checking for remaining syntax errors...${NC}"

# Test the Python syntax
if python3 -m py_compile "$APP_FILE" 2>/dev/null; then
    echo -e "${GREEN}✓ Python syntax is now valid${NC}"
else
    echo -e "${YELLOW}⚠ Still has syntax issues, applying emergency fix...${NC}"
    
    # Emergency fix: Look for the specific line mentioned in error
    python3 << 'EOF'
with open('src/frontend/app.py', 'r') as f:
    lines = f.readlines()

# Find line 3561 or nearby try: statements and fix them
for i, line in enumerate(lines, 1):
    if i >= 3560 and i <= 3565 and 'try:' in line:
        print(f"Found try statement at line {i}: {line.strip()}")
        
        # Check if next line exists and is indented
        if i < len(lines):
            next_line = lines[i]  # lines[i] is actually line i+1 since enumerate starts at 1
            if next_line.strip() and not next_line.startswith('    '):
                print(f"Line {i+1} needs indentation: {next_line.strip()}")
                
                # Fix this specific case
                current_indent = len(line) - len(line.lstrip())
                proper_indent = ' ' * (current_indent + 4)
                lines[i] = proper_indent + next_line.lstrip()
                print(f"Fixed line {i+1}")

# Write back
with open('src/frontend/app.py', 'w') as f:
    f.writelines(lines)

print("✓ Applied emergency indentation fix")
EOF
fi

# Final syntax check
echo -e "${YELLOW}Final syntax validation...${NC}"
if python3 -m py_compile "$APP_FILE" 2>/dev/null; then
    echo -e "${GREEN}✅ Python syntax is now valid!${NC}"
else
    echo -e "${RED}❌ Still has syntax errors. Showing error details:${NC}"
    python3 -m py_compile "$APP_FILE"
    
    echo -e "\n${YELLOW}Emergency restoration option:${NC}"
    echo "If the file is too corrupted, you can restore from backup:"
    echo "cp $APP_FILE.indent_backup_$TIMESTAMP $APP_FILE"
fi

# Show the fixed area
echo -e "\n${YELLOW}Fixed area around line 3561:${NC}"
sed -n '3555,3570p' "$APP_FILE" | nl -v3555

# Restart frontend container
echo -e "\n${YELLOW}Restarting frontend container...${NC}"
if command -v docker &> /dev/null && docker compose ps 2>/dev/null | grep -q "frontend"; then
    docker compose restart frontend
    echo -e "${GREEN}✓ Frontend container restarted${NC}"
else
    echo -e "${YELLOW}⚠ Docker not available or frontend not running${NC}"
fi

echo -e "\n${GREEN}🎉 Indentation fix completed!${NC}"
echo -e "${BLUE}The IndentationError should now be resolved.${NC}"
echo -e "${BLUE}Test by visiting http://localhost:8501${NC}"

echo -e "\n${YELLOW}Backup location: $APP_FILE.indent_backup_$TIMESTAMP${NC}"
