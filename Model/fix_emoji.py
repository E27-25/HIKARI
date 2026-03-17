"""
Strip ALL non-ASCII characters from every print() call in SIB.py
to prevent UnicodeEncodeError on Windows CP874 terminals.
"""
import re

with open('SIB.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

fixed_lines = []
count = 0
for i, line in enumerate(lines, 1):
    # Only process lines that contain print( and have non-ASCII chars
    if 'print(' in line and any(ord(c) > 127 for c in line):
        cleaned = ''.join(c if ord(c) < 128 else '' for c in line)
        # Fix broken multiplications like ''*20 -> '==='*20 style artifacts
        # e.g. "🏆"*20 becomes ""*20 which is fine, leave as is
        fixed_lines.append(cleaned)
        print(f"Fixed line {i}: {repr(cleaned.strip()[:80])}")
        count += 1
    else:
        fixed_lines.append(line)

with open('SIB.py', 'w', encoding='utf-8') as f:
    f.writelines(fixed_lines)

print(f"\nDone! Fixed {count} lines in SIB.py")
