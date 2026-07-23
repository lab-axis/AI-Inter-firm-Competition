import os

filepath = 'main.py'
with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
in_main = False
skip = False

for i, line in enumerate(lines):
    if line.strip() == 'tg=input(\"Target Company Ticker: \")':
        new_lines.append('    target_companies = [\"INTL\", \"TSLA\", \"META\", \"AAPL\"]\n')
        new_lines.append('    for company in target_companies:\n')
        new_lines.append('        print(f\"\\\\n{\'=\'*80}\")\n')
        new_lines.append('        print(f\" Starting Long-Time Test for Company: {company}\")\n')
        new_lines.append('        print(f\"{\'=\'*80}\")\n')
        skip = True
    elif skip and line.strip() == 'company = tg # editable target company':
        continue
    elif line.strip() == 'print(f\"\\\\n{\'=\'*80}\")' and 'All 12 months completed' in lines[i+1]:
        # end of loop logic
        new_lines.append('        ' + line.lstrip())
    elif line.strip() == 'print(f\"All 12 months completed. Meta Report saved to: {meta_report_file}\")':
        new_lines.append('        ' + line.lstrip())
    elif skip and line.strip() == 'end_time = time.time()':
        skip = False
        new_lines.append(line)
    elif skip:
        # indent by 4 spaces
        new_lines.append('    ' + line)
    else:
        new_lines.append(line)

with open(filepath, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print('Refactoring complete.')
