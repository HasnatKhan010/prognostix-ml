import json

nb_path = '01_data_understanding.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb_dict = json.load(f)

for cell in nb_dict.get('cells', []):
    if cell.get('cell_type') == 'code':
        new_source = []
        skip_next = False
        for line in cell.get('source', []):
            if skip_next and ("clip(upper=125)" in line or "Apply piecewise linear" in line):
                continue
            if "merged['RUL'] = merged['max_time_cycles'] - merged['time_cycles']\n" in line:
                new_source.append(line)
                new_source.append("    # Apply piecewise linear RUL capping (commonly capped at 125 for CMAPSS)\n")
                new_source.append("    merged['RUL'] = merged['RUL'].clip(upper=125)\n")
                skip_next = True
            elif "Apply piecewise linear" in line or "clip(upper=125)" in line:
                pass # remove all existing ones
            else:
                new_source.append(line)
                skip_next = False
        cell['source'] = new_source

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb_dict, f, indent=1)

print("Notebook cleaned up.")
