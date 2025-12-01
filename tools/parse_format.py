import re
import json

# Manual fixes for names that the parser might mess up
NAME_FIXES = {
    "hooh": "Ho-Oh",
    "mrmime": "Mr. Mime",
    "deoxysattack": "Deoxys-Attack",
    "deoxysdefense": "Deoxys-Defense",
    "deoxysspeed": "Deoxys-Speed",
    "deoxysnormal": "Deoxys", # Just in case
    "nidoranf": "Nidoran-F",
    "nidoranm": "Nidoran-M",
}

# Forms to IGNORE (Battle-only transformations)
IGNORE_LIST = [
    "castformsunny", "castformrainy", "castformsnowy", "unown" 
    # We keep 'unown' if you want, but usually specific letters are ignored in team preview
]

def parse_ts_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # Regex to find keys like "bulbasaur: {"
    # We look for a word followed by ": {" at the start of a line (or indented)
    pattern = r'^\s*([a-z0-9]+): \{'
    matches = re.findall(pattern, content, re.MULTILINE)

    clean_list = []
    
    for key in matches:
        if key in IGNORE_LIST:
            continue
            
        # 1. Check manual fixes
        if key in NAME_FIXES:
            clean_list.append(NAME_FIXES[key])
            continue
            
        # 2. Standard Capitalization (bulbasaur -> Bulbasaur)
        clean_name = key.capitalize()
        clean_list.append(clean_name)

    return clean_list

def save_pokedex(pokemon_list):
    output_path = "../src/data/pokedex.py"
    with open(output_path, "w") as f:
        f.write("# Auto-generated from formats-data.ts\n\n")
        f.write(f"ALL_POKEMON = {json.dumps(pokemon_list, indent=4)}\n\n")
        f.write(f"NUM_POKEMON = {len(pokemon_list)}\n")
        f.write("NAME_TO_IDX = {name: i for i, name in enumerate(ALL_POKEMON)}\n")
        f.write("IDX_TO_NAME = {i: name for i, name in enumerate(ALL_POKEMON)}\n")
    
    print(f"Success! Saved {len(pokemon_list)} Pokemon to {output_path}")

if __name__ == "__main__":
    # Ensure formats-data.ts is in the same folder or update path
    pokemon = parse_ts_file("../formats-data.ts")
    save_pokedex(pokemon)