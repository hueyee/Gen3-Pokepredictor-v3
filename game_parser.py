from bs4 import BeautifulSoup


def extract_battle_log_from_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    soup = BeautifulSoup(html_content, 'html.parser')
    battle_log_data = soup.find('script', {'class': 'battle-log-data'})
    if not battle_log_data:
        raise ValueError("Battle log data not found in the HTML file.")
    return battle_log_data.text.strip()

def parse_replay_data(battle_log):
    turns = []
    player_ratings = {}
    current_turn = []
    in_battle = False

    for line in battle_log.splitlines():
        line = line.strip()
        if not in_battle:
            if line.startswith("|player|"):
                parts = line.split("|")
                if len(parts) > 4:
                    player_id = parts[2]
                    rating = parts[-1]
                    if rating.isdigit():
                        player_ratings[player_id] = int(rating)
            elif line.startswith("|start"):
                in_battle = True
        elif line.startswith("|turn|"):
            if current_turn:
                turns.append(current_turn)
            current_turn = [line]
        elif line:
            current_turn.append(line)

    if current_turn:
        turns.append(current_turn)

    turns.insert(0, {"player_ratings": player_ratings})
    return turns

def extract_features(turns):
    nickname_mapping = {}
    features = []

    for i, turn in enumerate(turns):
        if i == 0:
            features.append({"player_ratings": turn.get("player_ratings", {})})
        else:
            current_features = {"turn_events": []}
            for event in turn:
                if i == len(turns) - 1 and (
                        event.startswith("|raw|") or event.startswith("|l|") or event.startswith("|player|")
                ):
                    continue

                parts = event.split('|')
                if len(parts) > 2 and parts[1] in ['switch', 'drag']:
                    player_pokemon = parts[2]
                    if ': ' in player_pokemon:
                        nickname = player_pokemon.split(': ')[1]
                        canonical_name = parts[3].split(',')[0]
                        nickname_mapping[nickname] = canonical_name

                updated_event = event
                for nickname, canonical_name in nickname_mapping.items():
                    updated_event = updated_event.replace(nickname, canonical_name)
                current_features["turn_events"].append(updated_event)
            features.append(current_features)
    return features

def refine_features(features):
    refined_features = {
        "player_ratings": None,
        "turns": [],
    }

    current_pokemon = {"p1": None, "p2": None}
    previous_pokemon = {
        "p1": [None, None, None, None, None],
        "p2": [None, None, None, None, None]
    }
    revealed_pokemon = {"p1": set(), "p2": set()}
    fainted_pokemon = {"p1": set(), "p2": set()}

    for feature in features:
        if "player_ratings" in feature:
            refined_features["player_ratings"] = feature["player_ratings"]
        else:
            turn_summary = {
                "active_pokemon": {"p1": current_pokemon["p1"], "p2": current_pokemon["p2"]},
                "previous_pokemon": {
                    "p1": previous_pokemon["p1"].copy(),
                    "p2": previous_pokemon["p2"].copy()
                },
                "num_revealed": {
                    "p1": len(revealed_pokemon["p1"]),
                    "p2": len(revealed_pokemon["p2"])
                },
                "moves_used": [],
                "move_attempts": [],
                "damage": [],
                "statuses": [],
                "fainted": [],
                "voluntary_switches": {"p1": False, "p2": False},
            }

            for event in feature["turn_events"]:
                parts = event.split('|')

                if len(parts) > 2 and parts[1] in ["switch", "drag"]:
                    player = parts[2].split(':')[0]
                    pokemon = parts[3].split(',')[0]

                    pid = "p1" if "p1" in player else "p2" if "p2" in player else None

                    if pid:
                        
                        if parts[1] == "switch":
                            turn_summary["voluntary_switches"][pid] = True

                        if current_pokemon[pid] != pokemon and current_pokemon[pid] is not None:
                            previous_pokemon[pid] = [current_pokemon[pid]] + previous_pokemon[pid][:-1]
                        current_pokemon[pid] = pokemon
                        revealed_pokemon[pid].add(pokemon)
                        revealed_pokemon["p1"].add(pokemon)

                elif len(parts) > 2 and parts[1] == "move":
                    pokemon = parts[2].split(': ')[1] if ': ' in parts[2] else None
                    move = parts[3]
                    if pokemon:
                        turn_summary["moves_used"].append({"pokemon": pokemon, "move": move})

                elif len(parts) > 2 and parts[1] == "cant":
                    pokemon_info = parts[2]
                    move_name = parts[4] if len(parts) > 4 else None
                    
                    if pokemon_info and move_name:
                         pokemon = pokemon_info.split(': ')[1] if ': ' in pokemon_info else None
                         if pokemon:
                             turn_summary["move_attempts"].append({"pokemon": pokemon, "move": move_name})

                elif len(parts) > 2 and parts[1] == "singleturn":
                    pokemon_info = parts[2]
                    status_info = parts[3] # "move: Focus Punch"
                    if status_info.startswith("move: "):
                        move_name = status_info.split("move: ")[1]
                        pokemon = pokemon_info.split(': ')[1] if ': ' in pokemon_info else None
                        if pokemon and move_name:
                            turn_summary["move_attempts"].append({"pokemon": pokemon, "move": move_name})

                elif len(parts) > 2 and parts[1] == "-damage":
                    target = parts[2].split(': ')[1] if ': ' in parts[2] else None
                    if target:
                        raw_hp = parts[3].split()[0]
                        if raw_hp == "0":
                            hp_val = 0.0
                        elif "/" in raw_hp:
                            try:
                                current, total = map(int, raw_hp.split("/")[0:2])
                                hp_val = round(current / total, 4)
                            except:
                                hp_val = None
                        else:
                            hp_val = None
                        turn_summary["damage"].append({"target": target, "hp": hp_val})

                elif len(parts) > 2 and parts[1] == "-status":
                    target = parts[2].split(': ')[1] if ': ' in parts[2] else None
                    status = parts[3]
                    if target:
                        turn_summary["statuses"].append({"target": target, "status": status})

                elif len(parts) > 2 and parts[1] == "faint":
                    pokemon_info = parts[2]
                    if ': ' in pokemon_info:
                        player, pokemon = pokemon_info.split(': ')[0], pokemon_info.split(': ')[1]
                        turn_summary["fainted"].append({"player": player, "pokemon": pokemon})
                        if "p1" in player:
                            fainted_pokemon["p1"].add(pokemon)
                        elif "p2" in player:
                            fainted_pokemon["p2"].add(pokemon)

            turn_summary["active_pokemon"] = current_pokemon.copy()
            turn_summary["previous_pokemon"] = {
                "p1": previous_pokemon["p1"].copy(),
                "p2": previous_pokemon["p2"].copy()
            }
            turn_summary["num_revealed"] = {
                "p1": len(revealed_pokemon["p1"]),
                "p2": len(revealed_pokemon["p2"])
            }
            turn_summary["fainted_pokemon"] = {
                "p1": list(fainted_pokemon["p1"]),
                "p2": list(fainted_pokemon["p2"])
            }

            refined_features["turns"].append(turn_summary)

    return refined_features

def generate_parquet_rows(game_id, refined, turn_limit = -1):
    rows = []
    p1_rating = refined["player_ratings"].get("p1")
    p2_rating = refined["player_ratings"].get("p2")

    if turn_limit == -1:
        num_turns = len(refined["turns"])
    else:
        num_turns = min(turn_limit, len(refined["turns"]))

    for turn_id, turn in enumerate(refined["turns"][:num_turns]):
        p1_pokemon = turn["active_pokemon"]["p1"]
        p2_pokemon = turn["active_pokemon"]["p2"]

        p1_prev = turn["previous_pokemon"]["p1"]
        p2_prev = turn["previous_pokemon"]["p2"]

        p1_num_revealed = turn["num_revealed"]["p1"]
        p2_num_revealed = turn["num_revealed"]["p2"]

        p1_fainted = ",".join(turn["fainted_pokemon"]["p1"]) if turn["fainted_pokemon"]["p1"] else None
        p2_fainted = ",".join(turn["fainted_pokemon"]["p2"]) if turn["fainted_pokemon"]["p2"] else None

        p1_move = next((m["move"] for m in turn["moves_used"] if m["pokemon"] == p1_pokemon), None)
        p2_move = next((m["move"] for m in turn["moves_used"] if m["pokemon"] == p2_pokemon), None)

        p1_damage = next((d["hp"] for d in turn["damage"] if d["target"] == p1_pokemon), None)
        p2_damage = next((d["hp"] for d in turn["damage"] if d["target"] == p2_pokemon), None)

        p1_status = next((s["status"] for s in turn["statuses"] if s["target"] == p1_pokemon), None)
        p2_status = next((s["status"] for s in turn["statuses"] if s["target"] == p2_pokemon), None)

        rows.append({
            "game_id": game_id,
            "turn_id": turn_id,
            "p1_rating": p1_rating,
            "p2_rating": p2_rating,

            "p1_current_pokemon": p1_pokemon,
            "p2_current_pokemon": p2_pokemon,

            "p1_first_previous_pokemon": p1_prev[0],
            "p1_second_previous_pokemon": p1_prev[1],
            "p1_third_previous_pokemon": p1_prev[2],
            "p1_fourth_previous_pokemon": p1_prev[3],
            "p1_fifth_previous_pokemon": p1_prev[4],

            "p2_first_previous_pokemon": p2_prev[0],
            "p2_second_previous_pokemon": p2_prev[1],
            "p2_third_previous_pokemon": p2_prev[2],
            "p2_fourth_previous_pokemon": p2_prev[3],
            "p2_fifth_previous_pokemon": p2_prev[4],

            "p1_number_of_pokemon_revealed": p1_num_revealed,
            "p2_number_of_pokemon_revealed": p2_num_revealed,

            "p1_fainted_pokemon": p1_fainted,
            "p2_fainted_pokemon": p2_fainted,

            "p1_move": p1_move,
            "p2_move": p2_move,
            "p1_damage_taken": p1_damage,
            "p1_status": p1_status,
            "p2_damage_taken": p2_damage,
            "p2_status": p2_status,
        })

    return rows