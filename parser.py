import re
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table
from graphviz import Digraph

# Lexical Analiz
def lexical_analysis(expression):
    # Tokens
    token_specifications = [
        ("NUMBER", r'\d+'),  # Sayılar
        ("PLUS", r'\+'),  # +
        ("MINUS", r'-'),  # -
        ("TIMES", r'\*'),  # *
        ("DIVIDE", r'\/'),  # /
        ("LPAREN", r'\('),  # (
        ("RPAREN", r'\)'),  # )
        ("SKIP", r'\s+'),  # Boşlukları atla
        ("MISMATCH", r'.'),  # Tanımlanmayan karakter
    ]

    token_regex = '|'.join(f'(?P<{pair[0]}>{pair[1]})' for pair in token_specifications)
    tokens = []

    for match in re.finditer(token_regex, expression):
        kind = match.lastgroup
        value = match.group()
        if kind == "SKIP":
            continue
        elif kind == "MISMATCH":
            raise SyntaxError(f"Geçersiz karakter: {value}")
        else:
            tokens.append((kind, value))
    return tokens

#------------------------------------------------------------------------------------
#Remove Left Recursion
def remove_left_recursion(grammar):
    new_grammar = {}

    for non_terminal, productions in grammar.items():
        left_recursive = []
        non_left_recursive = []

        for production in productions:
            if production[0] == non_terminal:
                left_recursive.append(production[1:])
            else:
                non_left_recursive.append(production)

        if left_recursive:
            new_non_terminal = f"{non_terminal}PRIME"
            new_grammar[non_terminal] = [[*list(p), new_non_terminal] for p in non_left_recursive]
            new_grammar[new_non_terminal] = [[*list(p), new_non_terminal] for p in left_recursive] + [['ε']]
        else:
            new_grammar[non_terminal] = [[*list(p)] for p in productions]

    return new_grammar

#------------------------------------------------------------------------------------
# Left Factoring
def left_factoring(grammar):
    new_grammar = {}
    for non_terminal, productions in grammar.items():
        prefixes = {}
        for prod in productions:
            prefix = prod[0]
            if prefix not in prefixes:
                prefixes[prefix] = []
            prefixes[prefix].append(prod)

        if any(len(prods) > 1 for prods in prefixes.values()):
            for prefix, prods in prefixes.items():
                if len(prods) > 1:
                    new_non_terminal = non_terminal + "'"
                    new_grammar[non_terminal] = [prefix + new_non_terminal]
                    new_grammar[new_non_terminal] = [prod[len(prefix):] for prod in prods if prod[len(prefix):]] + ['ε']
                    break
        else:
            new_grammar[non_terminal] = productions
    return new_grammar

#------------------------------------------------------------------------------------
# Calculate First Set
def calculate_first(grammar):
    first = {non_terminal: set() for non_terminal in grammar}
    visited = set() 


    def first_of(symbol):
        if symbol not in grammar:
            return {symbol}

        if symbol in visited:
            return first[symbol]

        visited.add(symbol)


        for production in grammar[symbol]:
            if production == "ε":
                first[symbol].add("ε")
            else:
                for char in production:
                    first[symbol] |= first_of(char)                    
                    if "ε" not in first_of(char): 
                        break

        return first[symbol]


    for non_terminal in grammar:
        first_of(non_terminal)

    return first

#------------------------------------------------------------------------------------
# Calculate Follow Set
def compute_follow_sets(grammar, first_sets):
    follow_sets = {non_terminal: set() for non_terminal in grammar}
    start_symbol = next(iter(grammar)) 
    follow_sets[start_symbol].add('$') 

    changed = True

    while changed:
        changed = False 

        for lhs, productions in grammar.items():
            for production in productions:
                for i, symbol in enumerate(production):
                    if symbol in grammar: 
                        follow_before = follow_sets[symbol].copy()

                        if i + 1 < len(production):
                            next_symbol = production[i + 1]
                            if next_symbol in grammar:
                                follow_sets[symbol].update(first_sets[next_symbol] - {'ε'})

                                if 'ε' in first_sets[next_symbol]:
                                    j = i + 1
                                    while j < len(production):
                                        if 'ε' in first_sets.get(production[j-1], set()):
                                            next_symbol = production[j]
                                            follow_sets[symbol].update(first_sets.get(next_symbol, set()) - {'ε'})
                                            j += 1
                                        else:
                                            break

                            else:  # Terminal
                                follow_sets[symbol].add(next_symbol)

                        if i + 1 == len(production) or 'ε' in first_sets.get(production[i + 1], set()):
                            follow_sets[symbol].update(follow_sets[lhs])

                        if follow_before != follow_sets[symbol]:
                            changed = True

    return follow_sets

#------------------------------------------------------------------------------------
def computed_sets(grammar, first, follow):
    data = []
    for non_terminal in grammar.keys():
        nullable = "Yes" if "ε" in first.get(non_terminal, set()) else "No"
        
        row = {
            "Nonterminal": non_terminal,
            "Nullable": nullable,
            "First Set": ", ".join(sorted(first.get(non_terminal, set()))),
            "Follow Set": ", ".join(sorted(follow.get(non_terminal, set()))),
        }
        data.append(row)
    

    return pd.DataFrame(data)
#------------------------------------------------------------------------------------
def show_table(dataframe):
    fig, ax = plt.subplots(figsize=(10, len(dataframe) * 0.5))  # Dinamik boyutlandırma
    ax.axis('off')  # Çerçeveyi kapat

    # Pandas tablosunu Matplotlib ile görselleştir
    table = ax.table(
        cellText=dataframe.values,
        colLabels=dataframe.columns,
        cellLoc='center',
        loc='center'
    )

    # Tablo stil ayarları
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(dataframe.columns))))
    
    plt.show()
#------------------------------------------------------------------------------------
def create_parse_table(grammar, computed_sets, first_sets):
    terminals = {')', '(', '+', '-', '*', '/', '$'}
    non_terminals = grammar.keys()

    parse_table = {non_terminal: {terminal: None for terminal in terminals} for non_terminal in non_terminals}

    for row in computed_sets.itertuples():
        non_terminal = row.Nonterminal
        first_set = row._3.split(", ") if row._3 else []
        follow_set = row._4.split(", ") if row._4 else []

        for terminal in first_set:
            if terminal != "ε":  
                for production in grammar[non_terminal]:
                    if production[0] in non_terminals:    
                        if terminal in first_sets[production[0]]:
                            parse_table[non_terminal][terminal] = production
                    else:
                        parse_table[non_terminal][production[0]] = production

        if row.Nullable == "Yes":
            for terminal in follow_set:
                if parse_table[non_terminal][terminal] is None:
                    parse_table[non_terminal][terminal] = "ε"

    for non_terminal in parse_table:
        if "ε" in parse_table[non_terminal]:
            del parse_table[non_terminal]["ε"]

    return parse_table
#------------------------------------------------------------------------------------
# Parse tablosunu görselleştirme
def display_parse_table(parse_table):
    df = pd.DataFrame(parse_table).T
    df.fillna("", inplace=True)  # Boş hücreleri daha okunabilir hale getir

    # Tamamen boş sütunları kaldır
    df = df.loc[:, (df != "").any(axis=0)]
    print(df)

    # Matplotlib ile tabloyu çiz
    import matplotlib.pyplot as plt
    from pandas.plotting import table

    fig, ax = plt.subplots(figsize=(15, len(parse_table) + 3))
    ax.axis("tight")
    ax.axis("off")
    tbl = table(ax, df, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.2)
    plt.show()


#------------------------------------------------------------------------------------
def parse_input(parse_table, input_tokens, start_symbol):
    stack = ['$', start_symbol]
    transitions = []

    while stack:
        top_of_stack = stack.pop() 
        current_token = input_tokens[0] 

        print(f"Stack top: {top_of_stack}, Current token: {current_token}")

        if top_of_stack == '$' and current_token == '$':
            print('Accept')
            break

        if top_of_stack not in parse_table:
            if top_of_stack == current_token:

                input_tokens.pop(0)
            else:
                raise ValueError(f"Unexpected token: {current_token}. Expected: {top_of_stack}")

        else:
            if current_token in parse_table[top_of_stack]:
                rule = parse_table[top_of_stack][current_token]
                if rule and rule != 'ε':
                    transitions.append(f"Apply rule: {top_of_stack} -> {' '.join(rule)}")
                    stack.extend(reversed(rule))
                else:
                    transitions.append(f"Apply rule: {top_of_stack} -> ε")
            else:
                raise ValueError(f"Unexpected token: {current_token}. Expected: {top_of_stack}")

    return transitions
#------------------------------------------------------------------------------------
def format_transitions(transitions):
    formatted_transitions = []

    for transition in transitions:
        if transition.startswith("Apply rule"):
            parts = transition.split(" -> ")
            parent = parts[0].replace("Apply rule: ", "").strip()
            children = parts[1].split()
            formatted_transitions.append((parent, children))
        elif transition.startswith("Match"):
            symbol = transition.replace("Match: ", "").strip()
            formatted_transitions.append((symbol, [symbol]))
        elif transition == "Accept":
            formatted_transitions.append(("Accept", []))

    return formatted_transitions

#------------------------------------------------------------------------------------
from graphviz import Digraph

def draw_parse_tree(transitions, start_symbol):
    tree = Digraph(format="png", graph_attr={"rankdir": "TB"})

    nodes = []
    edges = {}

    root = {"id": 0, "label": start_symbol, "children": []}
    nodes.append(root)
    tree.node(str(root["id"]), label=start_symbol)

    def dfs_find_and_expand(symbol, children):
        for node in nodes:
            if node["label"] == symbol and not node["children"]:
                for child_label in children:
                    child_id = len(nodes)
                    child_node = {"id": child_id, "label": child_label, "children": []}
                    nodes.append(child_node)
                    node["children"].append(child_node)
                    tree.node(str(child_id), label=child_label)
                    tree.edge(str(node["id"]), str(child_id)) 
                return
        raise ValueError(f"Eşleşen bir düğüm bulunamadı: {symbol}")


    for transition in transitions:
        parent_symbol, child_symbols = transition
        dfs_find_and_expand(parent_symbol, child_symbols)

    tree.render("C:/Intel/parse_tree", format="png", view=True)
#------------------------------------------------------------------------------------
def validate_input(input_tokens):
    if not input_tokens:
        raise ValueError("Input is empty.")
    
    expect_operand = True
    valid_operators = {'+', '-', '*', '/'} 

    if input_tokens[0] in valid_operators:
        raise ValueError(f"Input cannot start with an operator '{input_tokens[0]}'.")
    if input_tokens[-1] in valid_operators:
        raise ValueError(f"Input cannot end with an operator '{input_tokens[-1]}'.")
    
    for token in input_tokens:
        if expect_operand:
            if not token.isdigit():
                raise ValueError(f"Unexpected operator '{token}'. Expected a number.")
            expect_operand = False
        else:
            if token not in valid_operators:
                raise ValueError(f"Unexpected operand '{token}'. Expected an operator.")
            expect_operand = True

    for i in range(1, len(input_tokens)):
        if input_tokens[i] in valid_operators and input_tokens[i-1] in valid_operators:
            raise ValueError(f"Operators cannot appear consecutively: '{input_tokens[i-1]}{input_tokens[i]}'.")
    
    return True

#---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        expression = input("Write mathematical expression: ")
        input_exp = list(expression)
        print("Expression Accepted.\n" if validate_input(input_exp) else "Expression is not valid!\n")
        input_tokens = lexical_analysis(expression)
        print("Tokens of the mathematical expression: ", input_tokens)

        grammar = {
            'E': ['E+T', 'E-T', 'T'],
            'T': ['T*F', 'T/F', 'F'],
            'F': ['(E)', 'n']
        }

        # Remove Left Recursion and Left Factoring
        grammar_no_left_recursion = remove_left_recursion(grammar)
        print(f"\nLeft Recursion Removed.\n{grammar_no_left_recursion}\n")
        grammar_no_left_factoring = left_factoring(grammar_no_left_recursion)
        print(f"Left Factoring Removed.\n{grammar_no_left_factoring}\n")


        # First ve Follow
        first_sets = calculate_first(grammar_no_left_factoring)
        print(f"First set:\n{first_sets}\n")
        follow_sets = compute_follow_sets(grammar_no_left_factoring, first_sets)
        print(f"Follow set:\n{follow_sets}\n")

        # Computed Sets Table
        computed_sets_table = computed_sets(grammar_no_left_factoring, first_sets, follow_sets)
        show_table(computed_sets_table)
        
        # Parse tablosu oluştur ve görüntüle
        parse_table = create_parse_table(grammar_no_left_factoring, computed_sets_table, first_sets)
        display_parse_table(parse_table)

        #Parsing
        start_symbol = 'E'
        
        token_map = {
            'NUMBER': 'n',
            'PLUS': '+',
            'MINUS': '-',
            'TIMES':'*',
            'DIVIDE': '/',
            'LPAREN': '(',
            'RPAREN': ')',
        }
        # Dönüştürülmüş çıktı
        output = [token_map.get(token[0], token[1]) for token in input_tokens] + ['$']

        transitions = parse_input(parse_table, output, start_symbol)
        
        print("\n".join(transitions))

        
        # Transition'ları formatla
        formatted_transitions = format_transitions(transitions)
        # Sonuçları yazdır
        #print("Formatted Transitions:")
        #for transition in formatted_transitions:
        #    print(transition)
        draw_parse_tree(formatted_transitions, start_symbol)
    except SyntaxError as e:
        print("Hata:", e)