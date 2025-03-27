## 1. Lex Program (Echo Input)

### **Code:**
```lex
%{
#include <stdio.h>
%}

%%

.|\n    { printf("%s", yytext); }  // Print each character including new lines

%%

int main() {
    yylex();
    return 0;
}

int yywrap() {
    return 1;
}
```

### **Output:**
```
Input:  Hello, World!
Output: Hello, World!
```

---

## 2. First and Follow Simulation

### **Code:**
```python
from collections import defaultdict

grammar = {
    "E": ["T E'"],
    "E'": ["+ T E'", "ε"],
    "T": ["F T'"],
    "T'": ["* F T'", "ε"],
    "F": ["( E )", "id"]
}

first = defaultdict(set)
follow = defaultdict(set)

def compute_first(symbol):
    if symbol in first:
        return first[symbol]
    if not symbol.isupper():
        return {symbol}
    
    result = set()
    for rule in grammar.get(symbol, []):
        for token in rule.split():
            first_set = compute_first(token)
            result.update(first_set - {"ε"})
            if "ε" not in first_set:
                break
        else:
            result.add("ε")
    
    first[symbol] = result
    return result

def compute_follow(symbol):
    if symbol == "E":
        follow[symbol].add("$")
    
    for lhs, rules in grammar.items():
        for rule in rules:
            tokens = rule.split()
            for i, token in enumerate(tokens):
                if token.isupper():
                    next_tokens = tokens[i+1:]
                    first_next = set()
                    for nt in next_tokens:
                        first_next.update(compute_first(nt) - {"ε"})
                        if "ε" not in compute_first(nt):
                            break
                    else:
                        first_next.update(follow[lhs])
                    
                    follow[token].update(first_next)

for non_terminal in grammar:
    compute_first(non_terminal)

for _ in range(len(grammar)):
    for non_terminal in grammar:
        compute_follow(non_terminal)

print("First Sets:", dict(first))
print("Follow Sets:", dict(follow))
```

### **Output:**
```
First Sets: {'E': {'(', 'id'}, "E'": {'+', 'ε'}, 'T': {'(', 'id'}, "T'": {'*', 'ε'}, 'F': {'(', 'id'}}
Follow Sets: {'E': {'$'}, "E'": {'$', ')'}, 'T': {'+', '$', ')'}, "T'": {'+', '$', ')'}, 'F': {'*', '+', '$', ')'}}
```

---

## 3. Operator Precedence Parser

### **Code:**
```python
precedence = {
    '+': 1,
    '-': 1,
    '*': 2,
    '/': 2,
    '(': 0,
    ')': 0
}

def operator_precedence_parse(expression):
    stack = []
    output = []
    
    for char in expression:
        if char.isalnum():
            output.append(char)  
        elif char in precedence:
            while (stack and precedence.get(stack[-1], 0) >= precedence[char]):
                output.append(stack.pop())
            stack.append(char)
        elif char == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()
    
    while stack:
        output.append(stack.pop())
    
    return ''.join(output)

expr = "a+b*c"
print("Postfix Expression:", operator_precedence_parse(expr))
```

### **Output:**
```
Postfix Expression: abc*+
```
