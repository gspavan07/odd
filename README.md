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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define MAX_NON_TERMINALS 100
#define MAX_PRODUCTIONS 100
#define MAX_SYMBOLS 100

// Structure to represent a production rule
typedef struct {
    char lhs;
    char rhs[MAX_SYMBOLS];
} Production;

// Structure to represent the grammar
typedef struct {
    char nonTerminals[MAX_NON_TERMINALS];
    Production productions[MAX_PRODUCTIONS];
    int numNonTerminals;
    int numProductions;
} Grammar;

// Function to initialize the grammar
void initGrammar(Grammar *grammar) {
    grammar->numNonTerminals = 0;
    grammar->numProductions = 0;
}

// Function to add a non-terminal to the grammar
void addNonTerminal(Grammar *grammar, char nonTerminal) {
    grammar->nonTerminals[grammar->numNonTerminals++] = nonTerminal;
}

// Function to add a production rule to the grammar
void addProduction(Grammar *grammar, char lhs, char *rhs) {
    grammar->productions[grammar->numProductions].lhs = lhs;
    strcpy(grammar->productions[grammar->numProductions].rhs, rhs);
    grammar->numProductions++;
}

// Function to compute the FIRST set of a symbol
void computeFirst(Grammar *grammar, char symbol, bool firstSets[][MAX_NON_TERMINALS]) {
    for (int i = 0; i < grammar->numProductions; i++) {
        if (grammar->productions[i].lhs == symbol) {
            char *rhs = grammar->productions[i].rhs;
            if (rhs[0] >= 'a' && rhs[0] <= 'z') {
                firstSets[symbol - 'A'][rhs[0] - 'a'] = true;
            } else if (rhs[0] == 'ε') {
                firstSets[symbol - 'A'][0] = true;
            } else {
                computeFirst(grammar, rhs[0], firstSets);
            }
        }
    }
}

// Function to compute the FOLLOW set of a symbol
void computeFollow(Grammar *grammar, char symbol, bool followSets[][MAX_NON_TERMINALS]) {
    for (int i = 0; i < grammar->numProductions; i++) {
        char *rhs = grammar->productions[i].rhs;
        for (int j = 0; j < strlen(rhs); j++) {
            if (rhs[j] == symbol) {
                if (j == strlen(rhs) - 1) {
                    followSets[symbol - 'A'][grammar->productions[i].lhs - 'A'] = true;
                } else {
                    computeFirst(grammar, rhs[j + 1], followSets);
                    for (int k = 0; k < MAX_NON_TERMINALS; k++) {
                        if (followSets[rhs[j + 1] - 'A'][k]) {
                            followSets[symbol - 'A'][k] = true;
                        }
                    }
                }
            }
        }
    }
}

int main() {
    Grammar grammar;
    initGrammar(&grammar);

    printf("Enter the number of non-terminals: ");
    int numNonTerminals;
    scanf("%d", &numNonTerminals);

    for (int i = 0; i < numNonTerminals; i++) {
        char nonTerminal;
        printf("Enter non-terminal %d: ", i + 1);
        scanf(" %c", &nonTerminal);
        addNonTerminal(&grammar, nonTerminal);
    }

    printf("Enter the number of production rules: ");
    int numProductions;
    scanf("%d", &numProductions);

    for (int i = 0; i < numProductions; i++) {
        char lhs;
        char rhs[MAX_SYMBOLS];
        printf("Enter production rule %d (LHS RHS): ", i + 1);
        scanf(" %c %s", &lhs, rhs);
        addProduction(&grammar, lhs, rhs);
    }

    bool firstSets[MAX_NON_TERMINALS][MAX_NON_TERMINALS] = {false};
    bool followSets[MAX_NON_TERMINALS][MAX_NON_TERMINALS] = {false};

    for (int i = 0; i < grammar.numNonTerminals; i++) {
        computeFirst(&grammar, grammar.nonTerminals[i], firstSets);
    }

    for (int i = 0; i < grammar.numNonTerminals; i++) {
        computeFollow(&grammar, grammar.nonTerminals[i], followSets);
    }

    printf("First Sets:\n");
    for (int i = 0; i < grammar.numNonTerminals; i++) {
        printf("%c: ", grammar.nonTerminals[i]);
        for (

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
#include <stdio.h>
#include <string.h>

#define MAX_SIZE 100

int precedence(char c) {
    if (c == '+' || c == '-') return 1;
    if (c == '*' || c == '/') return 2;
    return 0;
}

void parse(char *expr) {
    char stack[MAX_SIZE];
    int top = -1;
    char output[MAX_SIZE];
    int outputIndex = 0;

    for (int i = 0; i < strlen(expr); i++) {
        char c = expr[i];

        if (c >= 'a' && c <= 'z') {
            output[outputIndex++] = c;
        } else if (c == '(') {
            stack[++top] = c;
        } else if (c == ')') {
            while (stack[top] != '(') {
                output[outputIndex++] = stack[top--];
            }
            top--;
        } else {
            while (top >= 0 && precedence(stack[top]) >= precedence(c)) {
                output[outputIndex++] = stack[top--];
            }
            stack[++top] = c;
        }
    }

    while (top >= 0) {
        output[outputIndex++] = stack[top--];
    }

    output[outputIndex] = '\0';
    printf("Postfix Expression: %s\n", output);
}

int main() {
    char expr[MAX_SIZE];
    printf("Enter an expression: ");
    scanf("%s", expr);

    parse(expr);

    return 0;
}

```

### **Output:**
```
Postfix Expression: abc*+
```
