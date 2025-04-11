### Merge Sort
```c
#include <stdio.h>
#include <time.h>

void merge(int arr[], int l, int m, int r) {
    int n1 = m - l + 1, n2 = r - m;
    int L[n1], R[n2];
    for (int i = 0; i < n1; i++) L[i] = arr[l + i];
    for (int j = 0; j < n2; j++) R[j] = arr[m + 1 + j];

    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2) arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
}

void mergeSort(int arr[], int l, int r) {
    if (l < r) {
        int m = (l + r) / 2;
        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

int main() {
    int arr[] = {12, 11, 13, 5, 6, 7}, n = 6;
    clock_t start = clock();
    mergeSort(arr, 0, n - 1);
    clock_t end = clock();

    printf("Sorted array: ");
    for (int i = 0; i < n; i++) printf("%d ", arr[i]);
    printf("\nTime: %lf sec\n", (double)(end - start)/CLOCKS_PER_SEC);
    return 0;
}

```

**output**
```
Sorted array: 5 6 7 11 12 13 
Time: 0.000003 sec
```

---

### 2. Hamiltonian Cycle (Backtracking)

```c
#include <stdio.h>
#include <stdbool.h>
#include <time.h>

#define V 5

int graph[V][V] = {
    {0, 1, 0, 1, 0},
    {1, 0, 1, 1, 1},
    {0, 1, 0, 0, 1},
    {1, 1, 0, 0, 1},
    {0, 1, 1, 1, 0}
};

int path[V];

bool isSafe(int v, int pos) {
    if (!graph[path[pos - 1]][v]) return false;
    for (int i = 0; i < pos; i++)
        if (path[i] == v) return false;
    return true;
}

bool hamCycleUtil(int pos) {
    if (pos == V) {
        return graph[path[pos - 1]][path[0]] == 1;
    }

    for (int v = 1; v < V; v++) {
        if (isSafe(v, pos)) {
            path[pos] = v;
            if (hamCycleUtil(pos + 1)) return true;
            path[pos] = -1;
        }
    }
    return false;
}

void hamCycle() {
    for (int i = 0; i < V; i++) path[i] = -1;
    path[0] = 0;

    if (hamCycleUtil(1)) {
        printf("Cycle Exists: ");
        for (int i = 0; i < V; i++) printf("%d ", path[i]);
        printf("%d\n", path[0]);
    } else {
        printf("No Hamiltonian Cycle\n");
    }
}

int main() {
    clock_t start = clock();
    hamCycle();
    clock_t end = clock();
    printf("Time: %lf sec\n", (double)(end - start)/CLOCKS_PER_SEC);
    return 0;
}

```

**output**
```
Cycle Exists: 0 1 2 4 3 0
Time: 0.000057 sec
```
---

### 3. Kruskal’s Algorithm (Greedy Method)

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX 30

struct Edge {
    int u, v, w;
};

int parent[MAX];

int find(int i) {
    while (parent[i]) i = parent[i];
    return i;
}

int union_set(int i, int j) {
    if (i != j) {
        parent[j] = i;
        return 1;
    }
    return 0;
}

int main() {
    int n = 4, edges = 5, u, v, i, j;
    struct Edge edge[] = {{0,1,10}, {0,2,6}, {0,3,5}, {1,3,15}, {2,3,4}};
    struct Edge temp;
    clock_t start = clock();

    // Sort edges
    for (i = 0; i < edges-1; i++)
        for (j = 0; j < edges-i-1; j++)
            if (edge[j].w > edge[j+1].w) {
                temp = edge[j];
                edge[j] = edge[j+1];
                edge[j+1] = temp;
            }

    printf("Edges in MST:\n");
    int count = 0, cost = 0;
    for (i = 0; i < edges; i++) {
        u = find(edge[i].u);
        v = find(edge[i].v);
        if (union_set(u, v)) {
            printf("%d - %d : %d\n", edge[i].u, edge[i].v, edge[i].w);
            cost += edge[i].w;
            count++;
            if (count == n - 1) break;
        }
    }

    printf("Total Cost: %d\n", cost);
    clock_t end = clock();
    printf("Time: %lf sec\n", (double)(end - start)/CLOCKS_PER_SEC);
    return 0;
}

```

**output**
```
Edges in MST:
2 - 3 : 4
0 - 3 : 5
0 - 2 : 6
Total Cost: 15
Time: 0.000073 sec
```

---
