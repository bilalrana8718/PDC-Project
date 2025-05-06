#include <iostream>
#include <vector>
#include <unordered_map>
#include <queue>
#include <limits>
#include <algorithm>
#include <omp.h>

const int INF = std::numeric_limits<int>::max();
const int ASYNC_LEVEL = 3;

using namespace std;
using Edge = pair<int,int>;            // (neighbor, weight)
using Graph = vector<vector<Edge>>;    // 1-based indexing
using Tree  = vector<vector<int>>;     // parent -> children

// 1) Dijkstra to initialize distances and parents
void initializeSSSP(const Graph &G, vector<int> &Dist, vector<int> &Parent, int source) {
    int N = G.size() - 1;
    Dist.assign(N+1, INF);
    Parent.assign(N+1, -1);
    using P = pair<int,int>;
    priority_queue<P, vector<P>, greater<P>> pq;

    Dist[source] = 0;
    pq.push({0, source});

    while (!pq.empty()) {
        auto [d,u] = pq.top(); pq.pop();
        if (d > Dist[u]) continue;
        for (auto &e : G[u]) {
            int v = e.first, w = e.second;
            if (d + w < Dist[v]) {
                Dist[v] = d + w;
                Parent[v] = u;
                pq.push({Dist[v], v});
            }
        }
    }
}

// 2) ProcessCE: mark affected_by_deletion and affected flags after deletions/insertions
void ProcessCE(
    Graph &G,
    vector<int> &Dist,
    vector<int> &Parent,
    vector<bool> &AffectedDel,
    vector<bool> &Affected,
    const vector<pair<int,int>> &Delk,
    const vector<pair<int,int>> &Insk,
    const unordered_map<long long,int> &weightMap
) {
    int N = G.size() - 1;
    // reset flags
    AffectedDel.assign(N+1, false);
    Affected.assign(N+1, false);

    // 2.1 deletions
    for (auto &e : Delk) {
        int u = e.first, v = e.second;
        // check if on tree
        if (Parent[u] == v || Parent[v] == u) {
            // pick farther endpoint y
            int y = (Dist[u] > Dist[v] ? u : v);
            Dist[y] = INF;
            Parent[y] = -1;
            AffectedDel[y] = true;
            Affected[y] = true;
        }
        // remove edge
        auto remove_edge = [&](int a, int b) {
            auto &vec = G[a];
            vec.erase(remove_if(vec.begin(), vec.end(),
                [&](const Edge &ed){ return ed.first == b; }), vec.end());
        };
        remove_edge(u,v);
        remove_edge(v,u);
    }

    // 2.2 insertions
    for (auto &e : Insk) {
        int u = e.first, v = e.second;
        long long key = ((long long)u<<32) | (unsigned long long)v;
        auto it = weightMap.find(key);
        if (it == weightMap.end()) continue;
        int w = it->second;
        int x = (Dist[u] <= Dist[v] ? u : v);
        int y = (x == u ? v : u);
        if (Dist[x] != INF && Dist[x] + w < Dist[y]) {
            Dist[y] = Dist[x] + w;
            Parent[y] = x;
            Affected[y] = true;
        }
        G[u].push_back({v,w});
        G[v].push_back({u,w});
    }
}

// 3) Asynchronous Updating (Algorithm 4)
void AsynchronousUpdating(
    const Graph &G,
    const Tree &T,
    vector<int> &Dist,
    vector<int> &Parent,
    vector<bool> &Affected,
    vector<bool> &AffectedDel
) {
    bool Change = true;
    int N = G.size() - 1;

    // Phase 1: propagate deletions down the tree
    while (Change) {
        Change = false;
        #pragma omp parallel for schedule(dynamic)
        for (int v = 1; v <= N; ++v) {
            if (!AffectedDel[v]) continue;
            AffectedDel[v] = false;
            queue<pair<int,int>> Q;
            Q.push({v,0});
            while (!Q.empty()) {
                auto [x, level] = Q.front(); Q.pop();
                for (int c : T[x]) {
                    #pragma omp critical
                    {
                        Dist[c] = INF;
                        AffectedDel[c] = true;
                        Affected[c] = true;
                        Change = true;
                    }
                    if (level + 1 <= ASYNC_LEVEL)
                        Q.push({c, level+1});
                }
            }
        }
    }

    // Phase 2: re-relaxation asynchronously
    Change = true;
    while (Change) {
        Change = false;
        #pragma omp parallel for schedule(dynamic)
        for (int v = 1; v <= N; ++v) {
            if (!Affected[v]) continue;
            Affected[v] = false;
            queue<pair<int,int>> Q;
            Q.push({v,0});
            while (!Q.empty()) {
                auto [x, level] = Q.front(); Q.pop();
                for (auto &ed : G[x]) {
                    int n = ed.first, w = ed.second;
                    bool local_change = false;
                    #pragma omp critical
                    {
                        if (Dist[x] != INF && Dist[x] + w < Dist[n]) {
                            Dist[n] = Dist[x] + w;
                            Parent[n] = x;
                            Affected[n] = true;
                            Change = true;
                            local_change = true;
                        }
                        if (Dist[n] != INF && Dist[n] + w < Dist[x]) {
                            Dist[x] = Dist[n] + w;
                            Parent[x] = n;
                            Affected[x] = true;
                            Change = true;
                            local_change = true;
                        }
                    }
                    if (local_change && level+1 <= ASYNC_LEVEL) {
                        Q.push({x, level+1});
                        Q.push({n, level+1});
                    }
                }
            }
        }
    }
}

// Helper to build tree from Parent array
Tree buildTree(const vector<int> &Parent) {
    int N = Parent.size() - 1;
    Tree T(N+1);
    for (int v = 1; v <= N; ++v) {
        int p = Parent[v];
        if (p != -1) T[p].push_back(v);
    }
    return T;
}

// Print distances and parents
void printState(const vector<int> &Dist, const vector<int> &Parent) {
    cout << "Node | Dist | Parent\n";
    cout << "-------------------\n";
    for (int i = 1; i < Dist.size(); ++i) {
        cout << "  " << i
             << "  | " << (Dist[i]==INF? string("INF") : to_string(Dist[i]))
             << "  |   " << Parent[i] << "\n";
    }
    cout << "\n";
}

int main() {
    const int N = 5;
    Graph G(N+1);
    auto addEdge = [&](int u,int v,int w){
        G[u].push_back({v,w});
        G[v].push_back({u,w});
    };
    addEdge(1,2,4);
    addEdge(1,3,2);
    addEdge(2,3,5);
    addEdge(2,4,10);
    addEdge(3,5,3);
    addEdge(5,4,4);

    // record weights for insertions
    unordered_map<long long,int> weightMap;
    auto recW = [&](int u,int v,int w){
        long long k1 = ((long long)u<<32)|(unsigned long long)v;
        long long k2 = ((long long)v<<32)|(unsigned long long)u;
        weightMap[k1] = w;
        weightMap[k2] = w;
    };
    recW(1,2,4); recW(1,3,2);
    recW(2,3,5); recW(2,4,10);
    recW(3,5,3); recW(5,4,4);
    recW(4,5,1); // for insertion

    vector<int> Dist, Parent;
    initializeSSSP(G, Dist, Parent, /*source=*/1);
    cout << "--- Initial SSSP ---\n";
    printState(Dist, Parent);

    // changes: delete (2,3),(5,4) and insert (4,5)
    vector<pair<int,int>> Delk = {{2,3},{5,4}};
    vector<pair<int,int>> Insk = {{4,5}};

    vector<bool> AffectedDel, Affected;
    ProcessCE(G, Dist, Parent, AffectedDel, Affected, Delk, Insk, weightMap);
    cout << "--- After ProcessCE (Alg.2) ---\n";
    printState(Dist, Parent);

    Tree T = buildTree(Parent);
    AsynchronousUpdating(G, T, Dist, Parent, Affected, AffectedDel);
    cout << "--- After AsynchronousUpdating (Alg.4) ---\n";
    printState(Dist, Parent);

    return 0;
}
