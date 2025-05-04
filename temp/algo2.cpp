// algo23_fixed.cpp
#include <iostream>
#include <vector>
#include <unordered_map>
#include <queue>
#include <limits>
#include <algorithm>
#include <set>

using namespace std;
const int INF = numeric_limits<int>::max();

struct Edge {
    int to, weight;
};

struct Node {
    int name;
    int parent;
    int distance;
    bool affected_by_deletion = false;
    bool affected = false;
};

using Graph   = unordered_map<int, vector<Edge>>;
using NodeMap = unordered_map<int, Node>;

// Dijkstra to initialize Dist & Parent
void initializeSSSP(const Graph &G, NodeMap &nodes, int source) {
    using P = pair<int,int>;
    priority_queue<P, vector<P>, greater<P>> pq;
    for (auto &kv : nodes) {
        kv.second.distance = INF;
        kv.second.parent   = -1;
        kv.second.affected = false;
    }
    nodes[source].distance = 0;
    pq.push({0, source});

    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > nodes.at(u).distance) continue;
        for (auto &e : G.at(u)) {
            int v = e.to, w = e.weight;
            if (d + w < nodes.at(v).distance) {
                nodes[v].distance = d + w;
                nodes[v].parent   = u;
                pq.push({nodes[v].distance, v});
            }
        }
    }
}

bool inTree(int u, int v, const NodeMap &nodes) {
    return nodes.at(v).parent == u
        || nodes.at(u).parent == v;
}

// Algorithm 2: mark deletions & potential insertions
void ProcessCE(Graph &G, NodeMap &nodes,
               const vector<pair<int,int>> &Delk,
               const vector<pair<int,int>> &Insk,
               const unordered_map<long long,int> &weightMap) {
    // reset flags
    for (auto &kv : nodes) {
        kv.second.affected_by_deletion = false;
        kv.second.affected            = false;
    }

    Graph Gu = G;

    // 1) deletions
    for (auto [u,v] : Delk) {
        if (inTree(u,v,nodes)) {
            int y = (nodes.at(u).distance > nodes.at(v).distance ? u : v);
            nodes[y].distance            = INF;
            nodes[y].parent              = -1;
            nodes[y].affected_by_deletion = true;
            nodes[y].affected            = true;
        }
        auto &vecU = Gu[u];
        vecU.erase(remove_if(vecU.begin(), vecU.end(),
                             [&](const Edge &e){ return e.to==v; }),
                   vecU.end());
        auto &vecV = Gu[v];
        vecV.erase(remove_if(vecV.begin(), vecV.end(),
                             [&](const Edge &e){ return e.to==u; }),
                   vecV.end());
    }

    // 2) insertions
    for (auto [u,v] : Insk) {
        // **Corrected key type to long long**
        long long key = ((long long)u << 32) | (unsigned long long)v;
        auto it = weightMap.find(key);
        if (it == weightMap.end()) {
            cerr << "Error: no weight recorded for insertion (" 
                 << u << "," << v << ")\n";
            continue;
        }
        int w = it->second;

        int x = (nodes.at(u).distance <= nodes.at(v).distance ? u : v);
        int y = (x == u ? v : u);

        if (nodes.at(x).distance != INF &&
            nodes.at(x).distance + w < nodes.at(y).distance) {
            nodes[y].distance = nodes.at(x).distance + w;
            nodes[y].parent   = x;
            nodes[y].affected = true;
        }
        Gu[u].push_back({v,w});
        Gu[v].push_back({u,w});
    }

    G.swap(Gu);
}

// Algorithm 3: propagate deletions & re-relaxations
void UpdateAffectedVertices(const Graph &Gu, NodeMap &nodes) {
    unordered_map<int, vector<int>> children;
    for (auto &kv : nodes) {
        int v = kv.first, p = kv.second.parent;
        if (p != -1) children[p].push_back(v);
    }

    bool moreDel = true;
    while (moreDel) {
        moreDel = false;
        for (auto &kv : nodes) {
            Node &vn = kv.second;
            if (vn.affected_by_deletion) {
                vn.affected_by_deletion = false;
                moreDel = true;
                for (int c : children[vn.name]) {
                    Node &cn = nodes.at(c);
                    cn.distance = INF;
                    cn.parent   = -1;
                    cn.affected_by_deletion = true;
                    cn.affected = true;
                }
            }
        }
    }

    bool moreAff = true;
    while (moreAff) {
        moreAff = false;
        for (auto &kv : nodes) {
            Node &vn = kv.second;
            if (vn.affected) {
                vn.affected = false;
                moreAff = true;
                int v = vn.name;
                for (auto &e : Gu.at(v)) {
                    int n = e.to, w = e.weight;
                    Node &nn = nodes.at(n);
                    // v → n
                    if (vn.distance != INF &&
                        vn.distance + w < nn.distance) {
                        nn.distance = vn.distance + w;
                        nn.parent   = v;
                        nn.affected = true;
                    }
                    // n → v
                    if (nn.distance != INF &&
                        nn.distance + w < vn.distance) {
                        vn.distance = nn.distance + w;
                        vn.parent   = n;
                        vn.affected = true;
                    }
                }
            }
        }
    }
}

void printNodes(const NodeMap &nodes) {
    cout << "Node | Dist  | Parent | Del? | Aff?\n"
         << "-------------------------------\n";
    for (auto &kv : nodes) {
        const Node &n = kv.second;
        cout << "  " << n.name
             << "   | " << (n.distance==INF ? "INF" : to_string(n.distance))
             << "   |   " << n.parent
             << "   |   " << (n.affected_by_deletion?'Y':'N')
             << "   |   " << (n.affected?'Y':'N')
             << "\n";
    }
    cout << "\n";
}

int main() {
    Graph G;
    for (int u = 1; u <= 5; ++u) G[u] = {};
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
    recW(4,5,1); // insertion weight

    NodeMap nodes;
    for (int u = 1; u <= 5; ++u)
        nodes[u] = Node{u, -1, INF, false, false};

    cout << "--- Initial SSSP ---\n";
    initializeSSSP(G, nodes, 1);
    printNodes(nodes);

    vector<pair<int,int>> Delk = {{5,4}};
    // vector<pair<int,int>> Insk = {{4,3}};
    vector<pair<int,int>> Insk = {};

    cout << "--- After ProcessCE (Alg.2) ---\n";
    ProcessCE(G, nodes, Delk, Insk, weightMap);
    printNodes(nodes);

    cout << "--- After UpdateAffectedVertices (Alg.3) ---\n";
    UpdateAffectedVertices(G, nodes);
    printNodes(nodes);

    return 0;
}
