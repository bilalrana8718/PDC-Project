// algo2_partitioned.cpp
#include <iostream>
#include <vector>
#include <unordered_map>
#include <queue>
#include <limits>
#include <algorithm>
#include <set>
#include <fstream>
#include <string>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <map>
#include <functional>

using namespace std;
namespace fs = std::filesystem;
const int INF = numeric_limits<int>::max();

// Structure definitions
struct Edge {
    int to, weight;
};

struct Node {
    int name;
    int parent;
    int distance;
    bool affected_by_deletion = false;
    bool affected = false;
    int partition = -1; // Store the partition this node belongs to
};

using Graph = unordered_map<int, vector<Edge>>;
using NodeMap = unordered_map<int, Node>;
using PartitionMap = unordered_map<int, vector<int>>;

// Visualize the SSSP tree
void visualizeSSPTree(const NodeMap &nodes, const string& phase = "Current") {
    cout << "\n== SSSP Tree Visualization (" << phase << ") ==\n\n";
    
    // First, build an adjacency list from parent pointers
    map<int, vector<int>> tree;
    int root = -1;
    
    // Find root and build tree structure
    for (const auto& [node_id, node] : nodes) {
        if (node.parent == -1 && node.distance == 0) {
            root = node_id;
        } else if (node.parent != -1) {
            tree[node.parent].push_back(node_id);
        }
    }
    
    if (root == -1) {
        cout << "No root found in the SSSP tree!\n";
        return;
    }
    
    // Function to print tree recursively with indentation
    std::function<void(int, const string&, int)> printTree = [&](int node, const string& prefix, int depth) {
        string indent = prefix;
        for (int i = 0; i < depth; i++) indent += "  ";
        
        const Node& n = nodes.at(node);
        cout << indent << node 
             << " [dist=" << (n.distance == INF ? "INF" : to_string(n.distance))
             << ", part=" << n.partition;
             
        if (n.affected_by_deletion)
            cout << ", DEL";
        if (n.affected)
            cout << ", AFF";
            
        cout << "]\n";
        
        // Sort children for consistent visualization
        vector<int> children = tree[node];
        sort(children.begin(), children.end());
        
        for (size_t i = 0; i < children.size(); i++) {
            bool isLast = (i == children.size() - 1);
            string childPrefix = isLast ? "└── " : "├── ";
            string nextPrefix = isLast ? "    " : "│   ";
            printTree(children[i], indent + childPrefix, 0);
            
            // Add additional children with proper indentation
            vector<int> grandchildren = tree[children[i]];
            sort(grandchildren.begin(), grandchildren.end());
            for (size_t j = 0; j < grandchildren.size(); j++) {
                bool isLastGrand = (j == grandchildren.size() - 1);
                printTree(grandchildren[j], indent + nextPrefix + (isLastGrand ? "└── " : "├── "), 0);
            }
        }
    };
    
    // Start printing from the root
    printTree(root, "", 0);
    cout << endl;
}

// Print statistics about the SSSP tree
void printTreeStats(const NodeMap &nodes) {
    int total_nodes = 0;
    int nodes_in_tree = 0;
    int affected_nodes = 0;
    int deleted_nodes = 0;
    int inf_distance_nodes = 0;
    
    for (const auto& [node_id, node] : nodes) {
        total_nodes++;
        if (node.parent != -1 || node.distance == 0) nodes_in_tree++;
        if (node.affected) affected_nodes++;
        if (node.affected_by_deletion) deleted_nodes++;
        if (node.distance == INF) inf_distance_nodes++;
    }
    
    cout << "\n== SSSP Tree Statistics ==\n";
    cout << "Total nodes: " << total_nodes << "\n";
    cout << "Nodes in tree: " << nodes_in_tree << "\n";
    cout << "Affected nodes: " << affected_nodes << "\n";
    cout << "Deleted nodes: " << deleted_nodes << "\n";
    cout << "Infinite distance nodes: " << inf_distance_nodes << "\n";
    cout << "Connected percentage: " << fixed << setprecision(2) 
         << (100.0 * nodes_in_tree / total_nodes) << "%\n\n";
}

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
    return nodes.at(v).parent == u || nodes.at(u).parent == v;
}

// Algorithm 2: mark deletions & potential insertions
void ProcessCE(Graph &G, NodeMap &nodes,
               const vector<pair<int,int>> &Delk,
               const vector<pair<int,int>> &Insk,
               const unordered_map<long long,int> &weightMap) {
    // reset flags
    for (auto &kv : nodes) {
        kv.second.affected_by_deletion = false;
        kv.second.affected = false;
    }

    Graph Gu = G;

    // 1) deletions
    for (auto [u,v] : Delk) {
        if (inTree(u,v,nodes)) {
            int y = (nodes.at(u).distance > nodes.at(v).distance ? u : v);
            nodes[y].distance = INF;
            nodes[y].parent = -1;
            nodes[y].affected_by_deletion = true;
            nodes[y].affected = true;
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
        // Use long long for the key
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

void printNodes(const NodeMap &nodes, bool show_details = true) {
    if (show_details) {
        cout << "Node | Partition | Dist  | Parent | Del? | Aff?\n"
             << "-----------------------------------------------\n";
        for (auto &kv : nodes) {
            const Node &n = kv.second;
            cout << setw(4) << n.name
                 << " | " << setw(9) << n.partition
                 << " | " << setw(5) << (n.distance==INF ? "INF" : to_string(n.distance))
                 << " | " << setw(6) << n.parent
                 << " | " << setw(4) << (n.affected_by_deletion?'Y':'N')
                 << " | " << setw(4) << (n.affected?'Y':'N')
                 << "\n";
        }
    } else {
        cout << "Processed " << nodes.size() << " nodes" << endl;
    }
    cout << "\n";
}

// Load graph from an edge list file (graph.txt)
bool loadGraphFromEdgeList(const string& filename, Graph& G, unordered_map<long long, int>& weightMap) {
    ifstream infile(filename);
    if (!infile) {
        cerr << "Error: Unable to open input file " << filename << endl;
        return false;
    }

    cout << "Loading graph from " << filename << "..." << endl;
    string line;
    long count = 0;

    while (getline(infile, line)) {
        count++;
        if (count % 1000000 == 0) {
            cout << "Processed " << count/1000000 << "M lines" << endl;
        }

        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;

        istringstream iss(line);
        int u, v;
        double weight = 1.0; // Default weight

        if (!(iss >> u >> v >> weight)) {
            if (!(iss >> u >> v)) {
                continue; // Skip malformed lines
            }
        }

        // Add edge to graph
        G[u].push_back({v, static_cast<int>(weight)});
        G[v].push_back({u, static_cast<int>(weight)}); // Undirected graph

        // Store weight in map
        long long k1 = ((long long)u << 32) | (unsigned long long)v;
        long long k2 = ((long long)v << 32) | (unsigned long long)u;
        weightMap[k1] = weightMap[k2] = static_cast<int>(weight);
    }

    cout << "\nGraph loaded with " << G.size() << " nodes and approximately " 
         << count << " edges" << endl;
    return true;
}

// Load partitions from a METIS partition file
bool loadPartitions(const string& partition_file, NodeMap& nodes, PartitionMap& partitions) {
    ifstream infile(partition_file);
    if (!infile) {
        cerr << "Error: Unable to open partition file " << partition_file << endl;
        return false;
    }

    cout << "Loading partitions from " << partition_file << "..." << endl;
    string line;
    int node_id = 0; // METIS uses 0-based indexing in the partition file
    
    while (getline(infile, line)) {
        int partition_id;
        istringstream iss(line);
        if (!(iss >> partition_id)) {
            cerr << "Error parsing partition ID on line " << node_id + 1 << endl;
            continue;
        }
        
        // Update node's partition information
        if (nodes.find(node_id) != nodes.end()) {
            nodes[node_id].partition = partition_id;
            partitions[partition_id].push_back(node_id);
        }
        
        node_id++;
    }
    
    cout << "Loaded partitions for " << node_id << " nodes into " 
         << partitions.size() << " partitions" << endl;
    return true;
}

// Generate cross-partition edge operations (deletions and insertions)
void generateCrossPartitionOperations(
    const Graph& G, 
    const NodeMap& nodes, 
    vector<pair<int,int>>& Delk, 
    vector<pair<int,int>>& Insk) {
    
    Delk.clear();
    Insk.clear();
    
    // Find edges that cross partitions and mark them for deletion
    for (const auto& [u, edges] : G) {
        if (nodes.find(u) == nodes.end()) continue;
        int u_partition = nodes.at(u).partition;
        
        for (const auto& edge : edges) {
            int v = edge.to;
            if (nodes.find(v) == nodes.end()) continue;
            int v_partition = nodes.at(v).partition;
            
            // If nodes are in different partitions, mark edge for deletion
            if (u_partition != v_partition && u < v) { // Avoid duplicates with u < v check
                Delk.push_back({u, v});
            }
        }
    }
    
    cout << "Generated " << Delk.size() << " cross-partition edge deletions" << endl;
}

// Process partitioned graph with the SSSP algorithm
void processPartitionedGraph(
    const string& graph_file,
    const string& partition_file,
    int source_node,
    bool show_full_tree = false,
    int max_tree_nodes = 50) {
    
    auto start_time = chrono::high_resolution_clock::now();
    
    // Load the graph
    Graph G;
    unordered_map<long long, int> weightMap;
    if (!loadGraphFromEdgeList(graph_file, G, weightMap)) {
        return;
    }
    
    // Initialize node map
    NodeMap nodes;
    for (const auto& [node_id, _] : G) {
        nodes[node_id] = Node{node_id, -1, INF, false, false, -1};
    }
    
    // Load partitions
    PartitionMap partitions;
    if (!loadPartitions(partition_file, nodes, partitions)) {
        return;
    }
    
    // Print partition statistics
    cout << "\nPartition Statistics:" << endl;
    cout << "--------------------" << endl;
    for (const auto& [part_id, nodes_in_part] : partitions) {
        cout << "Partition " << part_id << ": " << nodes_in_part.size() << " nodes" << endl;
    }
    
    // Initialize SSSP on the complete graph
    cout << "\nRunning initial SSSP from source node " << source_node << endl;
    auto sssp_start = chrono::high_resolution_clock::now();
    initializeSSSP(G, nodes, source_node);
    auto sssp_end = chrono::high_resolution_clock::now();
    auto sssp_duration = chrono::duration_cast<chrono::milliseconds>(sssp_end - sssp_start).count();
    cout << "Initial SSSP completed in " << sssp_duration << " ms" << endl;
    
    // Create a smaller subset of nodes for visualization if needed
    NodeMap display_nodes;
    if (!show_full_tree && nodes.size() > max_tree_nodes) {
        // Include the source node
        display_nodes[source_node] = nodes[source_node];
        
        // Add some nodes from each partition for diversity
        int nodes_added = 1; // Already added source
        for (const auto& [part_id, nodes_in_part] : partitions) {
            int added_from_part = 0;
            int to_add_from_part = std::max(1, static_cast<int>(max_tree_nodes / partitions.size()));
            
            for (int node_id : nodes_in_part) {
                if (node_id != source_node && added_from_part < to_add_from_part) {
                    display_nodes[node_id] = nodes[node_id];
                    added_from_part++;
                    nodes_added++;
                }
                
                if (nodes_added >= max_tree_nodes) break;
            }
            
            if (nodes_added >= max_tree_nodes) break;
        }
        
        cout << "\nNote: Showing a subset of " << display_nodes.size() << " nodes for visualization" << endl;
        cout << "Use --show-full-tree option to see the complete tree" << endl;
        
        // Visualize initial SSSP tree for the subset
        visualizeSSPTree(display_nodes, "Initial SSSP");
    } else {
        // Visualize the full tree
        visualizeSSPTree(nodes, "Initial SSSP");
    }
    
    // Print tree statistics
    printTreeStats(nodes);
    
    // Generate operations for cross-partition edges
    vector<pair<int,int>> Delk, Insk;
    generateCrossPartitionOperations(G, nodes, Delk, Insk);
    
    // Process edge operations (Algorithm 2)
    cout << "\nProcessing cross-partition edge operations (Algorithm 2)..." << endl;
    auto alg2_start = chrono::high_resolution_clock::now();
    ProcessCE(G, nodes, Delk, Insk, weightMap);
    auto alg2_end = chrono::high_resolution_clock::now();
    auto alg2_duration = chrono::duration_cast<chrono::milliseconds>(alg2_end - alg2_start).count();
    cout << "Algorithm 2 completed in " << alg2_duration << " ms" << endl;
    
    // Update the display nodes if using subset
    if (!show_full_tree && nodes.size() > max_tree_nodes) {
        for (auto& [node_id, _] : display_nodes) {
            display_nodes[node_id] = nodes[node_id];
        }
        visualizeSSPTree(display_nodes, "After Algorithm 2");
    } else {
        visualizeSSPTree(nodes, "After Algorithm 2");
    }
    
    printTreeStats(nodes);
    
    // Update affected vertices (Algorithm 3)
    cout << "\nUpdating affected vertices (Algorithm 3)..." << endl;
    auto alg3_start = chrono::high_resolution_clock::now();
    UpdateAffectedVertices(G, nodes);
    auto alg3_end = chrono::high_resolution_clock::now();
    auto alg3_duration = chrono::duration_cast<chrono::milliseconds>(alg3_end - alg3_start).count();
    cout << "Algorithm 3 completed in " << alg3_duration << " ms" << endl;
    
    // Update the display nodes if using subset
    if (!show_full_tree && nodes.size() > max_tree_nodes) {
        for (auto& [node_id, _] : display_nodes) {
            display_nodes[node_id] = nodes[node_id];
        }
        visualizeSSPTree(display_nodes, "After Algorithm 3");
    } else {
        visualizeSSPTree(nodes, "After Algorithm 3");
    }
    
    printTreeStats(nodes);
    
    // Print summary
    auto end_time = chrono::high_resolution_clock::now();
    auto total_duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    
    cout << "\nSummary:" << endl;
    cout << "--------" << endl;
    cout << "Total execution time: " << total_duration << " ms" << endl;
    cout << "Initial SSSP: " << sssp_duration << " ms" << endl;
    cout << "Algorithm 2: " << alg2_duration << " ms" << endl;
    cout << "Algorithm 3: " << alg3_duration << " ms" << endl;
}

int main(int argc, char* argv[]) {
    // Default parameters
    string graph_file = "graph.txt";
    string partition_file = "graph.metis.part.4"; // Default from metis.cpp with 4 partitions
    int source_node = 0;
    bool show_full_tree = false;
    int max_tree_nodes = 50;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-g" || arg == "--graph") {
            if (i + 1 < argc) graph_file = argv[++i];
        } else if (arg == "-p" || arg == "--partition") {
            if (i + 1 < argc) partition_file = argv[++i];
        } else if (arg == "-s" || arg == "--source") {
            if (i + 1 < argc) source_node = atoi(argv[++i]);
        } else if (arg == "-n" || arg == "--nodes") {
            if (i + 1 < argc) max_tree_nodes = atoi(argv[++i]);
        } else if (arg == "--show-full-tree") {
            show_full_tree = true;
        } else if (arg == "-h" || arg == "--help") {
            cout << "Usage: " << argv[0] << " [options]" << endl;
            cout << "Options:" << endl;
            cout << "  -g, --graph FILE        Input graph file (default: graph.txt)" << endl;
            cout << "  -p, --partition FILE    METIS partition file (default: graph.metis.part.4)" << endl;
            cout << "  -s, --source NODE       Source node for SSSP (default: 0)" << endl;
            cout << "  -n, --nodes COUNT       Max nodes to display in tree visualization (default: 50)" << endl;
            cout << "  --show-full-tree        Show the complete SSSP tree (may be very large)" << endl;
            cout << "  -h, --help              Display this help message" << endl;
            return 0;
        }
    }
    
    cout << "Partitioned Graph SSSP Algorithm" << endl;
    cout << "===============================" << endl;
    cout << "Graph file: " << graph_file << endl;
    cout << "Partition file: " << partition_file << endl;
    cout << "Source node: " << source_node << endl;
    cout << "Tree visualization: " << (show_full_tree ? "Full tree" : "Limited to " + to_string(max_tree_nodes) + " nodes") << endl;
    
    processPartitionedGraph(graph_file, partition_file, source_node, show_full_tree, max_tree_nodes);
    
    return 0;
} 