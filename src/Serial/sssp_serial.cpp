#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <set>
#include <algorithm>  
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath>

using namespace std;

const double INF = numeric_limits<double>::infinity();

struct Edge {
    int src, dest;
    double weight;
    Edge(int s, int d, double w) : src(s), dest(d), weight(w) {}
};

struct Graph {
    int num_nodes;
    int num_edges;
    vector<vector<pair<int, double>>> adj_list;
    Graph(int n) : num_nodes(n), num_edges(0) {
        adj_list.resize(n);
    }
};

struct SSSPData {
    vector<double> distances;
    vector<int> parents;
    vector<bool> is_affected;
    
    SSSPData(int size) {
        distances.resize(size, INF);
        parents.resize(size, -1);
        is_affected.resize(size, false);
    }
};

struct EdgeUpdate {
    int src, dest;
    double weight;
    bool is_insertion;  // true for insertion, false for deletion
    EdgeUpdate() : src(-1), dest(-1), weight(0.0), is_insertion(false) {}  // Default constructor
    EdgeUpdate(int s, int d, double w, bool ins) 
        : src(s), dest(d), weight(w), is_insertion(ins) {}
};

Graph read_graph(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }

    set<int> node_indices;
    int m = 0;
    int u, v;
    double weight;
    while (file >> u >> v >> weight) {
        node_indices.insert(u);
        node_indices.insert(v);
        m++;
    }
    
    unordered_map<int, int> node_map;
    int new_index = 0;
    for (int node : node_indices) {
        node_map[node] = new_index++;
    }
    
    file.clear();
    file.seekg(0);
    
    Graph graph(node_indices.size());
    graph.num_edges = m;

    while (file >> u >> v >> weight) {
        int mapped_u = node_map[u];
        int mapped_v = node_map[v];
        
        if (mapped_u >= graph.num_nodes || mapped_v >= graph.num_nodes || 
            mapped_u < 0 || mapped_v < 0) {
            cerr << "Error: Mapped node index out of bounds: " << mapped_u << " or " << mapped_v 
                 << " (n=" << graph.num_nodes << ")" << endl;
            exit(1);
        }
        graph.adj_list[mapped_u].emplace_back(mapped_v, weight);
        graph.adj_list[mapped_v].emplace_back(mapped_u, weight); // Undirected graph
    }

    file.close();
    return graph;
}

void print_graph(const Graph& graph) {
    cout << "\n-----------------------------------------" << endl;
    cout << "Graph Structure" << endl;
    cout << "-----------------------------------------" << endl;
    cout << "Total Nodes: " << graph.num_nodes << ", Total Edges: " << graph.num_edges << endl;
    cout << "-----------------------------------------" << endl;
    cout << "| Node | Connected To (Node, Weight) |" << endl;
    cout << "-----------------------------------------" << endl;
    
    for (int i = 0; i < graph.num_nodes; i++) {
        cout << "| " << setw(4) << i << " | ";
        if (graph.adj_list[i].empty()) {
            cout << "None";
        } else {
            bool first = true;
            for (const auto& edge : graph.adj_list[i]) {
                if (!first) cout << ", ";
                cout << "(" << edge.first << ", " << edge.second << ")";
                first = false;
            }
        }
        cout << " |" << endl;
    }
    cout << "-----------------------------------------" << endl;
}

Graph read_metis_graph(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening METIS file: " << filename << endl;
        exit(1);
    }

    string header;
    getline(file, header);
    istringstream header_ss(header);
    
    int num_nodes, num_edges;
    string format_info;
    
    header_ss >> num_nodes >> num_edges;
    
    bool weighted = false;
    if (header_ss >> format_info && format_info == "001") {
        weighted = true;
    }
    
    cout << "Reading METIS file with " << num_nodes << " nodes and " 
         << num_edges << " edges, weighted=" << weighted << endl;
    
    Graph graph(num_nodes);
    graph.num_edges = num_edges * 2; 
    
    for (int i = 0; i < num_nodes; i++) {
        string line;
        if (!getline(file, line)) {
            cerr << "Error reading adjacency list for node " << i << endl;
            exit(1);
        }
        
        istringstream line_ss(line);
        int neighbor;
        double weight;
        
        while (line_ss >> neighbor) {
            neighbor--;
            
            if (weighted) {
                if (!(line_ss >> weight)) {
                    cerr << "Error reading weight for edge (" << i << ", " << neighbor << ")" << endl;
                    exit(1);
                }
            } else {
                weight = 1.0; 
            }
            
            if (0 <= neighbor && neighbor < num_nodes) {
                graph.adj_list[i].emplace_back(neighbor, weight);
            } else {
                cerr << "Error: Invalid neighbor index " << neighbor+1 
                     << " for node " << i << " (must be 1-" << num_nodes << ")" << endl;
                exit(1);
            }
        }
    }
    
    file.close();
    return graph;
}

// Main SSSP computation function (serial version)
void compute_sssp(const Graph& graph, SSSPData& sssp_data, int source) {
    cout << "Initializing SSSP with source vertex: " << source << endl;
    
    // Initialize all distances to infinity and parents to -1
    for (int i = 0; i < graph.num_nodes; i++) {
        sssp_data.distances[i] = INF;
        sssp_data.parents[i] = -1;
    }
    
    sssp_data.distances[source] = 0.0;
    sssp_data.parents[source] = source;
    cout << "Setting distance[" << source << "] = 0.0" << endl;
    
    cout << "Starting iterative SSSP computation..." << endl;
    
    vector<double> new_distances(graph.num_nodes, INF);
    vector<int> new_parents(graph.num_nodes, -1);
    
    bool changes;
    int iterations = 0;
    const int MAX_ITERATIONS = graph.num_nodes;
    
    do {
        iterations++;
        if (iterations % 5 == 0) {
            cout << "SSSP iteration " << iterations << "..." << endl;
        }
        
        for (int i = 0; i < graph.num_nodes; i++) {
            new_distances[i] = sssp_data.distances[i];
            new_parents[i] = sssp_data.parents[i];
        }
        
        changes = false;
        int relaxations = 0;
        
        // Process vertices (edge relaxation)
        for (int u = 0; u < graph.num_nodes; u++) {
            if (sssp_data.distances[u] == INF) continue;
            
            for (const auto& edge : graph.adj_list[u]) {
                int v = edge.first;
                double weight = edge.second;
                double potential_dist = sssp_data.distances[u] + weight;
                
                // Relaxation: If we found a shorter path to v
                if (potential_dist < new_distances[v]) {
                    new_distances[v] = potential_dist;
                    new_parents[v] = u;
                    changes = true;
                    relaxations++;
                    
                    if (v == u || (new_parents[v] == u && new_parents[u] == v)) {
                        cout << "[Warning] Potential cycle detected between " << u << " and " << v << endl;
                    }
                }
            }
        }
        
        if (relaxations > 0 && iterations % 5 == 0) {
            cout << "Total relaxations in iteration " << iterations 
                 << ": " << relaxations << endl;
        }
        
        if (changes) {
            for (int i = 0; i < graph.num_nodes; i++) {
                sssp_data.distances[i] = new_distances[i];
                sssp_data.parents[i] = new_parents[i];
            }
        }
        
        // Enforce a maximum number of iterations to prevent infinite loops
        if (iterations >= MAX_ITERATIONS) {
            cout << "Warning: SSSP computation reached maximum iterations (" 
                 << MAX_ITERATIONS << "). There might be negative cycles in the graph." << endl;
            break;
        }
        
    } while (changes);
    
    cout << "SSSP computation converged after " << iterations << " iterations." << endl;
    
    bool has_cycle = false;
    for (int i = 0; i < graph.num_nodes; i++) {
        if (i != source && sssp_data.parents[i] != -1) {
            if (i == sssp_data.parents[i]) {
                cout << "Error: Node " << i << " is its own parent!" << endl;
                has_cycle = true;
            }
            
            int parent = sssp_data.parents[i];
            if (parent != -1 && parent != source && sssp_data.parents[parent] == i) {
                cout << "Error: Cycle detected between nodes " << i 
                     << " and " << parent << endl;
                has_cycle = true;
            }
        }
    }
}

void identify_affected_nodes(const Graph& graph, const vector<EdgeUpdate>& updates,
                           SSSPData& sssp_data, int source) {
    // Reset affected status
    fill(sssp_data.is_affected.begin(), sssp_data.is_affected.end(), false);
    
    for (const auto& update : updates) {
        if (update.src < 0 || update.src >= graph.num_nodes || 
            update.dest < 0 || update.dest >= graph.num_nodes) {
            cerr << "[Error] Invalid node indices in update: " 
                 << update.src << " -> " << update.dest << endl;
            continue;
        }
        
        int u = update.src;
        int v = update.dest;
        double weight = update.weight;
        bool is_insertion = update.is_insertion;
        
        if (is_insertion) {
            if (sssp_data.distances[u] != INF) {
                double potential_dist = sssp_data.distances[u] + weight;
                if (potential_dist < sssp_data.distances[v]) {
                    sssp_data.is_affected[v] = true;
                }
            }
            
            if (sssp_data.distances[v] != INF) {
                double potential_dist = sssp_data.distances[v] + weight;
                if (potential_dist < sssp_data.distances[u]) {
                    sssp_data.is_affected[u] = true;
                }
            }
        } else {
            if (sssp_data.parents[v] == u || sssp_data.parents[u] == v) {
                sssp_data.is_affected[u] = true;
                sssp_data.is_affected[v] = true;
                
                if (sssp_data.parents[v] == u) {
                    sssp_data.distances[v] = INF;  
                }
                if (sssp_data.parents[u] == v) {
                    sssp_data.distances[u] = INF;  
                }
            }
        }
    }
    
    if (source >= 0 && source < graph.num_nodes) {
        sssp_data.distances[source] = 0;
        sssp_data.parents[source] = source;
        sssp_data.is_affected[source] = false;
    }
    
    // Second pass: Propagate affected status to descendants in the SSSP tree
    // Using fixed number of iterations
    const int MAX_PROPAGATION_ITERATIONS = min(10, graph.num_nodes);
    
    for (int iter = 0; iter < MAX_PROPAGATION_ITERATIONS; iter++) {
        bool any_changes = false;
        
        for (int node = 0; node < graph.num_nodes; node++) {
            if (sssp_data.is_affected[node]) {
                for (const auto& edge : graph.adj_list[node]) {
                    int child = edge.first;
                    
                    if (sssp_data.parents[child] == node && !sssp_data.is_affected[child]) {
                        sssp_data.is_affected[child] = true;
                        any_changes = true;
                        
                        if (sssp_data.distances[node] == INF) {
                            sssp_data.distances[child] = INF;
                        }
                    }
                }
            }
        }
        
        if (!any_changes) {
            break;
        }
    }
    
    int affected_count = 0;
    for (int i = 0; i < graph.num_nodes; i++) {
        if (sssp_data.is_affected[i]) {
            affected_count++;
        }
    }
    
    cout << "\nIdentified " << affected_count << " affected nodes (" 
         << (100.0 * affected_count / graph.num_nodes) << "%)\n" << endl;
}

void compute_sssp_incremental(const Graph& graph, SSSPData& sssp_data, int source) {
    cout << "Starting incremental SSSP computation for affected nodes..." << endl;
    
    if (source < 0 || source >= graph.num_nodes) {
        cerr << "[Error] Invalid source vertex: " << source << endl;
        return;
    }
    
    sssp_data.distances[source] = 0;
    sssp_data.parents[source] = source;
    sssp_data.is_affected[source] = false;
    
    vector<double> new_distances = sssp_data.distances;
    vector<int> new_parents = sssp_data.parents;
    
    bool changes;
    int iterations = 0;
    const int MAX_ITERATIONS = min(100, graph.num_nodes);
    
    int affected_count = 0;
    for (int i = 0; i < graph.num_nodes; i++) {
        if (sssp_data.is_affected[i]) {
            affected_count++;
        }
    }
    
    do {    
        iterations++;
        
        changes = false;
        int relaxations = 0;
        
        // Process vertices that are affected or neighbors of affected nodes
        for (int u = 0; u < graph.num_nodes; u++) {
            // Skip vertices that are not reachable
            if (sssp_data.distances[u] == INF) continue;
            
            // Only process vertices that are affected or have affected neighbors
            bool process_vertex = sssp_data.is_affected[u];
            if (!process_vertex) {
                for (const auto& edge : graph.adj_list[u]) {
                    int neighbor = edge.first;
                    if (sssp_data.is_affected[neighbor]) {
                        process_vertex = true;
                        break;
                    }
                }
            }
            
            if (!process_vertex) continue;
            
            for (const auto& edge : graph.adj_list[u]) {
                int v = edge.first;
                double weight = edge.second;
                double potential_dist = sssp_data.distances[u] + weight;
                
                if (potential_dist < new_distances[v]) {
                    new_distances[v] = potential_dist;
                    new_parents[v] = u;
                    changes = true;
                    relaxations++;
                    sssp_data.is_affected[v] = true;
                }
            }
        }
        
        if (changes) {
            for (int i = 0; i < graph.num_nodes; i++) {
                sssp_data.distances[i] = new_distances[i];
                sssp_data.parents[i] = new_parents[i];
            }
            
            if (source >= 0 && source < graph.num_nodes) {
                sssp_data.distances[source] = 0;
                sssp_data.parents[source] = source;
            }
        }
        
        if (iterations >= MAX_ITERATIONS) {
            cout << "Warning: Incremental SSSP computation reached maximum iterations (" 
                 << MAX_ITERATIONS << "). Forcibly terminating the loop." << endl;
            break;
        }
        
    } while (changes);
    
    cout << "Incremental SSSP computation converged after " << iterations << " iterations." << endl;
    
    int processed_count = 0;
    for (int i = 0; i < graph.num_nodes; i++) {
        if (sssp_data.is_affected[i]) {
            processed_count++;
        }
    }
    
    cout << "Total affected nodes processed: " << processed_count << endl;
}

void process_updates(Graph& graph, const vector<EdgeUpdate>& updates,
                    SSSPData& sssp_data, int source, 
                    long& graph_update_time, long& identify_time, long& compute_time) {
    if (source < 0 || source >= graph.num_nodes) {
        cerr << "[Error] Invalid source vertex: " << source << endl;
        return;
    }
    
    cout << "Applying " << updates.size() << " updates to the graph structure..." << endl;
    
    auto start_time = chrono::high_resolution_clock::now();
    
    int invalid_update_count = 0;
    
    for (const auto& update : updates) {
        if (update.src < 0 || update.src >= graph.num_nodes || 
            update.dest < 0 || update.dest >= graph.num_nodes) {
            invalid_update_count++;
            continue;
        }
        
        int u = update.src;
        int v = update.dest;
        double weight = update.weight;
        bool is_insertion = update.is_insertion;
        
        if (is_insertion) {
            bool found_u_to_v = false;
            bool found_v_to_u = false;
            
            for (auto& edge : graph.adj_list[u]) {
                if (edge.first == v) {
                    edge.second = weight;  
                    found_u_to_v = true;
                    break;
                }
            }
            if (!found_u_to_v) {
                graph.adj_list[u].emplace_back(v, weight);  
            }
            
            for (auto& edge : graph.adj_list[v]) {
                if (edge.first == u) {
                    edge.second = weight;  
                    found_v_to_u = true;
                    break;
                }
            }
            if (!found_v_to_u) {
                graph.adj_list[v].emplace_back(u, weight);  
            }
        } else {
            for (auto it = graph.adj_list[u].begin(); it != graph.adj_list[u].end(); ) {
                if (it->first == v) {
                    it = graph.adj_list[u].erase(it);
                } else {
                    ++it;
                }
            }
            
            for (auto it = graph.adj_list[v].begin(); it != graph.adj_list[v].end(); ) {
                if (it->first == u) {
                    it = graph.adj_list[v].erase(it);
                } else {
                    ++it;
                }
            }
        }
    }
    
    if (invalid_update_count > 0) {
        cout << "Warning: " << invalid_update_count << " updates were invalid and skipped" << endl;
    }
    
    graph.num_edges = 0;
    for (int i = 0; i < graph.num_nodes; i++) {
        graph.num_edges += graph.adj_list[i].size();
    }
    graph.num_edges /= 2;
    
    auto graph_update_end = chrono::high_resolution_clock::now();
    graph_update_time = chrono::duration_cast<chrono::milliseconds>(graph_update_end - start_time).count();
    
    cout << "Graph structure updated in " << graph_update_time << " ms" << endl;
    cout << "New graph size: " << graph.num_nodes << " nodes, " << graph.num_edges << " edges" << endl;
    
    cout << "Identifying affected nodes in the SSSP tree..." << endl;
    
    auto identify_start = chrono::high_resolution_clock::now();
    identify_affected_nodes(graph, updates, sssp_data, source);
    auto identify_end = chrono::high_resolution_clock::now();
    identify_time = chrono::duration_cast<chrono::milliseconds>(identify_end - identify_start).count();
    
    cout << "Affected nodes identified in " << identify_time << " ms" << endl;
    
    int affected_count = 0;
    for (int i = 0; i < graph.num_nodes; i++) {
        if (sssp_data.is_affected[i]) {
            affected_count++;
        }
    }
    
    cout << "Total affected nodes: " << affected_count << endl;
    
    if (affected_count > 0) {
        cout << "\n[Incremental Update] Performing incremental SSSP computation for affected nodes..." << endl;
        
        auto compute_start = chrono::high_resolution_clock::now();
        compute_sssp_incremental(graph, sssp_data, source);
        auto compute_end = chrono::high_resolution_clock::now();
        compute_time = chrono::duration_cast<chrono::milliseconds>(compute_end - compute_start).count();
        
        cout << "\nIncremental SSSP on Affected nodes computation completed in " << compute_time << " ms\n" << endl;
        cout << "Total update processing time: " << (graph_update_time + identify_time + compute_time) << " ms" << endl;
        
        cout << "\n-----------------------------------------" << endl;
        cout << "Affected Nodes After Updates (Source: " << source << ")" << endl;
        cout << "-----------------------------------------" << endl;
        cout << "| Node | Distance | Parent | Affected |" << endl;
        cout << "-----------------------------------------" << endl;
        
        for (int i = 0; i < graph.num_nodes; i++) {
            if (sssp_data.is_affected[i]) {
                cout << "| " << setw(4) << i << " | ";
                if (sssp_data.distances[i] == INF) {
                    cout << setw(8) << "INF" << " | ";
                } else {
                    cout << setw(8) << sssp_data.distances[i] << " | ";
                }
                cout << setw(6) << sssp_data.parents[i] << " | ";
                cout << setw(8) << "Yes" << " |" << endl;
            }
        }
        cout << "-----------------------------------------" << endl;
    } else {
        cout << "No nodes affected by the updates, SSSP tree remains unchanged." << endl;
        cout << "Total update processing time: " << (graph_update_time + identify_time) << " ms" << endl;
        compute_time = 0; // Set to 0 when no affected nodes found
    }
    
    bool found_cycle = false;
    int relaxable_edges = 0;
    
    for (int u = 0; u < graph.num_nodes; u++) {
        if (u != source && sssp_data.parents[u] == u) {
            cout << "Warning: Node " << u << " is its own parent!" << endl;
            found_cycle = true;
        }
        
        if (sssp_data.distances[u] != INF) {
            for (const auto& edge : graph.adj_list[u]) {
                int v = edge.first;
                double weight = edge.second;
                double potential_dist = sssp_data.distances[u] + weight;
                
                if (potential_dist < sssp_data.distances[v] - 1e-9) {
                    relaxable_edges++;
                    if (relaxable_edges <= 5) {
                        cout << "Warning: Edge (" << u << " -> " << v 
                             << ") is relaxable: " << sssp_data.distances[u] << " + " 
                             << weight << " < " << sssp_data.distances[v] << endl;
                    }
                }
            }
        }
    }
    
    if (relaxable_edges > 0) {
        cout << "[Warning] Final solution has " << relaxable_edges 
             << " relaxable edges! Solution may be suboptimal." << endl;
    } else {
        cout << "[Validation] Final SSSP solution is optimal (no relaxable edges found)." << endl;
    }
    
    if (found_cycle) {
        cout << "[Warning] Found cycles in the parent pointers! Solution may be invalid." << endl;
    } else {
        cout << "[Validation] No cycles found in the SSSP tree." << endl;
    }
}

void compute_sssp(const Graph& graph, SSSPData& sssp_data, int source);
void identify_affected_nodes(const Graph& graph, const vector<EdgeUpdate>& updates, SSSPData& sssp_data, int source);
void compute_sssp_incremental(const Graph& graph, SSSPData& sssp_data, int source);
void process_updates(Graph& graph, const vector<EdgeUpdate>& updates, SSSPData& sssp_data, int source, long& graph_update_time, long& identify_time, long& compute_time);

int main(int argc, char* argv[]) {
    auto program_start_time = chrono::high_resolution_clock::now();
    
    cout << "\n===================================================" << endl;
    cout << "   Serial SSSP Computation" << endl;
    cout << "===================================================" << endl;
    
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <graph_file> <source_vertex>" << endl;
        return 1;
    }
    
    string graph_file = argv[1];
    int source_vertex = atoi(argv[2]);
    
    cout << "Graph file: " << graph_file << endl;
    cout << "Source vertex: " << source_vertex << endl;
    
    Graph graph(0);
    
    cout << "\n[Phase 1: Graph Loading]" << endl;
    cout << "-------------------------------------------------" << endl;
    
    // Check if the file is a METIS format file based on extension
    bool is_metis_file = (graph_file.find(".metis") != string::npos);
    
    cout << "Reading graph from " << graph_file << " (format: " 
         << (is_metis_file ? "METIS" : "Standard") << ")" << endl;
    
    auto read_start = chrono::high_resolution_clock::now();
    
    if (is_metis_file) {
        graph = read_metis_graph(graph_file);
    } else {
        graph = read_graph(graph_file);
    }
    
    auto read_end = chrono::high_resolution_clock::now();
    auto read_time = chrono::duration_cast<chrono::milliseconds>(read_end - read_start).count();
    
    cout << "Graph read successfully in " << read_time << " ms" << endl;
    cout << "Nodes: " << graph.num_nodes << ", Edges: " << graph.num_edges << endl;
    
    print_graph(graph);
    
    cout << "\n[Phase 2: Initial SSSP Computation]" << endl;
    cout << "-------------------------------------------------" << endl;
    
    // Initialize SSSP data
    SSSPData sssp_data(graph.num_nodes);
    
    auto start_time = chrono::high_resolution_clock::now();
    compute_sssp(graph, sssp_data, source_vertex);
    auto end_time = chrono::high_resolution_clock::now();
    auto initial_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    
    vector<double> initial_distances = sssp_data.distances;
    
    cout << "\nInitial SSSP computation time: " << initial_time << " ms" << endl;
    
    cout << "\n-----------------------------------------" << endl;
    cout << "Initial SSSP Results (Source: " << source_vertex << ")" << endl;
    cout << "-----------------------------------------" << endl;
    cout << "| Node | Distance | Parent |" << endl;
    cout << "-----------------------------------------" << endl;
    
    for (int i = 0; i < graph.num_nodes; i++) {
        cout << "| " << setw(4) << i << " | ";
        if (sssp_data.distances[i] == INF) {
            cout << setw(8) << "INF" << " | ";
        } else {
            cout << setw(8) << sssp_data.distances[i] << " | ";
        }
        cout << setw(6) << sssp_data.parents[i] << " |" << endl;
    }
    cout << "-----------------------------------------" << endl;
    
    // Generate random edge updates
    vector<EdgeUpdate> updates;
    int num_affected_nodes = max(1, static_cast<int>(0.05 * graph.num_nodes));
    vector<int> affected_nodes;
    
    random_device rd;   
    mt19937 gen(rd());
    
    vector<int> all_nodes(graph.num_nodes);
    iota(all_nodes.begin(), all_nodes.end(), 0); 
    shuffle(all_nodes.begin(), all_nodes.end(), gen);
    
    affected_nodes.assign(all_nodes.begin(), all_nodes.begin() + num_affected_nodes);
    
    cout << "Generating updates for " << num_affected_nodes << " affected nodes (5% of nodes)" << endl;
    cout << "Affected nodes: ";
    for (int node : affected_nodes) {
        cout << node << " ";
    }
    cout << endl;
    
    uniform_int_distribution<> integer_weight_dist(1, 10);  
    
    set<pair<int, int>> existing_edges;
    for (int i = 0; i < graph.num_nodes; i++) {
        for (const auto& edge : graph.adj_list[i]) {
            int j = edge.first;
            if (i < j) { 
                existing_edges.insert({i, j});
            }
        }
    }
    
    set<pair<int, int>> potential_new_edges;
    for (int i : affected_nodes) {
        for (int j : affected_nodes) {
            if (i < j) { 
                pair<int, int> edge = {i, j};
                if (existing_edges.find(edge) == existing_edges.end()) {
                    potential_new_edges.insert(edge);
                }
            }
        }
    }
    
    vector<pair<int, int>> existing_edges_vec(existing_edges.begin(), existing_edges.end());
    vector<pair<int, int>> potential_new_edges_vec(potential_new_edges.begin(), potential_new_edges.end());
    
    int num_insertions = min(static_cast<int>(potential_new_edges_vec.size()), 
                             max(1, static_cast<int>(num_affected_nodes * 2)));
    int num_deletions = min(static_cast<int>(existing_edges_vec.size()), 
                           max(1, static_cast<int>(num_affected_nodes * 2)));
    
    cout << "Generating " << num_insertions << " insertions and " 
         << num_deletions << " deletions" << endl;
    
    if (!potential_new_edges_vec.empty()) {
        shuffle(potential_new_edges_vec.begin(), potential_new_edges_vec.end(), gen);
        for (int i = 0; i < num_insertions && i < potential_new_edges_vec.size(); i++) {
            int u = potential_new_edges_vec[i].first;
            int v = potential_new_edges_vec[i].second;
            int weight = integer_weight_dist(gen);
            updates.emplace_back(u, v, static_cast<double>(weight), true);
        }
    }
    
    if (!existing_edges_vec.empty()) {
        shuffle(existing_edges_vec.begin(), existing_edges_vec.end(), gen);
        for (int i = 0; i < num_deletions && i < existing_edges_vec.size(); i++) {
            int u = existing_edges_vec[i].first;
            int v = existing_edges_vec[i].second;
            
            double weight = 0.0;
            for (const auto& edge : graph.adj_list[u]) {
                if (edge.first == v) {
                    weight = edge.second;
                    break;
                }
            }
            
            updates.emplace_back(u, v, weight, false);
        }
    }
    
    shuffle(updates.begin(), updates.end(), gen);
    
    cout << "Generated " << updates.size() << " total updates" << endl;
    
    cout << "\n[Phase 3: Processing Updates]" << endl;
    cout << "-------------------------------------------------" << endl;
    
    long graph_update_time = 0;
    long identify_time = 0;
    long compute_time = 0;
    
    start_time = chrono::high_resolution_clock::now();
    process_updates(graph, updates, sssp_data, source_vertex, graph_update_time, identify_time, compute_time);
    end_time = chrono::high_resolution_clock::now();
    auto update_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    
    cout << "\nTotal update processing time: " << update_time << " ms" << endl;
    cout << "Time per update: " << (double)update_time / updates.size() << " ms" << endl;
    
    cout << "\n-----------------------------------------" << endl;
    cout << "Updates Statistics" << endl;
    cout << "-----------------------------------------" << endl;
    cout << "Total updates processed: " << updates.size() << endl;
    int insertions = 0, deletions = 0;
    for (const auto& update : updates) {
        if (update.is_insertion) insertions++;
        else deletions++;
    }
    cout << "Insertion updates: " << insertions << endl;
    cout << "Deletion updates: " << deletions << endl;
    
    const int sample_size = min(10, (int)updates.size());
    cout << "\nSample of first " << sample_size << " updates:" << endl;
    cout << "-----------------------------------------" << endl;
    cout << "| Type | Source | Destination | Weight |" << endl;
    cout << "-----------------------------------------" << endl;
    for (int i = 0; i < sample_size; i++) {
        const auto& update = updates[i];
        cout << "| " << setw(4) << (update.is_insertion ? "INS" : "DEL") << " | ";
        cout << setw(6) << update.src << " | ";
        cout << setw(11) << update.dest << " | ";
        cout << setw(6) << update.weight << " |" << endl;
    }
    cout << "-----------------------------------------" << endl;
    
    cout << "\n-----------------------------------------" << endl;
    cout << "Updated SSSP Results (Source: " << source_vertex << ")" << endl;
    cout << "-----------------------------------------" << endl;
    cout << "| Node | Distance | Parent |" << endl;
    cout << "-----------------------------------------" << endl;
    
    for (int i = 0; i < graph.num_nodes; i++) {
        cout << "| " << setw(4) << i << " | ";
        if (sssp_data.distances[i] == INF) {
            cout << setw(8) << "INF" << " | ";
        } else {
            cout << setw(8) << sssp_data.distances[i] << " | ";
        }
        cout << setw(6) << sssp_data.parents[i] << " |" << endl;
    }
    cout << "-----------------------------------------" << endl;
    
    cout << "\n-----------------------------------------" << endl;
    cout << "Distance Changes After Updates" << endl;
    cout << "-----------------------------------------" << endl;
    cout << "| Node | Initial Dist | Final Dist | Change |" << endl;
    cout << "-----------------------------------------" << endl;
    
    int changed_count = 0;
    for (int i = 0; i < graph.num_nodes; i++) {
        if (initial_distances[i] != sssp_data.distances[i]) {
            changed_count++;
            cout << "| " << setw(4) << i << " | ";
            
            if (initial_distances[i] == INF) {
                cout << setw(12) << "INF" << " | ";
            } else {
                cout << setw(12) << initial_distances[i] << " | ";
            }
            
            if (sssp_data.distances[i] == INF) {
                cout << setw(10) << "INF" << " | ";
            } else {
                cout << setw(10) << sssp_data.distances[i] << " | ";
            }
            
            if (initial_distances[i] == INF && sssp_data.distances[i] != INF) {
                cout << " Connected |" << endl;
            } else if (initial_distances[i] != INF && sssp_data.distances[i] == INF) {
                cout << " Disconnected |" << endl;
            } else if (initial_distances[i] != INF && sssp_data.distances[i] != INF) {
                double diff = sssp_data.distances[i] - initial_distances[i];
                cout << setw(7) << diff << " |" << endl;
            }
        }
    }
    
    if (changed_count == 0) {
        cout << "| No changes in distances after updates |" << endl;
    }
    cout << "-----------------------------------------" << endl;
    
    auto program_end_time = chrono::high_resolution_clock::now();
    auto total_time = chrono::duration_cast<chrono::milliseconds>(program_end_time - program_start_time).count();
    
    cout << "\n===================================================" << endl;
    cout << "Performance Summary" << endl;
    cout << "===================================================" << endl;
    cout << "Graph Reading Time: " << read_time << " ms" << endl;
    cout << "Initial SSSP Time: " << initial_time << " ms" << endl;
    cout << "Update Processing Breakdown:" << endl;
    
    // Calculate average times
    cout << "  - Graph Structure Update: " << graph_update_time << " ms" << endl;
    cout << "  - Affected Nodes Identification: " << identify_time << " ms" << endl;
    cout << "  - Incremental SSSP Computation: " << compute_time << " ms" << endl;
    cout << "Total Update Processing Time: " << update_time << " ms" << endl;
    cout << "Average Time per Update: " << (double)update_time / updates.size() << " ms" << endl;
    cout << "Average Time per Node: " << (double)initial_time / graph.num_nodes << " ms (initial SSSP)" << endl;
    cout << "--------------------------------------------------" << endl;
    cout << "Total Execution Time: " << total_time << " ms" << endl;
    cout << "===================================================" << endl;
    
    cout << "\n===================================================" << endl;
    cout << "Program Completed Successfully" << endl;
    cout << "Total execution time: " << total_time << " ms" << endl;
    cout << "===================================================" << endl;
    
    return 0;
} 