#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <fstream>
#include <sstream>
#include <mpi.h>
#include <omp.h>
// METIS dependency removed - we'll use the partition file directly
#include <unordered_map>
#include <set>
#include <algorithm>  
#include <chrono>
#include <random>
#include <iomanip>

using namespace std;

// Constants
const double INF = numeric_limits<double>::infinity();

// Structures for graph representation
struct Edge {
    int src, dest;
    double weight;
    Edge(int s, int d, double w) : src(s), dest(d), weight(w) {}
};

struct Graph {
    int num_nodes;
    int num_edges;
    vector<vector<pair<int, double>>> adj_list;
    // Constructor
    Graph(int n) : num_nodes(n), num_edges(0) {
        adj_list.resize(n);
    }
};

// Structure for partition information
struct PartitionInfo {
    int partition_id;
    vector<int> local_vertices;
    vector<int> ghost_vertices;
    unordered_map<int, int> global_to_local;
    vector<int> local_to_global;
};

// Structure for SSSP data
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

// Structure for edge updates
struct EdgeUpdate {
    int src, dest;
    double weight;
    bool is_insertion;  // true for insertion, false for deletion
    EdgeUpdate() : src(-1), dest(-1), weight(0.0), is_insertion(false) {}  // Default constructor
    EdgeUpdate(int s, int d, double w, bool ins) 
        : src(s), dest(d), weight(w), is_insertion(ins) {}
};

// Function to read graph from file
Graph read_graph(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // First pass: collect all unique node indices and count edges
    set<int> node_indices;
    int m = 0;
    int u, v;
    double weight;
    while (file >> u >> v >> weight) {
        node_indices.insert(u);
        node_indices.insert(v);
        m++;
    }
    
    // Create mapping from original indices to contiguous indices
    unordered_map<int, int> node_map;
    int new_index = 0;
    for (int node : node_indices) {
        node_map[node] = new_index++;
    }
    
    // Reset file to beginning
    file.clear();
    file.seekg(0);
    
    // Create graph with actual number of nodes
    Graph graph(node_indices.size());
    graph.num_edges = m;

    // Read edges and map indices
    while (file >> u >> v >> weight) {
        int mapped_u = node_map[u];
        int mapped_v = node_map[v];
        
        if (mapped_u >= graph.num_nodes || mapped_v >= graph.num_nodes || 
            mapped_u < 0 || mapped_v < 0) {
            cerr << "Error: Mapped node index out of bounds: " << mapped_u << " or " << mapped_v 
                 << " (n=" << graph.num_nodes << ")" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        graph.adj_list[mapped_u].emplace_back(mapped_v, weight);
        graph.adj_list[mapped_v].emplace_back(mapped_u, weight); // Undirected graph
    }

    file.close();
    return graph;
}

// Function to read partitions from a file
vector<int> read_partitions(const string& partition_file, int num_nodes) {
    vector<int> partition(num_nodes, 0);  // Default all nodes to partition 0
    
    ifstream file(partition_file);
    if (!file.is_open()) {
        cerr << "Error opening partition file: " << partition_file << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Read partition info for as many nodes as available in the file
    for (int i = 0; i < num_nodes; i++) {
        int part_id;
        if (file >> part_id) {
            partition[i] = part_id;
        } else {
            cout << "Note: Using default partition (0) for node " << i 
                 << " and any remaining nodes" << endl;
            break;  // Stop reading if we reach the end of the file
        }
    }
    file.close();
    
    return partition;
}

// Function to initialize partition information
PartitionInfo initialize_partition(const Graph& graph, const vector<int>& partition, int rank, int num_partitions) {
    PartitionInfo info;
    info.partition_id = rank;
    
    // Map partition IDs to process ranks if they don't match
    vector<int> partition_to_rank(num_partitions, -1);
    vector<int> unique_partitions;
    
    // Collect unique partition IDs
    for (int p : partition) {
        if (find(unique_partitions.begin(), unique_partitions.end(), p) == unique_partitions.end()) {
            unique_partitions.push_back(p);
        }
    }
    
    // Assign partitions to ranks round-robin
    for (size_t i = 0; i < unique_partitions.size(); i++) {
        partition_to_rank[unique_partitions[i]] = i % num_partitions;
    }
    
    if (rank == 0) {
        cout << "Partition to rank mapping: ";
        for (size_t i = 0; i < unique_partitions.size(); i++) {
            cout << unique_partitions[i] << "->" << (i % num_partitions) << " ";
        }
        cout << endl;
    }
    
    // Identify local vertices using the mapping
    for (int i = 0; i < graph.num_nodes; i++) {
        int assigned_rank = partition_to_rank[partition[i]];
        if (assigned_rank == rank) {
            info.local_vertices.push_back(i);
            info.global_to_local[i] = info.local_vertices.size() - 1;
            info.local_to_global.push_back(i);
        }
    }
    
    // Identify ghost vertices (neighbors of local vertices in other partitions)
    set<int> ghost_set;
    for (int local_vertex : info.local_vertices) {
        for (const auto& edge : graph.adj_list[local_vertex]) {
            int neighbor = edge.first;
            int neighbor_rank = partition_to_rank[partition[neighbor]];
            if (neighbor_rank != rank) {
                ghost_set.insert(neighbor);
            }
        }
    }
    info.ghost_vertices = vector<int>(ghost_set.begin(), ghost_set.end());
    
    return info;
}

// Function to print partition information
void print_partition_info(const vector<int>& partition, int num_nodes, int num_procs) {
    cout << "\n-----------------------------------------" << endl;
    cout << "Partition Information" << endl;
    cout << "-----------------------------------------" << endl;
    cout << "Number of partitions: " << num_procs << endl;
    
    // Count nodes per partition
    vector<int> nodes_per_partition(num_procs, 0);
    for (int i = 0; i < num_nodes; i++) {
        nodes_per_partition[partition[i]]++;
    }
    
    cout << "-----------------------------------------" << endl;
    cout << "| Partition | Node Count | Nodes |" << endl;
    cout << "-----------------------------------------" << endl;
    
    for (int p = 0; p < num_procs; p++) {
        cout << "| " << setw(9) << p << " | " << setw(10) << nodes_per_partition[p] << " | ";
        
        // List nodes in this partition
        bool first = true;
        for (int i = 0; i < num_nodes; i++) {
            if (partition[i] == p) {
                if (!first) cout << ", ";
                cout << i;
                first = false;
            }
        }
        cout << " |" << endl;
    }
    cout << "-----------------------------------------" << endl;
}

// Function to print graph in a tabular format
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

// Main SSSP computation function
void compute_sssp_distributed(const Graph& graph, const PartitionInfo& part_info, 
                            SSSPData& sssp_data, int source, int rank, int num_procs) {
    // Initialize distances for local vertices
    if (rank == 0) {
        cout << "Initializing SSSP with source vertex: " << source << endl;
        cout << "Rank " << rank << " has " << part_info.local_vertices.size() 
             << " local vertices: ";
        for (int i = 0; i < min(10, (int)part_info.local_vertices.size()); i++) {
            cout << part_info.local_vertices[i] << " ";
        }
        if (part_info.local_vertices.size() > 10) cout << "...";
        cout << endl;
    }
    
    // Initialize all distances to infinity and parents to -1
    for (size_t i = 0; i < graph.num_nodes; i++) {
        sssp_data.distances[i] = INF;
        sssp_data.parents[i] = -1;
    }
    
    // Set source distance to 0
    if (rank == 0) {
        sssp_data.distances[source] = 0.0;
        sssp_data.parents[source] = source;
        cout << "  Setting distance[" << source << "] = 0.0" << endl;
    }
    
    // Make sure all processes have the initial state
    for (int i = 0; i < graph.num_nodes; i++) {
        MPI_Bcast(&sssp_data.distances[i], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&sssp_data.parents[i], 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    
    // Create temporary arrays for relaxation
    vector<double> new_distances(graph.num_nodes, INF);
    vector<int> new_parents(graph.num_nodes, -1);
    
    bool global_changes;
    int iterations = 0;
    const int MAX_ITERATIONS = graph.num_nodes; // Maximum possible iterations for convergence
    
    do {
        // Copy current state to temporary arrays
        for (int i = 0; i < graph.num_nodes; i++) {
            new_distances[i] = sssp_data.distances[i];
            new_parents[i] = sssp_data.parents[i];
        }
        
        bool local_changes = false;
        
        // Process local vertices (edge relaxation)
        #pragma omp parallel for reduction(|:local_changes)
        for (size_t i = 0; i < part_info.local_vertices.size(); i++) {
            int u = part_info.local_vertices[i];
            
            // Skip vertices that are not reachable
            if (sssp_data.distances[u] == INF) continue;
            
            for (const auto& edge : graph.adj_list[u]) {
                int v = edge.first;
                double weight = edge.second;
                double potential_dist = sssp_data.distances[u] + weight;
                
                // Relaxation: If we found a shorter path to v
                if (potential_dist < new_distances[v]) {
                    #pragma omp critical
                    {
                        if (potential_dist < new_distances[v]) {
                            new_distances[v] = potential_dist;
                            new_parents[v] = u;
                            local_changes = true;
                            
                            // Debug output for potential cycles
                            if (rank == 0 && (v == u || (new_parents[v] == u && new_parents[u] == v))) {
                                cout << "Warning: Potential cycle detected between " << u << " and " << v << endl;
                            }
                        }
                    }
                }
            }
        }
        
        // Synchronize changes across processes
        MPI_Allreduce(&local_changes, &global_changes, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
        
        // If there are changes, synchronize the new distances and parents
        if (global_changes) {
            // Gather all updates from all processes
            for (int r = 0; r < num_procs; r++) {
                // Send local updates to all processes
                for (int i = 0; i < graph.num_nodes; i++) {
                    double dist_i = (r == rank) ? new_distances[i] : INF;
                    int parent_i = (r == rank) ? new_parents[i] : -1;
                    
                    // Broadcast updates
                    MPI_Bcast(&dist_i, 1, MPI_DOUBLE, r, MPI_COMM_WORLD);
                    MPI_Bcast(&parent_i, 1, MPI_INT, r, MPI_COMM_WORLD);
                    
                    // Apply updates if better
                    if (dist_i < sssp_data.distances[i]) {
                        sssp_data.distances[i] = dist_i;
                        sssp_data.parents[i] = parent_i;
                        
                        // Also update the temporary arrays
                        new_distances[i] = dist_i;
                        new_parents[i] = parent_i;
                    }
                }
            }
        }
        
        iterations++;
        
        // Enforce a maximum number of iterations to prevent infinite loops
        if (iterations >= MAX_ITERATIONS) {
            if (rank == 0) {
                cout << "Warning: SSSP computation reached maximum iterations (" << MAX_ITERATIONS 
                     << "). There might be negative cycles in the graph." << endl;
            }
            break;
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
    } while (global_changes);
    
    if (rank == 0) {
        cout << "SSSP computation converged after " << iterations << " iterations." << endl;
        
        // Verify no cycles in the parent tree
        bool has_cycle = false;
        for (int i = 0; i < graph.num_nodes; i++) {
            if (i != source && sssp_data.parents[i] != -1) {
                // Check for immediate cycle
                if (i == sssp_data.parents[i]) {
                    cout << "Error: Node " << i << " is its own parent!" << endl;
                    has_cycle = true;
                }
                
                // Check for two-node cycle
                int parent = sssp_data.parents[i];
                if (parent != -1 && parent != source && sssp_data.parents[parent] == i) {
                    cout << "Error: Cycle detected between nodes " << i << " and " << parent << endl;
                    has_cycle = true;
                }
            }
        }
        
        if (!has_cycle) {
            cout << "SSSP tree verification passed: No cycles detected." << endl;
        }
    }
}

// Function to identify affected vertices for edge deletion
vector<int> identify_deletion_affected(const Graph& graph, const SSSPData& sssp_data, 
                                     const EdgeUpdate& update, const PartitionInfo& part_info) {
    vector<int> affected;
    int u = update.src;
    int v = update.dest;
    
    // Check if the edge was part of the SSSP tree
    if (sssp_data.parents[v] == u || sssp_data.parents[u] == v) {
        int y = (sssp_data.distances[u] > sssp_data.distances[v]) ? u : v;
        if (find(part_info.local_vertices.begin(), 
                 part_info.local_vertices.end(), y) != part_info.local_vertices.end()) {
            affected.push_back(y);
        }
    }
    return affected;
}

// Function to identify affected vertices for edge insertion
vector<int> identify_insertion_affected(const Graph& graph, const SSSPData& sssp_data, 
                                      const EdgeUpdate& update, const PartitionInfo& part_info) {
    vector<int> affected;
    int u = update.src;
    int v = update.dest;
    double weight = update.weight;
    
    // Determine which vertex might be affected
    int x, y;
    if (sssp_data.distances[u] > sssp_data.distances[v]) {
        x = v;
        y = u;
    } else {
        x = u;
        y = v;
    }
    
    // Check if the new edge creates a shorter path
    if (sssp_data.distances[y] > sssp_data.distances[x] + weight) {
        if (find(part_info.local_vertices.begin(), 
                 part_info.local_vertices.end(), y) != part_info.local_vertices.end()) {
            affected.push_back(y);
        }
    }
    return affected;
}

// Function to process deletion updates
void process_deletion_updates(Graph& graph, const vector<EdgeUpdate>& deletions,
                            const PartitionInfo& part_info, SSSPData& sssp_data) {
    vector<int> all_affected;
    
    // First pass: Identify affected vertices
    #pragma omp parallel for
    for (size_t i = 0; i < deletions.size(); i++) {
        const EdgeUpdate& update = deletions[i];
        vector<int> affected = identify_deletion_affected(graph, sssp_data, update, part_info);
        #pragma omp critical
        {
            all_affected.insert(all_affected.end(), affected.begin(), affected.end());
        }
    }
    
    // Remove duplicates
    sort(all_affected.begin(), all_affected.end());
    all_affected.erase(unique(all_affected.begin(), all_affected.end()), all_affected.end());
    
    // Process affected vertices
    while (!all_affected.empty()) {
        vector<int> new_affected;
        
        #pragma omp parallel for
        for (size_t i = 0; i < all_affected.size(); i++) {
            int v = all_affected[i];
            sssp_data.distances[v] = INF;
            sssp_data.parents[v] = -1;
            
            // Find all children in the SSSP tree
            for (int u : part_info.local_vertices) {
                if (sssp_data.parents[u] == v) {
                    #pragma omp critical
                    {
                        new_affected.push_back(u);
                    }
                }
            }
        }
        
        all_affected = new_affected;
    }
}

// Function to process insertion updates
void process_insertion_updates(Graph& graph, const vector<EdgeUpdate>& insertions,
                             const PartitionInfo& part_info, SSSPData& sssp_data) {
    vector<int> all_affected;
    
    // First pass: Identify affected vertices
    #pragma omp parallel for
    for (size_t i = 0; i < insertions.size(); i++) {
        const EdgeUpdate& update = insertions[i];
        vector<int> affected = identify_insertion_affected(graph, sssp_data, update, part_info);
        #pragma omp critical
        {
            all_affected.insert(all_affected.end(), affected.begin(), affected.end());
        }
    }
    
    // Remove duplicates
    sort(all_affected.begin(), all_affected.end());
    all_affected.erase(unique(all_affected.begin(), all_affected.end()), all_affected.end());
    
    // Process affected vertices
    while (!all_affected.empty()) {
        vector<int> new_affected;
        
        #pragma omp parallel for
        for (size_t i = 0; i < all_affected.size(); i++) {
            int v = all_affected[i];
            
            // Check all neighbors for potential updates
            for (const auto& edge : graph.adj_list[v]) {
                int u = edge.first;
                double weight = edge.second;
                
                if (sssp_data.distances[u] > sssp_data.distances[v] + weight) {
                    #pragma omp critical
                    {
                        sssp_data.distances[u] = sssp_data.distances[v] + weight;
                        sssp_data.parents[u] = v;
                        new_affected.push_back(u);
                    }
                }
            }
        }
        
        all_affected = new_affected;
    }
}

// Modified process_updates function to use the new two-phase approach
void process_updates(Graph& graph, const vector<EdgeUpdate>& updates,
                    const PartitionInfo& part_info, SSSPData& sssp_data,
                    int source, int rank, int num_procs) {
    // First, apply all updates to the graph structure
    if (rank == 0) {
        cout << "Applying " << updates.size() << " updates to the graph structure..." << endl;
    }
    
    #pragma omp parallel for
    for (size_t i = 0; i < updates.size(); i++) {
        const EdgeUpdate& update = updates[i];
        int u = update.src;
        int v = update.dest;
        double weight = update.weight;
        bool is_insertion = update.is_insertion;
        
        if (is_insertion) {
            // Add or update edge in both directions for undirected graph
            bool found_u_to_v = false;
            bool found_v_to_u = false;
            
            #pragma omp critical
            {
                // Update u -> v
                for (auto& edge : graph.adj_list[u]) {
                    if (edge.first == v) {
                        edge.second = weight;  // Update existing edge weight
                        found_u_to_v = true;
                        break;
                    }
                }
                if (!found_u_to_v) {
                    graph.adj_list[u].emplace_back(v, weight);  // Add new edge
                }
                
                // Update v -> u (for undirected)
                for (auto& edge : graph.adj_list[v]) {
                    if (edge.first == u) {
                        edge.second = weight;  // Update existing edge weight
                        found_v_to_u = true;
                        break;
                    }
                }
                if (!found_v_to_u) {
                    graph.adj_list[v].emplace_back(u, weight);  // Add new edge
                }
            }
        } else {
            // Deletion - remove the edge in both directions for undirected graph
            #pragma omp critical
            {
                // Remove u -> v
                for (auto it = graph.adj_list[u].begin(); it != graph.adj_list[u].end(); ) {
                    if (it->first == v) {
                        it = graph.adj_list[u].erase(it);
                    } else {
                        ++it;
                    }
                }
                
                // Remove v -> u
                for (auto it = graph.adj_list[v].begin(); it != graph.adj_list[v].end(); ) {
                    if (it->first == u) {
                        it = graph.adj_list[v].erase(it);
                    } else {
                        ++it;
                    }
                }
            }
        }
    }
    
    // Now recompute SSSP entirely
    if (rank == 0) {
        cout << "Recomputing SSSP after updates..." << endl;
    }
    
    // Reinitialize the SSSP data
    for (int i = 0; i < graph.num_nodes; i++) {
        sssp_data.distances[i] = INF;
        sssp_data.parents[i] = -1;
    }
    
    // Recompute SSSP
    compute_sssp_distributed(graph, part_info, sssp_data, source, rank, num_procs);
}

// Function to read graph from METIS format file
Graph read_metis_graph(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening METIS file: " << filename << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Read header line
    string header;
    getline(file, header);
    istringstream header_ss(header);
    
    int num_nodes, num_edges;
    string format_info;
    
    header_ss >> num_nodes >> num_edges;
    
    // Check if there's format information (001 means weighted edges)
    bool weighted = false;
    if (header_ss >> format_info && format_info == "001") {
        weighted = true;
    }
    
    cout << "Reading METIS file with " << num_nodes << " nodes and " 
         << num_edges << " edges, weighted=" << weighted << endl;
    
    // Create the graph
    Graph graph(num_nodes);
    graph.num_edges = num_edges * 2; // METIS stores undirected edges, but we need both directions
    
    // Read adjacency lists
    for (int i = 0; i < num_nodes; i++) {
        string line;
        if (!getline(file, line)) {
            cerr << "Error reading adjacency list for node " << i << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        istringstream line_ss(line);
        int neighbor;
        double weight;
        
        while (line_ss >> neighbor) {
            // METIS uses 1-based indexing, convert to 0-based
            neighbor--;
            
            // Read weight if the graph is weighted
            if (weighted) {
                if (!(line_ss >> weight)) {
                    cerr << "Error reading weight for edge (" << i << ", " << neighbor << ")" << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            } else {
                weight = 1.0; // Default weight
            }
            
            // Add edge to graph
            if (0 <= neighbor && neighbor < num_nodes) {
                graph.adj_list[i].emplace_back(neighbor, weight);
            } else {
                cerr << "Error: Invalid neighbor index " << neighbor+1 
                     << " for node " << i << " (must be 1-" << num_nodes << ")" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }
    
    file.close();
    return graph;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    if (argc < 3 || argc > 4) {
        if (rank == 0) {
            cerr << "Usage: " << argv[0] << " <graph_file> <source_vertex> [partition_file]" << endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    string graph_file = argv[1];
    int source_vertex = atoi(argv[2]);
    string partition_file = (argc == 4) ? argv[3] : "graph.metis.part.4";
    
    // Only rank 0 reads the graph
    Graph graph(0);
    int num_nodes = 0;
    int num_edges = 0;
    
    if (rank == 0) {
        // Check if the file is a METIS format file based on extension
        bool is_metis_file = (graph_file.find(".metis") != string::npos);
        
        // Read the graph file using the appropriate function
        if (is_metis_file) {
            graph = read_metis_graph(graph_file);
        } else {
            graph = read_graph(graph_file);
        }
        
        num_nodes = graph.num_nodes;
        num_edges = graph.num_edges;
        cout << "Graph read successfully. Nodes: " << num_nodes 
             << ", Edges: " << num_edges << endl;
        
        // Print the graph structure
        print_graph(graph);
    }
    
    // Broadcast graph size
    MPI_Bcast(&num_nodes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Initialize graph on all processes
    if (rank != 0) {
        graph = Graph(num_nodes);
        graph.num_edges = num_edges;
    }
    
    // Broadcast graph structure
    for (int i = 0; i < num_nodes; i++) {
        int num_edges_i;
        if (rank == 0) {
            num_edges_i = graph.adj_list[i].size();
        }
        MPI_Bcast(&num_edges_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank != 0) {
            graph.adj_list[i].resize(num_edges_i);
        }
        
        if (num_edges_i > 0) {
            vector<int> neighbors(num_edges_i);
            vector<double> weights(num_edges_i);
            
            if (rank == 0) {
                for (int j = 0; j < num_edges_i; j++) {
                    neighbors[j] = graph.adj_list[i][j].first;
                    weights[j] = graph.adj_list[i][j].second;
                }
            }
            
            MPI_Bcast(neighbors.data(), num_edges_i, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(weights.data(), num_edges_i, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            
            if (rank != 0) {
                for (int j = 0; j < num_edges_i; j++) {
                    graph.adj_list[i][j] = make_pair(neighbors[j], weights[j]);
                }
            }
        }
    }
    
    // Read partition information
    vector<int> partition(num_nodes);
    if (rank == 0) {
        partition = read_partitions(partition_file, num_nodes);
    }
    
    // Broadcast partition information
    MPI_Bcast(partition.data(), num_nodes, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Print partition information (only by rank 0)
    if (rank == 0) {
        print_partition_info(partition, num_nodes, num_procs);
    }
    
    // Initialize partition information
    PartitionInfo part_info = initialize_partition(graph, partition, rank, num_procs);
    
    // Initialize SSSP data
    SSSPData sssp_data(num_nodes);
    
    // Compute initial SSSP
    auto start_time = chrono::high_resolution_clock::now();
    compute_sssp_distributed(graph, part_info, sssp_data, source_vertex, rank, num_procs);
    auto end_time = chrono::high_resolution_clock::now();
    auto initial_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    
    // Save initial distances for comparison later
    vector<double> initial_distances;
    if (rank == 0) {
        initial_distances = sssp_data.distances;
    }
    
    if (rank == 0) {
        cout << "\nInitial SSSP computation time: " << initial_time << " ms" << endl;
        
        // Print tabular results of initial SSSP computation
        cout << "\n-----------------------------------------" << endl;
        cout << "Initial SSSP Results (Source: " << source_vertex << ")" << endl;
        cout << "-----------------------------------------" << endl;
        cout << "| Node | Distance | Parent |" << endl;
        cout << "-----------------------------------------" << endl;
        
        for (int i = 0; i < num_nodes; i++) {
            cout << "| " << setw(4) << i << " | ";
            if (sssp_data.distances[i] == INF) {
                cout << setw(8) << "INF" << " | ";
            } else {
                cout << setw(8) << sssp_data.distances[i] << " | ";
            }
            cout << setw(6) << sssp_data.parents[i] << " |" << endl;
        }
        cout << "-----------------------------------------" << endl;
    }
    
    // Generate random edge updates
    vector<EdgeUpdate> updates;
    if (rank == 0) {
        // Select 20% of nodes to be affected by updates
        int num_affected_nodes = max(1, static_cast<int>(0.2 * num_nodes));
        vector<int> affected_nodes;
        
        random_device rd;
        mt19937 gen(rd());
        
        // Create a vector of all node indices and shuffle it
        vector<int> all_nodes(num_nodes);
        iota(all_nodes.begin(), all_nodes.end(), 0); // Fill with 0, 1, 2, ..., num_nodes-1
        shuffle(all_nodes.begin(), all_nodes.end(), gen);
        
        // Take the first 20% of nodes as affected
        affected_nodes.assign(all_nodes.begin(), all_nodes.begin() + num_affected_nodes);
        
        cout << "Generating updates for " << num_affected_nodes << " affected nodes (20% of nodes)" << endl;
        cout << "Affected nodes: ";
        for (int node : affected_nodes) {
            cout << node << " ";
        }
        cout << endl;
        
        // Uniform distributions for random selections
        uniform_int_distribution<> node_selector(0, num_affected_nodes - 1);
        uniform_int_distribution<> integer_weight_dist(1, 20);  // Whole numbers 1-20 for insertions
        uniform_int_distribution<> type_dist(0, 1);             // 50% insertions, 50% deletions
        
        // Create set of existing edges for quick lookup
        set<pair<int, int>> existing_edges;
        for (int i = 0; i < num_nodes; i++) {
            for (const auto& edge : graph.adj_list[i]) {
                int j = edge.first;
                if (i < j) { // Add only once (i,j) where i < j to avoid duplicates
                    existing_edges.insert({i, j});
                }
            }
        }
        
        // Create set of potential new edges (not in existing_edges)
        set<pair<int, int>> potential_new_edges;
        for (int i : affected_nodes) {
            for (int j : affected_nodes) {
                if (i < j) { // Only consider (i,j) where i < j to avoid duplicates
                    pair<int, int> edge = {i, j};
                    if (existing_edges.find(edge) == existing_edges.end()) {
                        potential_new_edges.insert(edge);
                    }
                }
            }
        }
        
        // Convert sets to vectors for random access
        vector<pair<int, int>> existing_edges_vec(existing_edges.begin(), existing_edges.end());
        vector<pair<int, int>> potential_new_edges_vec(potential_new_edges.begin(), potential_new_edges.end());
        
        // Calculate how many updates to generate
        int num_insertions = min(static_cast<int>(potential_new_edges_vec.size()), 
                                 max(1, static_cast<int>(num_affected_nodes * 2)));
        int num_deletions = min(static_cast<int>(existing_edges_vec.size()), 
                               max(1, static_cast<int>(num_affected_nodes * 2)));
        
        cout << "Generating " << num_insertions << " insertions and " 
             << num_deletions << " deletions" << endl;
        
        // Generate insertion updates (create new edges)
        if (!potential_new_edges_vec.empty()) {
            shuffle(potential_new_edges_vec.begin(), potential_new_edges_vec.end(), gen);
            for (int i = 0; i < num_insertions && i < potential_new_edges_vec.size(); i++) {
                int u = potential_new_edges_vec[i].first;
                int v = potential_new_edges_vec[i].second;
                int weight = integer_weight_dist(gen);  // Whole numbers 1-20
                updates.emplace_back(u, v, static_cast<double>(weight), true);
            }
        }
        
        // Generate deletion updates (remove existing edges)
        if (!existing_edges_vec.empty()) {
            shuffle(existing_edges_vec.begin(), existing_edges_vec.end(), gen);
            for (int i = 0; i < num_deletions && i < existing_edges_vec.size(); i++) {
                int u = existing_edges_vec[i].first;
                int v = existing_edges_vec[i].second;
                
                // Find the weight in the original graph
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
        
        // Shuffle the updates to mix insertions and deletions
        shuffle(updates.begin(), updates.end(), gen);
        
        cout << "Generated " << updates.size() << " total updates" << endl;
    }
    
    // Broadcast number of updates
    int num_updates = updates.size();
    MPI_Bcast(&num_updates, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank != 0) {
        updates.resize(num_updates);
    }
    
    // Broadcast updates
    for (int i = 0; i < num_updates; i++) {
        int src, dest;
        double weight;
        bool is_insertion;
        
        if (rank == 0) {
            src = updates[i].src;
            dest = updates[i].dest;
            weight = updates[i].weight;
            is_insertion = updates[i].is_insertion;
        }
        
        MPI_Bcast(&src, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&dest, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&weight, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&is_insertion, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        
        if (rank != 0) {
            updates[i] = EdgeUpdate(src, dest, weight, is_insertion);
        }
    }
    
    // Process updates and measure time
    start_time = chrono::high_resolution_clock::now();
    process_updates(graph, updates, part_info, sssp_data, source_vertex, rank, num_procs);
    end_time = chrono::high_resolution_clock::now();
    auto update_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    
    if (rank == 0) {
        cout << "\nUpdate processing time: " << update_time << " ms" << endl;
        cout << "Time per update: " << (double)update_time / num_updates << " ms" << endl;
        
        // Print update statistics
        cout << "\n-----------------------------------------" << endl;
        cout << "Updates Statistics" << endl;
        cout << "-----------------------------------------" << endl;
        cout << "Total updates processed: " << num_updates << endl;
        int insertions = 0, deletions = 0;
        for (const auto& update : updates) {
            if (update.is_insertion) insertions++;
            else deletions++;
        }
        cout << "Insertion updates: " << insertions << endl;
        cout << "Deletion updates: " << deletions << endl;
        
        // Display a sample of the first few updates
        const int sample_size = min(10, num_updates);
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
        
        // Print updated SSSP results
        cout << "\n-----------------------------------------" << endl;
        cout << "Updated SSSP Results (Source: " << source_vertex << ")" << endl;
        cout << "-----------------------------------------" << endl;
        cout << "| Node | Distance | Parent |" << endl;
        cout << "-----------------------------------------" << endl;
        
        for (int i = 0; i < num_nodes; i++) {
            cout << "| " << setw(4) << i << " | ";
            if (sssp_data.distances[i] == INF) {
                cout << setw(8) << "INF" << " | ";
            } else {
                cout << setw(8) << sssp_data.distances[i] << " | ";
            }
            cout << setw(6) << sssp_data.parents[i] << " |" << endl;
        }
        cout << "-----------------------------------------" << endl;
        
        cout << "-----------------------------------------" << endl;
        
        // Print comparison of distances
        cout << "\n-----------------------------------------" << endl;
        cout << "Distance Changes After Updates" << endl;
        cout << "-----------------------------------------" << endl;
        cout << "| Node | Initial Dist | Final Dist | Change |" << endl;
        cout << "-----------------------------------------" << endl;
        
        int changed_count = 0;
        for (int i = 0; i < num_nodes; i++) {
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
        
        cout << "Number of updates: " << num_updates << endl;
        cout << "Distributed update time: " << update_time << " ms" << endl;
        cout << "Distributed time per update: " << (double)update_time / num_updates << " ms" << endl;
    }
    
    MPI_Finalize();
    return 0;
} 