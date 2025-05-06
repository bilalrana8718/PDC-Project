#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <fstream>
#include <sstream>
#include <mpi.h>
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

struct PartitionInfo {
    int partition_id;
    vector<int> local_vertices;
    vector<int> ghost_vertices;
    unordered_map<int, int> global_to_local;
    vector<int> local_to_global;
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
    bool is_insertion;  
    EdgeUpdate() : src(-1), dest(-1), weight(0.0), is_insertion(false) {}  
    EdgeUpdate(int s, int d, double w, bool ins) 
        : src(s), dest(d), weight(w), is_insertion(ins) {}
};

Graph read_graph(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
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
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        graph.adj_list[mapped_u].emplace_back(mapped_v, weight);
        graph.adj_list[mapped_v].emplace_back(mapped_u, weight); // Undirected graph
    }

    file.close();
    return graph;
}

vector<int> read_partitions(const string& partition_file, int num_nodes) {
    vector<int> partition(num_nodes, 0);  // Default all nodes to partition 0
    
    ifstream file(partition_file);
    if (!file.is_open()) {
        cerr << "Error opening partition file: " << partition_file << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    for (int i = 0; i < num_nodes; i++) {
        int part_id;
        if (file >> part_id) {
            partition[i] = part_id;
        } else {
            cout << "Note: Using default partition (0) for node " << i 
                 << " and any remaining nodes" << endl;
            break;
        }
    }
    file.close();
    
    return partition;
}

PartitionInfo initialize_partition(const Graph& graph, const vector<int>& partition, int rank, int num_partitions) {
    PartitionInfo info;
    info.partition_id = rank;
    
    vector<int> partition_to_rank(num_partitions, -1);
    vector<int> unique_partitions;
    
    for (int p : partition) {
        if (find(unique_partitions.begin(), unique_partitions.end(), p) == unique_partitions.end()) {
            unique_partitions.push_back(p);
        }
    }
    
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
    
    for (int i = 0; i < graph.num_nodes; i++) {
        int assigned_rank = partition_to_rank[partition[i]];
        if (assigned_rank == rank) {
            info.local_vertices.push_back(i);
            info.global_to_local[i] = info.local_vertices.size() - 1;
            info.local_to_global.push_back(i);
        }
    }
    
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

void print_partition_info(const vector<int>& partition, int num_nodes, int num_procs) {
    cout << "\n-----------------------------------------" << endl;
    cout << "Partition Information" << endl;
    cout << "-----------------------------------------" << endl;
    cout << "Number of partitions: " << num_procs << endl;
    
    vector<int> nodes_per_partition(num_procs, 0);
    for (int i = 0; i < num_nodes; i++) {
        nodes_per_partition[partition[i]]++;
    }
    
    cout << "-----------------------------------------" << endl;
    cout << "| Partition | Node Count | Nodes |" << endl;
    cout << "-----------------------------------------" << endl;
    
    for (int p = 0; p < num_procs; p++) {
        cout << "| " << setw(9) << p << " | " << setw(10) << nodes_per_partition[p] << " | ";
        
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

void compute_sssp_distributed(const Graph& graph, const PartitionInfo& part_info, 
                            SSSPData& sssp_data, int source, int rank, int num_procs) {
    // Initialize distances for local vertices
    if (rank == 0) {
        cout << "[Rank " << rank << "] Initializing SSSP with source vertex: " << source << endl;
    }
    
    // Initialize all distances to infinity and parents to -1
    for (size_t i = 0; i < graph.num_nodes; i++) {
        sssp_data.distances[i] = INF;
        sssp_data.parents[i] = -1;
    }
    
    if (rank == 0) {
        sssp_data.distances[source] = 0.0;
        sssp_data.parents[source] = source;
        cout << "[Rank " << rank << "] Setting distance[" << source << "] = 0.0" << endl;
    }
    
    // Make sure all processes have the initial state
    for (int i = 0; i < graph.num_nodes; i++) {
        MPI_Bcast(&sssp_data.distances[i], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&sssp_data.parents[i], 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    
    if (rank == 0) cout << "[Rank " << rank << "] Starting iterative SSSP computation..." << endl;
    
    vector<double> new_distances(graph.num_nodes, INF);
    vector<int> new_parents(graph.num_nodes, -1);
    
    bool global_changes;
    int iterations = 0;
    const int MAX_ITERATIONS = graph.num_nodes;
    
    do {
        iterations++;
        if (rank == 0 && iterations % 5 == 0) {
            cout << "[Rank " << rank << "] SSSP iteration " << iterations << "..." << endl;
        }
        
        for (int i = 0; i < graph.num_nodes; i++) {
            new_distances[i] = sssp_data.distances[i];
            new_parents[i] = sssp_data.parents[i];
        }
        
        bool local_changes = false;
        int local_relaxations = 0;
        
        // Process local vertices (edge relaxation)
        for (size_t i = 0; i < part_info.local_vertices.size(); i++) {
            int u = part_info.local_vertices[i];
            
            if (sssp_data.distances[u] == INF) continue;
            
            for (const auto& edge : graph.adj_list[u]) {
                int v = edge.first;
                double weight = edge.second;
                double potential_dist = sssp_data.distances[u] + weight;
                
                // Relaxation: If we found a shorter path to v
                if (potential_dist < new_distances[v]) {
                    new_distances[v] = potential_dist;
                    new_parents[v] = u;
                    local_changes = true;
                    local_relaxations++;
                    
                    if (rank == 0 && (v == u || (new_parents[v] == u && new_parents[u] == v))) {
                        cout << "[Warning] Potential cycle detected between " << u << " and " << v << endl;
                    }
                }
            }
        }
        
        int total_relaxations = 0;
        MPI_Reduce(&local_relaxations, &total_relaxations, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0 && total_relaxations > 0 && iterations % 5 == 0) {
            cout << "[Rank " << rank << "] Total relaxations in iteration " << iterations 
                 << ": " << total_relaxations << endl;
        }
        
        MPI_Allreduce(&local_changes, &global_changes, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
        
        if (global_changes) {
            for (int r = 0; r < num_procs; r++) {
                for (int i = 0; i < graph.num_nodes; i++) {
                    double dist_i = (r == rank) ? new_distances[i] : INF;
                    int parent_i = (r == rank) ? new_parents[i] : -1;
                    
                    MPI_Bcast(&dist_i, 1, MPI_DOUBLE, r, MPI_COMM_WORLD);
                    MPI_Bcast(&parent_i, 1, MPI_INT, r, MPI_COMM_WORLD);
                    
                    if (dist_i < sssp_data.distances[i]) {
                        sssp_data.distances[i] = dist_i;
                        sssp_data.parents[i] = parent_i;
                        
                        new_distances[i] = dist_i;
                        new_parents[i] = parent_i;
                    }
                }
            }
        }
        
        // Enforce a maximum number of iterations to prevent infinite loops
        if (iterations >= MAX_ITERATIONS) {
            if (rank == 0) {
                cout << "[Rank " << rank << "] Warning: SSSP computation reached maximum iterations (" 
                     << MAX_ITERATIONS << "). There might be negative cycles in the graph." << endl;
            }
            break;
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
    } while (global_changes);
    
    if (rank == 0) {
        cout << "[Rank " << rank << "] SSSP computation converged after " 
             << iterations << " iterations." << endl;
        
        bool has_cycle = false;
        for (int i = 0; i < graph.num_nodes; i++) {
            if (i != source && sssp_data.parents[i] != -1) {
                if (i == sssp_data.parents[i]) {
                    cout << "[Rank " << rank << "] Error: Node " << i << " is its own parent!" << endl;
                    has_cycle = true;
                }
                
                int parent = sssp_data.parents[i];
                if (parent != -1 && parent != source && sssp_data.parents[parent] == i) {
                    cout << "[Rank " << rank << "] Error: Cycle detected between nodes " << i 
                         << " and " << parent << endl;
                    has_cycle = true;
                }
            }
        }
        
        // if (!has_cycle) {
        //     // cout << "[Rank " << rank << "] SSSP tree verification passed: No cycles detected." << endl;
        // }
    }
}

void identify_affected_nodes(const Graph& graph, const vector<EdgeUpdate>& updates,
                           SSSPData& sssp_data, int source, int rank, const PartitionInfo& part_info) {
    // Reset affected status
    fill(sssp_data.is_affected.begin(), sssp_data.is_affected.end(), false);
    
    for (size_t i = 0; i < updates.size(); i++) {
        const EdgeUpdate& update = updates[i];
        
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
    
    vector<int> local_affected(graph.num_nodes, 0);
    vector<int> global_affected(graph.num_nodes, 0);
    
    for (int i = 0; i < graph.num_nodes; i++) {
        if (sssp_data.is_affected[i]) {
            local_affected[i] = 1;
        }
    }
    
    MPI_Allreduce(local_affected.data(), global_affected.data(), graph.num_nodes, 
                 MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    
    for (int i = 0; i < graph.num_nodes; i++) {
        if (global_affected[i] == 1) {
            sssp_data.is_affected[i] = true;
        }
    }
    
    // Second pass: Propagate affected status to descendants in the SSSP tree
    // Using fixed number of iterations to avoid deadlocks
    const int MAX_PROPAGATION_ITERATIONS = min(10, graph.num_nodes);
    
    for (int iter = 0; iter < MAX_PROPAGATION_ITERATIONS; iter++) {

        fill(local_affected.begin(), local_affected.end(), 0);
        
        for (size_t idx = 0; idx < part_info.local_vertices.size(); idx++) {
            int node = part_info.local_vertices[idx];
            if (node < 0 || node >= graph.num_nodes) continue; 
            
            if (sssp_data.is_affected[node]) {
                for (const auto& edge : graph.adj_list[node]) {
                    int child = edge.first;
                    if (child < 0 || child >= graph.num_nodes) continue; 
                    
                    if (sssp_data.parents[child] == node && !sssp_data.is_affected[child]) {
                        local_affected[child] = 1;
                        
                        if (sssp_data.distances[node] == INF) {
                            sssp_data.distances[child] = INF;
                        }
                    }
                }
            }
        }
        
        fill(global_affected.begin(), global_affected.end(), 0);
        MPI_Allreduce(local_affected.data(), global_affected.data(), graph.num_nodes, 
                     MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        
        bool any_changes = false;
        
        for (int i = 0; i < graph.num_nodes; i++) {
            if (global_affected[i] == 1 && !sssp_data.is_affected[i]) {
                sssp_data.is_affected[i] = true;
                any_changes = true;
            }
        }
        
        bool global_done;
        MPI_Allreduce(&any_changes, &global_done, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
        
        if (!global_done) {
            break; 
        }
    }
    
    int local_affected_count = 0;
    int local_partition_affected = 0;
    
    for (int i = 0; i < graph.num_nodes; i++) {
        if (sssp_data.is_affected[i]) {
            local_affected_count++;
        }
    }
    
    for (size_t i = 0; i < part_info.local_vertices.size(); i++) {
        int node = part_info.local_vertices[i];
        if (node < 0 || node >= graph.num_nodes) continue; 
        
        if (sssp_data.is_affected[node]) {
            local_partition_affected++;
        }
    }
    
    int global_affected_count = 0;
    MPI_Reduce(&local_affected_count, &global_affected_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    
    cout << "\n[Rank " << rank << "] Has " << local_partition_affected << " affected nodes in its partition (";
    if (part_info.local_vertices.size() > 0) {
        cout << (100.0 * local_partition_affected / part_info.local_vertices.size());
    } else {
        cout << "0.0";
    }
    cout << "%)\n" << endl;
}

void compute_sssp_incremental(const Graph& graph, const PartitionInfo& part_info, 
                            SSSPData& sssp_data, int source, int rank, int num_procs) {
    if (rank == 0) {
        cout << "[Rank " << rank << "] Starting incremental SSSP computation for affected nodes..." << endl;
    }
    
    if (source < 0 || source >= graph.num_nodes) {
        if (rank == 0) {
            cerr << "[Error] Invalid source vertex: " << source << endl;
        }
        return;
    }
    
    sssp_data.distances[source] = 0;
    sssp_data.parents[source] = source;
    sssp_data.is_affected[source] = false;
    
    vector<double> new_distances = sssp_data.distances;
    vector<int> new_parents = sssp_data.parents;
    
    bool global_changes;
    int iterations = 0;
    const int MAX_ITERATIONS = min(100, graph.num_nodes); 
    
    int local_affected_count = 0;
    for (size_t i = 0; i < part_info.local_vertices.size(); i++) {
        int node = part_info.local_vertices[i];
        if (node < 0 || node >= graph.num_nodes) continue; 
        
        if (sssp_data.is_affected[node]) {
            local_affected_count++;
        }
    }
    
    do {    
        iterations++;
        
        bool local_changes = false;
        int local_relaxations = 0;
        
        for (size_t i = 0; i < part_info.local_vertices.size(); i++) {
            int u = part_info.local_vertices[i];
            if (u < 0 || u >= graph.num_nodes) continue; 
            
            if (sssp_data.distances[u] == INF) continue;
            
            bool process_vertex = sssp_data.is_affected[u];
            if (!process_vertex) {
                for (const auto& edge : graph.adj_list[u]) {
                    int neighbor = edge.first;
                    if (neighbor < 0 || neighbor >= graph.num_nodes) continue; 
                    
                    if (sssp_data.is_affected[neighbor]) {
                        process_vertex = true;
                        break;
                    }
                }
            }
            
            if (!process_vertex) continue;
            
            for (const auto& edge : graph.adj_list[u]) {
                int v = edge.first;
                if (v < 0 || v >= graph.num_nodes) continue; 
                
                double weight = edge.second;
                double potential_dist = sssp_data.distances[u] + weight;
                
                if (potential_dist < new_distances[v]) {
                    new_distances[v] = potential_dist;
                    new_parents[v] = u;
                    local_changes = true;
                    local_relaxations++;
                }
            }
        }
        
        vector<double> send_distances(graph.num_nodes, INF);
        vector<int> send_parents(graph.num_nodes, -1);
        
        for (int i = 0; i < graph.num_nodes; i++) {
            if (new_distances[i] < sssp_data.distances[i]) {
                send_distances[i] = new_distances[i];
                send_parents[i] = new_parents[i];
            }
        }
        
        vector<double> recv_distances(graph.num_nodes, INF);
        vector<int> recv_parents(graph.num_nodes, -1);
        
        MPI_Allreduce(&local_changes, &global_changes, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
        
        if (global_changes) {
            MPI_Allreduce(send_distances.data(), recv_distances.data(), graph.num_nodes, 
                         MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
            
            for (int i = 0; i < graph.num_nodes; i++) {
                bool has_min = false;
                
                if (send_distances[i] < INF && recv_distances[i] < INF) {
                    has_min = (fabs(send_distances[i] - recv_distances[i]) < 1e-9);
                }
                
                int best_rank = has_min ? rank : -1;
                int global_best_rank;
                MPI_Allreduce(&best_rank, &global_best_rank, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
                
                if (global_best_rank >= 0) {
                    int parent_from_best = (rank == global_best_rank) ? send_parents[i] : -1;
                    MPI_Bcast(&parent_from_best, 1, MPI_INT, global_best_rank, MPI_COMM_WORLD);
                    recv_parents[i] = parent_from_best;
                    
                    if (recv_distances[i] < sssp_data.distances[i]) {
                        sssp_data.distances[i] = recv_distances[i];
                        sssp_data.parents[i] = recv_parents[i];
                        sssp_data.is_affected[i] = true;
                        
                        new_distances[i] = recv_distances[i];
                        new_parents[i] = recv_parents[i];
                    }
                }
            }
            
            if (source >= 0 && source < graph.num_nodes) {
                sssp_data.distances[source] = 0;
                sssp_data.parents[source] = source;
                new_distances[source] = 0;
                new_parents[source] = source;
            }
        }
        
        if (iterations >= MAX_ITERATIONS) {
            if (rank == 0) {
                cout << "[Rank " << rank << "] Warning: Incremental SSSP computation reached maximum iterations (" 
                     << MAX_ITERATIONS << "). Forcibly terminating the loop." << endl;
            }
            break;
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        
    } while (global_changes);
    
    if (rank == 0) {
        cout << "[Rank " << rank << "] Incremental SSSP computation converged after " 
             << iterations << " iterations." << endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    int local_processed = 0;
    for (size_t i = 0; i < part_info.local_vertices.size(); i++) {
        int node = part_info.local_vertices[i];
        if (node < 0 || node >= graph.num_nodes) continue; 
        
        if (sssp_data.is_affected[node]) {
            local_processed++;
        }
    }
    
    int total_processed = 0;
    MPI_Reduce(&local_processed, &total_processed, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        cout << "[Rank " << rank << "] Total affected nodes processed: " << total_processed << endl;
    }
}

void process_updates(Graph& graph, const vector<EdgeUpdate>& updates,
                    const PartitionInfo& part_info, SSSPData& sssp_data,
                    int source, int rank, int num_procs,
                    long& graph_update_time, long& identify_time, long& compute_time) {
    if (source < 0 || source >= graph.num_nodes) {
        if (rank == 0) {
            cerr << "[Error] Invalid source vertex: " << source << endl;
        }
        return;
    }
    
    if (rank == 0) {
        cout << "Applying " << updates.size() << " updates to the graph structure..." << endl;
    }
    
    auto start_time = chrono::high_resolution_clock::now();
    
    int invalid_update_count = 0;
    
    for (size_t i = 0; i < updates.size(); i++) {
        const EdgeUpdate& update = updates[i];
        
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
        cout << "[Rank " << rank << "] Warning: " << invalid_update_count 
             << " updates were invalid and skipped" << endl;
    }
    
    graph.num_edges = 0;
    for (int i = 0; i < graph.num_nodes; i++) {
        graph.num_edges += graph.adj_list[i].size();
    }
    graph.num_edges /= 2;
    
    auto graph_update_end = chrono::high_resolution_clock::now();
    graph_update_time = chrono::duration_cast<chrono::milliseconds>(graph_update_end - start_time).count();
    
    if (rank == 0) {
        cout << "Graph structure updated in " << graph_update_time << " ms" << endl;
        cout << "New graph size: " << graph.num_nodes << " nodes, " << graph.num_edges << " edges" << endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD); 
    
    if (rank == 0) {
        cout << "Identifying affected nodes in the SSSP tree..." << endl;
    }
    
    auto identify_start = chrono::high_resolution_clock::now();
    identify_affected_nodes(graph, updates, sssp_data, source, rank, part_info);
    auto identify_end = chrono::high_resolution_clock::now();
    identify_time = chrono::duration_cast<chrono::milliseconds>(identify_end - identify_start).count();
    
    if (rank == 0) {
        cout << "Affected nodes identified in " << identify_time << " ms" << endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD); 
    
    int local_affected_count = 0;
    for (size_t i = 0; i < part_info.local_vertices.size(); i++) {
        int u = part_info.local_vertices[i];
        if (u < 0 || u >= graph.num_nodes) continue; 
        
        if (sssp_data.is_affected[u]) {
            local_affected_count++;
        }
    }
    
    int global_affected_count = 0;
    MPI_Reduce(&local_affected_count, &global_affected_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        cout << "Total affected nodes across all partitions: " << global_affected_count << endl;
    }
    
    MPI_Bcast(&global_affected_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (global_affected_count > 0) {
        if (rank == 0) {
            cout << "\n[Incremental Update] Performing incremental SSSP computation for affected nodes..." << endl;
        }
        
        auto compute_start = chrono::high_resolution_clock::now();
        compute_sssp_incremental(graph, part_info, sssp_data, source, rank, num_procs);
        auto compute_end = chrono::high_resolution_clock::now();
        compute_time = chrono::duration_cast<chrono::milliseconds>(compute_end - compute_start).count();
        
        if (rank == 0) {
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
        }
    } else {
        if (rank == 0) {
            cout << "No nodes affected by the updates, SSSP tree remains unchanged." << endl;
            cout << "Total update processing time: " << (graph_update_time + identify_time) << " ms" << endl;
        }
        compute_time = 0; 
    }
    
    bool found_cycle = false;
    int relaxable_edges = 0;
    
    for (size_t i = 0; i < part_info.local_vertices.size(); i++) {
        int u = part_info.local_vertices[i];
        if (u < 0 || u >= graph.num_nodes) continue;
        
        if (u != source && sssp_data.parents[u] == u) {
            cout << "[Rank " << rank << "] Warning: Node " << u << " is its own parent!" << endl;
            found_cycle = true;
        }
        
        if (sssp_data.distances[u] != INF) {
            for (const auto& edge : graph.adj_list[u]) {
                int v = edge.first;
                if (v < 0 || v >= graph.num_nodes) continue;
                
                double weight = edge.second;
                double potential_dist = sssp_data.distances[u] + weight;
                
                if (potential_dist < sssp_data.distances[v] - 1e-9) {
                    relaxable_edges++;
                    if (relaxable_edges <= 5) {
                        cout << "[Rank " << rank << "] Warning: Edge (" << u << " -> " << v 
                             << ") is relaxable: " << sssp_data.distances[u] << " + " 
                             << weight << " < " << sssp_data.distances[v] << endl;
                    }
                }
            }
        }
    }
    
    int global_relaxable_edges = 0;
    MPI_Reduce(&relaxable_edges, &global_relaxable_edges, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    bool global_found_cycle;
    MPI_Reduce(&found_cycle, &global_found_cycle, 1, MPI_C_BOOL, MPI_LOR, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        if (global_relaxable_edges > 0) {
            cout << "[Warning] Final solution has " << global_relaxable_edges 
                 << " relaxable edges! Solution may be suboptimal." << endl;
        } else {
            cout << "[Validation] Final SSSP solution is optimal (no relaxable edges found)." << endl;
        }
        
        if (global_found_cycle) {
            cout << "[Warning] Found cycles in the parent pointers! Solution may be invalid." << endl;
        } else {
            cout << "[Validation] No cycles found in the SSSP tree." << endl;
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
}

Graph read_metis_graph(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening METIS file: " << filename << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
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
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        istringstream line_ss(line);
        int neighbor;
        double weight;
        
        while (line_ss >> neighbor) {
            neighbor--;
            
            if (weighted) {
                if (!(line_ss >> weight)) {
                    cerr << "Error reading weight for edge (" << i << ", " << neighbor << ")" << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            } else {
                weight = 1.0; 
            }
            
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
    
    auto program_start_time = chrono::high_resolution_clock::now();
    
    if (rank == 0) {
        cout << "\n===================================================" << endl;
        cout << "   Distributed SSSP Computation with " << num_procs << " processes" << endl;
        cout << "===================================================" << endl;
    }
    
    if (argc < 3) {
        if (rank == 0) {
            cerr << "Usage: " << argv[0] << " <graph_file> <source_vertex> [partition_file]" << endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    string graph_file = argv[1];
    int source_vertex = atoi(argv[2]);
    
    string partition_file = "graph.metis.part.4";
    
    if (argc >= 4) {
        partition_file = argv[3];
    }
    
    if (rank == 0) {
        cout << "Graph file: " << graph_file << endl;
        cout << "Source vertex: " << source_vertex << endl;
        cout << "Partition file: " << partition_file << endl;
    }
    
    Graph graph(0);
    int num_nodes = 0;
    int num_edges = 0;
    
    // Timing variables
    long read_time = 0;
    long bcast_time = 0;
    long part_time = 0;
    long initial_time = 0;
    long update_time = 0;
    long graph_update_time = 0;
    long identify_time = 0;
    long compute_time = 0;
    
    if (rank == 0) {
        cout << "\n[Phase 1: Graph Loading] [Rank " << rank << "]" << endl;
        cout << "-------------------------------------------------" << endl;
        
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
        read_time = chrono::duration_cast<chrono::milliseconds>(read_end - read_start).count();
        
        num_nodes = graph.num_nodes;
        num_edges = graph.num_edges;
        cout << "Graph read successfully in " << read_time << " ms" << endl;
        cout << "Nodes: " << num_nodes << ", Edges: " << num_edges << endl;
        
        print_graph(graph);
    }
    
    if (rank == 0) cout << "\n[Phase 2: Graph Distribution] [Rank " << rank << "]" << endl;
    if (rank == 0) cout << "-------------------------------------------------" << endl;
    
    auto bcast_start = chrono::high_resolution_clock::now();
    
    MPI_Bcast(&num_nodes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);
    

    // Initialize graph on all processes
    if (rank != 0) {
        graph = Graph(num_nodes);
        graph.num_edges = num_edges;
    }
    
    // Broadcast graph structure
    if (rank == 0) cout << "[Rank " << rank << "] Broadcasting graph structure to all processes..." << endl;
    
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
    
    auto bcast_end = chrono::high_resolution_clock::now();
    bcast_time = chrono::duration_cast<chrono::milliseconds>(bcast_end - bcast_start).count();
    
    if (rank == 0) {
        cout << "[Rank " << rank << "] Graph distribution completed in " 
             << bcast_time << " ms" << endl;
    }
    
    if (rank == 0) cout << "\n[Phase 3: Partitioning] [Rank " << rank << "]" << endl;
    if (rank == 0) cout << "-------------------------------------------------" << endl;
    
    auto part_start = chrono::high_resolution_clock::now();
    
    vector<int> partition(num_nodes);
    if (rank == 0) {
        cout << "[Rank " << rank << "] Reading partition from: " << partition_file << endl;
        partition = read_partitions(partition_file, num_nodes);
    }
    
    // Broadcast partition information
    if (rank == 0) cout << "[Rank " << rank << "] Broadcasting partition information..." << endl;
    MPI_Bcast(partition.data(), num_nodes, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        // print_partition_info(partition, num_nodes, num_procs);
    }
    
    PartitionInfo part_info = initialize_partition(graph, partition, rank, num_procs);
    
    auto part_end = chrono::high_resolution_clock::now();
    part_time = chrono::duration_cast<chrono::milliseconds>(part_end - part_start).count();
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        cout << "[Rank " << rank << "] Partitioning completed in " << part_time << " ms" << endl;
    }
    
    for (int r = 0; r < num_procs; r++) {
        if (r == rank) {

            cout << "[Rank " << rank << "] Local vertices: ";
            for (int i = 0; i < min(10, (int)part_info.local_vertices.size()); i++) {
                cout << part_info.local_vertices[i] << " ";
            }
            if (part_info.local_vertices.size() > 10) cout << "...";
            cout << endl;

        }
        MPI_Barrier(MPI_COMM_WORLD); 
    }
    
    SSSPData sssp_data(num_nodes);
    
    if (rank == 0) {
        cout << "\n[Phase 4: Initial SSSP Computation] [Rank " << rank << "]\n" << endl;
    }
    
    auto start_time = chrono::high_resolution_clock::now();
    compute_sssp_distributed(graph, part_info, sssp_data, source_vertex, rank, num_procs);
    auto end_time = chrono::high_resolution_clock::now();
    initial_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    
    vector<double> initial_distances;
    if (rank == 0) {
        initial_distances = sssp_data.distances;
    }
    
    if (rank == 0) {
        cout << "\nInitial SSSP computation time: " << initial_time << " ms" << endl;
        
        cout << "\n" << endl;
        cout << "Initial SSSP Results (Source: " << source_vertex << ")" << endl;
        cout << " " << endl;
        cout << "| Node | Distance | Parent |" << endl;
        cout << " " << endl;
        
        for (int i = 0; i < num_nodes; i++) {
            cout << "| " << setw(4) << i << " | ";
            if (sssp_data.distances[i] == INF) {
                cout << setw(8) << "INF" << " | ";
            } else {
                cout << setw(8) << sssp_data.distances[i] << " | ";
            }
            cout << setw(6) << sssp_data.parents[i] << " |" << endl;
        }
        cout << endl;
    }
    
    vector<EdgeUpdate> updates;
    if (rank == 0) {
        int num_affected_nodes = max(1, static_cast<int>(0.05 * num_nodes));
        vector<int> affected_nodes;
        
        random_device rd;   
        mt19937 gen(rd());
        
        vector<int> all_nodes(num_nodes);
        iota(all_nodes.begin(), all_nodes.end(), 0); 
        shuffle(all_nodes.begin(), all_nodes.end(), gen);
        
        affected_nodes.assign(all_nodes.begin(), all_nodes.begin() + num_affected_nodes);
        
        cout << "Generating updates for " << num_affected_nodes << " affected nodes (20% of nodes)" << endl;
        // cout << "Affected nodes: ";
        // for (int node : affected_nodes) {
        //     cout << node << " ";
        // }
        // cout << endl;
        
        uniform_int_distribution<> node_selector(0, num_affected_nodes - 1);
        uniform_int_distribution<> integer_weight_dist(1, 10);  
        uniform_int_distribution<> type_dist(0, 1);             
        
        set<pair<int, int>> existing_edges;
        for (int i = 0; i < num_nodes; i++) {
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
                int weight = integer_weight_dist(gen);  // Whole numbers 1-20
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
    
    if (rank == 0) {
        cout << "\n[Phase 5: Processing Updates] [Rank " << rank << "]\n" << endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    start_time = chrono::high_resolution_clock::now();
    process_updates(graph, updates, part_info, sssp_data, source_vertex, rank, num_procs,
                    graph_update_time, identify_time, compute_time);
    end_time = chrono::high_resolution_clock::now();
    update_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        cout << "\nTotal update processing time: " << update_time << " ms" << endl;
        cout << "Time per update: " << (double)update_time / num_updates << " ms" << endl;
        
        cout << "\nUpdates Statistics\n" << endl;
        cout << "Total updates processed: " << num_updates << endl;
        int insertions = 0, deletions = 0;
        for (const auto& update : updates) {
            if (update.is_insertion) insertions++;
            else deletions++;
        }
        cout << "Insertion updates: " << insertions << endl;
        cout << "Deletion updates: " << deletions << endl;
        
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
        
        auto program_end_time = chrono::high_resolution_clock::now();
        auto total_time = chrono::duration_cast<chrono::milliseconds>(program_end_time - program_start_time).count();
        
        cout << "\n===================================================" << endl;
        cout << "Performance Summary" << endl;
        cout << "===================================================" << endl;
        cout << "Graph Reading Time: " << read_time << " ms" << endl;
        cout << "Graph Distribution Time: " << bcast_time << " ms" << endl;
        cout << "Partitioning Time: " << part_time << " ms" << endl;
        cout << "Initial SSSP Time: " << initial_time << " ms" << endl;
        cout << "Update Processing Breakdown:" << endl;
        cout << "  - Graph Structure Update: " << graph_update_time << " ms" << endl;
        cout << "  - Affected Nodes Identification: " << identify_time << " ms" << endl;
        cout << "  - Incremental SSSP Computation: " << compute_time << " ms" << endl;
        cout << "Total Update Processing Time: " << update_time << " ms" << endl;
        cout << "Average Time per Update: " << (double)update_time / num_updates << " ms" << endl;
        cout << "Average Time per Node: " << (double)initial_time / num_nodes << " ms (initial SSSP)" << endl;
        cout << "--------------------------------------------------" << endl;
        cout << "Total Execution Time: " << total_time << " ms" << endl;
        cout << "===================================================" << endl;
        
        cout << "\n===================================================" << endl;
        cout << "Program Completed Successfully" << endl;
        cout << "Total execution time: " << total_time << " ms" << endl;
        cout << "===================================================" << endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    MPI_Finalize();
    return 0;
} 