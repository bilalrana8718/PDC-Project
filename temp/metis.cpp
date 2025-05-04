#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <cstdlib>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <cstring>

using namespace std;
namespace fs = std::filesystem;

// Function to create a temporary edge list file from weighted graph
void create_edge_list(const string& input_file, const string& temp_file, int& num_nodes, long long& num_edges, bool weighted = true) {
    auto start_time = chrono::high_resolution_clock::now();
    
    ifstream infile(input_file);
    if (!infile) {
        cerr << "Error: Unable to open input file " << input_file << endl;
        exit(1);
    }
    
    ofstream outfile(temp_file);
    if (!outfile) {
        cerr << "Error: Unable to create temporary file " << temp_file << endl;
        exit(1);
    }
    
    string line;
    int max_node = 0;
    long long line_count = 0;
    num_edges = 0;
    
    cout << "Processing input graph.txt file and symmetrizing edges..." << endl;
    
    // Track edges to avoid duplicates
    unordered_map<int, unordered_map<int, double>> edge_cache;
    const int CACHE_SIZE = 1000000; // Max nodes in cache before flushing
    
    auto flush_cache = [&]() {
        for (const auto& [u, neighbors] : edge_cache) {
            for (const auto& [v, weight] : neighbors) {
                outfile << u << " " << v << " " << weight << "\n";
                num_edges++;
            }
        }
        edge_cache.clear();
    };
    
    while (getline(infile, line)) {
        line_count++;
        if (line_count % 1000000 == 0) {
            cout << "Processed " << line_count/1000000 << "M lines" << endl;
            flush_cache(); // Periodically flush to avoid memory buildup
        }
        
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;
        
        istringstream iss(line);
        int u, v;
        double weight = 1.0; // Default weight
        
        if (weighted) {
            if (!(iss >> u >> v >> weight)) continue;
        } else {
            if (!(iss >> u >> v)) continue;
        }
        
        // Ensure node IDs are non-negative
        if (u < 0 || v < 0) continue;
        
        max_node = max(max_node, max(u, v));
        
        // Add both directions (symmetrize)
        edge_cache[u][v] = weight;
        edge_cache[v][u] = weight; // Symmetrize with the same weight
        
        // If cache gets too large, flush it
        if (edge_cache.size() > CACHE_SIZE) {
            flush_cache();
        }
    }
    
    // Flush any remaining edges
    flush_cache();
    
    infile.close();
    outfile.close();
    
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    
    num_nodes = max_node + 1;
    cout << "Graph has " << num_nodes << " nodes and approximately " << num_edges << " edges." << endl;
    cout << "Phase 1 execution time: " << duration << " ms" << endl;
}

// Efficient batched approach to generate METIS format file
void generate_metis_file(const string& temp_file, const string& metis_file, int num_nodes, long long num_edges, bool weighted = true) {
    auto start_time = chrono::high_resolution_clock::now();
    
    ofstream metisfile(metis_file);
    if (!metisfile) {
        cerr << "Error: Unable to create METIS file" << endl;
        exit(1);
    }
    
    // Write header: number of nodes, edges, and format (weighted or not)
    metisfile << num_nodes << " " << (num_edges/2);
    if (weighted) metisfile << " 001"; // Add format indicator for edge weights
    metisfile << "\n";
    
    cout << "Writing METIS file..." << endl;
    
    const int BATCH_SIZE = 10000; // Process this many nodes at once
    
    for (int start_node = 0; start_node < num_nodes; start_node += BATCH_SIZE) {
        int end_node = min(start_node + BATCH_SIZE, num_nodes);
        
        // For each batch of nodes, read the entire edge file once
        vector<unordered_map<int, double>> batch_neighbors(BATCH_SIZE);
        
        ifstream edgefile(temp_file);
        string line;
        
        while (getline(edgefile, line)) {
            istringstream iss(line);
            int u, v;
            double weight = 1.0;
            
            if (weighted) {
                if (!(iss >> u >> v >> weight)) continue;
            } else {
                if (!(iss >> u >> v)) continue;
            }
            
            // Check if either endpoint is in our batch
            if (u >= start_node && u < end_node) {
                batch_neighbors[u - start_node][v] = weight;
            }
        }
        
        // Write this batch to the METIS file
        for (int i = 0; i < end_node - start_node; i++) {
            const auto& neighbors = batch_neighbors[i];
            for (const auto& [v, weight] : neighbors) {
                metisfile << (v + 1); // METIS uses 1-based indexing
                if (weighted) {
                    metisfile << " " << weight;
                }
                metisfile << " ";
            }
            metisfile << "\n";
        }
        
        // Explicitly clear memory
        vector<unordered_map<int, double>>().swap(batch_neighbors);
    }
    
    metisfile.close();
    
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    
    cout << "METIS file created successfully." << endl;
    cout << "Phase 2 execution time: " << duration << " ms" << endl;
}

// Run gpmetis to partition graph
void run_metis_partition(const string& metis_file, int parts) {
    auto start_time = chrono::high_resolution_clock::now();
    
    // On Linux, try to locate gpmetis in PATH
    string gpmetis_cmd;
    FILE* which_output = popen("which gpmetis", "r");
    if (which_output) {
        char buffer[1024];
        if (fgets(buffer, sizeof(buffer), which_output) != NULL) {
            // Remove trailing newline
            size_t len = strlen(buffer);
            if (len > 0 && buffer[len-1] == '\n') {
                buffer[len-1] = '\0';
            }
            gpmetis_cmd = buffer;
        }
        pclose(which_output);
    }
    
    if (gpmetis_cmd.empty()) {
        gpmetis_cmd = "gpmetis"; // Default to just the executable name
    }
    
    // Build and execute the command
    string command = "\"" + gpmetis_cmd + "\" -ptype=kway \"" + metis_file + "\" " + to_string(parts);
    cout << "Executing: " << command << endl;
    
    int ret = system(command.c_str());
    if (ret != 0) {
        cerr << "Error: METIS partitioning failed with return code " << ret << endl;
        cerr << "Make sure gpmetis is installed on your Ubuntu system." << endl;
        cerr << "You can install it with: sudo apt-get install metis" << endl;
        exit(1);
    }
    
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    
    cout << "METIS partitioning completed successfully." << endl;
    cout << "Phase 3 execution time: " << duration << " ms" << endl;
}

// Main function
int main(int argc, char* argv[]) {
    auto start_time = chrono::high_resolution_clock::now();
    
    // Default values
    string input_file = "graph.txt";
    string metis_file = "graph.metis";
    int num_partitions = 4;
    bool weighted = true; // Default to weighted since we're processing graph.txt
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-i" || arg == "--input") {
            if (i + 1 < argc) input_file = argv[++i];
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) metis_file = argv[++i];
        } else if (arg == "-p" || arg == "--partitions") {
            if (i + 1 < argc) num_partitions = atoi(argv[++i]);
        } else if (arg == "-w" || arg == "--weighted") {
            weighted = true;
        } else if (arg == "-u" || arg == "--unweighted") {
            weighted = false;
        } else if (arg == "-h" || arg == "--help") {
            cout << "METIS Graph Partitioner" << endl;
            cout << "======================" << endl;
            cout << "Usage: " << argv[0] << " [options]" << endl;
            cout << "Options:" << endl;
            cout << "  -i, --input FILE         Input graph file (default: graph.txt)" << endl;
            cout << "  -o, --output FILE        Output METIS graph file (default: graph.metis)" << endl;
            cout << "  -p, --partitions NUM     Number of partitions (default: 4)" << endl;
            cout << "  -w, --weighted           Process input as a weighted graph (default)" << endl;
            cout << "  -u, --unweighted         Process input as an unweighted graph" << endl;
            cout << "  -h, --help               Display this help message" << endl;
            return 0;
        }
    }
    
    cout << "METIS Graph Partitioning Pipeline" << endl;
    cout << "=================================" << endl;
    cout << "Input file: " << input_file << endl;
    cout << "Output METIS file: " << metis_file << endl;
    cout << "Number of partitions: " << num_partitions << endl;
    cout << "Graph type: " << (weighted ? "Weighted" : "Unweighted") << endl;
    
    try {
        // Verify input file exists
        if (!fs::exists(input_file)) {
            cerr << "Error: Input file '" << input_file << "' not found." << endl;
            return 1;
        }
        
        // Create a temporary file for edges
        string temp_file = metis_file + ".temp_edges";
        
        int num_nodes;
        long long num_edges;
        
        cout << "\nPhase 1: Converting graph to edge list format" << endl;
        cout << "-------------------------------------------" << endl;
        create_edge_list(input_file, temp_file, num_nodes, num_edges, weighted);
        
        cout << "\nPhase 2: Converting to METIS format" << endl;
        cout << "--------------------------------" << endl;
        generate_metis_file(temp_file, metis_file, num_nodes, num_edges, weighted);
        
        // Delete temporary file
        if (fs::exists(temp_file)) {
            fs::remove(temp_file);
        }
        
        cout << "\nPhase 3: Partitioning with METIS" << endl;
        cout << "-----------------------------" << endl;
        run_metis_partition(metis_file, num_partitions);
        
        // Check if the partition file was created successfully
        string partition_file = metis_file + ".part." + to_string(num_partitions);
        ifstream part_check(partition_file);
        if (!part_check) {
            cerr << "Error: Partition file was not created successfully." << endl;
            return 1;
        }
        
        cout << "\nPartitioning complete!" << endl;
        cout << "Output partition file: " << partition_file << endl;
        
        auto end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
        cout << "\nTotal execution time: " << duration << " ms" << endl;
        
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}
