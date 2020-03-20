#ifndef _SERVER_H_
#define _SERVER_H_

#include <vector>
#include <string>
#include <random>
using namespace std;


// Logical edge server
class server_t 
{
public:
    int layer;                      // Layer of the server
    int sensingArea;                // Size of sensing range
    int capacity;                   // Storage capacity
    float utilization;              // [0,1]
    vector<bool> area;              // Sensing area
    vector<float> weight;           // Attribution weights
    vector<int> lower_neighbours;   // Index of connected servers in lower layer
    vector<int> upper_neighbours;   // Index of connected servers in upper layer

    // Constructor
    server_t(int layer, int area_size, int attr_size, int sensingArea, int capacity, 
        float lb_u, float ub_u);
};

// Generate L1 servers (each range has 1 server)
vector<server_t> generate_L1_servers(int range, int area_size, int attr_size, int capacity,
    float lb_u, float ub_u);

#endif