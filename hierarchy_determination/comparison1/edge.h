#ifndef _EDGE_H_
#define _EDGE_H_

#include <vector>
using namespace std;

// Sensors.
class edge_t
{
public:
    vector<float> attr;             // attributions of sensor (a_k)
    vector<bool> area;              // area covered (m_k)
    vector<edge_t*> neighbours;     // connections
    bool checked;                   // true as checked
    int index;                      // index of this sensor

    // Edge initializtion.
    edge_t(int attr_size, int area_size);

    // Show all information of this edge. (debug)
    void print_edge();
};

class graph_t
{
    int avg_neighbours;
public:
    vector<edge_t*> edges;

    // Initialize the graph
    graph_t(int num_edges, int avg_neighbours, int attr_size, int area_size);

    // Update edge connections
    void update_connections();
    
    ~graph_t();
};

#endif