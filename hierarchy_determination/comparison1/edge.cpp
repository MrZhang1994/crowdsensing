#include "edge.h"
#include <ctime>
#include <iostream>
#include <vector>
using namespace std;

edge_t::edge_t(int attr_size, int area_size)
{
    checked = false;
    attr.resize(attr_size);
    area.resize(area_size);

    // set attribution values
    //srand48(time(NULL));
    for (int i = 0; i < attr_size; i++)
    {
        attr[i] = drand48();
    }

    // set sensing areas
    int starting_point = lrand48() % area_size;
    int length = lrand48() % (area_size - starting_point) + 1;
    for (int i = 0; i < area_size; i++)
    {
        area[i] = false;
    }
    for (int i = 0; i < length; i++)
    {
        area[starting_point+i] = true;
    }
}

void edge_t::print_edge()
{
    cout << "Attributions:" << endl;
    for (unsigned int i = 0; i < attr.size(); i++)
        cout << attr[i] << " ";
    cout << endl;
    cout << "Sensing areas:" << endl;
    for (unsigned int i = 0; i < area.size(); i++)
        cout << (int)area[i] << " ";
    cout << endl;
    cout << "Number of neighbours:" << endl;
    cout << (int)neighbours.size() << endl;
}

graph_t::graph_t(int num_edges, int avg_neighbours, int attr_size, int area_size)
{
    this->avg_neighbours = avg_neighbours;

    for (int i = 0; i < num_edges; i++)
    {
        edge_t *new_edge = new edge_t(attr_size, area_size);
        new_edge->index = i;
        edges.push_back(new_edge);
    }
}

void graph_t::update_connections()
{
    //srand48(time(NULL));
    int num_edges = edges.size();

    // Clear current neighbours
    for (int i = 0; i <num_edges; i++)
    {
        edges[i]->neighbours.resize(0);
        edges[i]->checked = false;
    }

    //int relations = num_edges * num_edges;
    int total = avg_neighbours * num_edges / 2; // number of loops

    // resize and initialize mapping
    vector<vector<bool>> mapping;
    mapping.resize(num_edges);
    for (int i = 0; i < num_edges; i++)
    {
        mapping[i].resize(num_edges);
        for (int j = 0; j < num_edges; j++)
        {
            mapping[i][j] = false;
        }
    }

    // Create connections
    for (int i = 0; i < total; i++)
    {
        int target1 = lrand48() % num_edges;
        int target2 = lrand48() % num_edges;

        if (mapping[target1][target2] == true || target1 == target2)
        {
            // If repeated, neglect
            continue;
        }
        else
        {
            // Build connections
            this->edges[target1]->neighbours.push_back(this->edges[target2]);
            this->edges[target2]->neighbours.push_back(this->edges[target1]);
            mapping[target1][target2] = true;
            mapping[target2][target1] = true;
        }      

    }

}

graph_t::~graph_t()
{
    int num = edges.size();
    for (int i = 0; i < num; i++)
    {
        delete edges[i];
    }
}