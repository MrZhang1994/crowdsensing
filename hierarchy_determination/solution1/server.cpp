#include "server.h"
#include <iostream>
#include <ctime>
using namespace std;


server_t::server_t(int layer, int area_size, int attr_size, int sensingArea, int capacity,
    float lb_u, float ub_u)
{
    this->layer = layer;
    this->sensingArea = sensingArea;
    this->capacity = capacity;
    this->weight.resize(attr_size);
    this->area.resize(area_size);
    area.assign(area_size, 0);
    weight.assign(attr_size, 0);

    if (layer == 1)
    {
        srand48(time(NULL));
        this->utilization = drand48()*(ub_u-lb_u)+lb_u;
    }
    else
    {
        utilization = 1;
    }
}

vector<server_t> generate_L1_servers(int range, int area_size, int attr_size, int capacity,
    float lb_u, float ub_u)
{
    vector<server_t> ret;
    int num = area_size/range;
    for (int i = 0; i < num; i++)
    {
        server_t server(1, area_size, attr_size, range, capacity, lb_u, ub_u);
        for (int j = 0; j < range; j++)
        {
            server.area[i*range+j] = 1;
        }
        server.weight.assign(attr_size, 1);
        ret.push_back(server);
    }
    if (num*range != area_size)
    {
        server_t server(1, area_size, attr_size, range, capacity, lb_u, ub_u);
        for (int j = 0; j < range; j++)
        {
            server.area[area_size-1-j] = 1;
        }
        server.weight.assign(attr_size, 1);
        ret.push_back(server);
    }
    return ret;
}
