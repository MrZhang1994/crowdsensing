#ifndef _TASK_H_
#define _TASK_H_

#include <vector>
#include <string>
#include "server.h"
using namespace std;


class task_t 
{
public:
    int selectedClass;         // Pre-selected class (layer) in task generator
    int clusteredClass;        // Class (layer) decided by clustering algorithm (-1: undefined)
    int sensingArea;           // Size of required sensing area ("# 1s")
    vector<bool> area;         // Required sensing areas (m_n)
    vector<float> weight;      // Weight of each attribution (w_n)
    vector<bool> attr;         // Attributions
    vector<server_t> server;   // Valid servers for this task
    pair<int, int> target;     // Index of server found in dispatcher: servers[first][second]
                               // (-1, -1): undefined
    int delay;                 // Transmission delay
    int transCost;             // Transmission cost

    // Generate a task
    task_t(int area_size, int attr_size, int num_layer, float HP1, float HP2);

    // Log the task
    string logTask(string seperator);
};

// Task generator
vector<task_t> taskGenerator(int area_size, int attr_size, int num_task, int num_layer, 
    int HP1, int HP2);

#endif