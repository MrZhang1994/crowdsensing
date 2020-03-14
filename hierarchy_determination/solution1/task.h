#ifndef _TASK_H_
#define _TASK_H

#include <vector>
#include <string>
#include <random>
using namespace std;


class task_t 
{
public:
    int selectedClass;         // Pre-selected class (layer) in task generator
    int clusteredClass;        // Class (layer) decided by clustering algorithm
    int sensingArea;           // Size of required sensing area ("# 1s")
    vector<bool> area;         // Required sensing areas (m_n)
    vector<float> weight;      // weight of each attribution (w_n)
    vector<bool> attr;         // Attributions

    // Generate a task
    task_t(int area_size, int attr_size, int num_layer, float HP1, float HP2);

    // Log the task
    string logTask(string seperator);
};

// Task generator
vector<task_t> taskGenerator(int area_size, int attr_size, int num_task, int num_layer, 
    int HP1, int HP2);

#endif