#ifndef _TASK_H_
#define _TASK_H

#include <vector>
using namespace std;

class task_t 
{
public:
    vector<float> attr;             // attribution of sensors (a_n)
    vector<float> attr_weight;      // weight of each attribution (w_n)
    int sensor_num;                 // required number of sensors (s_n')
    vector<bool> area;              // required sensing areas (m_n)
    float threshold;                // similarity threshold (b_n)

    // Initialize a task randomly.
    task_t(int attr_size, int sensor_num, int area_size, float threshold);

    // Show all information of this task. (debug)
    void print_task();
};



#endif