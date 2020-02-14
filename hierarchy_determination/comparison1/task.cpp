#include "task.h"
#include <ctime>
#include <iostream>
using namespace std;

task_t::task_t(int attr_size, int sensor_num, int area_size, float threshold)
{
    this->attr.resize(attr_size);
    this->attr_weight.resize(attr_size);
    this->sensor_num = sensor_num;
    this->area.resize(area_size);
    this->threshold = threshold;

    // set attribution values
    //srand48(time(NULL));
    for (int i = 0; i < attr_size; i++)
    {
        this->attr[i] = drand48();
    }

    // set attribution weights
    float tmp = 1;
    for (int i = 0; i < attr_size-1; i++)
    {
        this->attr_weight[i] = tmp - tmp*drand48();
        tmp = tmp - attr_weight[i];
    }
    this->attr_weight[attr_size-1] = tmp;

    // set required sensing areas
    int starting_point = lrand48() % area_size;
    int length = lrand48() % (area_size - starting_point) + 1;
    for (int i = 0; i < area_size; i++)
    {
        this->area[i] = false;
    }
    for (int i = 0; i < length; i++)
    {
        this->area[starting_point+i] = true;
    }
}

void task_t::print_task()
{
    cout << "attributions:" << endl;
    for (unsigned int i = 0; i < attr.size(); i++)
        cout << attr[i] << " ";
    cout << endl;
    cout << "weights:" << endl;
    for (unsigned int i = 0; i < attr.size(); i++)
        cout << attr_weight[i] << " ";
    cout << endl;
    cout << "Required number of sensors: " << endl;
    cout << sensor_num << endl;
    cout << "Required sensing areas:" << endl;
    for (unsigned int i = 0; i < area.size(); i++)
        cout << (int)area[i] << " ";
    cout << endl;
    cout << "Threshold: " << endl;
    cout << threshold << endl;

}