#include "task.h"
#include <iostream>
#include <fstream>
#include <ctime>
#include <chrono>
using namespace std;


task_t::task_t(int area_size, int attr_size, int num_layer, float HP1, float HP2)
{
    // Initialization
    area.resize(area_size);
    weight.resize(attr_size);
    attr.resize(attr_size);
    area.assign(area_size, 0);
    weight.assign(attr_size, 0);
    attr.assign(attr_size, 0);

    selectedClass = lrand48()%num_layer+1;
    clusteredClass = -1;    // undefined

    // Set sensing areas
    default_random_engine generator;
    generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
    //generator.seed(time(NULL));
    normal_distribution<float> normal(float(selectedClass*area_size)/float(num_layer+1), HP1);
    sensingArea = -1;
    while(sensingArea <= 0 || sensingArea > area_size)
    {
        sensingArea = round(normal(generator));
    }
    int sensingStart;
    if (area_size != sensingArea)
        sensingStart = lrand48()%(area_size-sensingArea);
    else
        sensingStart = 0;   // Avoid floating point exception
    for (int i = 0; i < sensingArea; i++)
    {
        area[sensingStart+i] = 1;
    }

    // Set attributions and weights
    int attrStart = (selectedClass-1)*attr_size/(num_layer+1);
    if (lrand48()%10 > HP2 || selectedClass == num_layer)    
    {
        float tmp = 1;
        for (int i = 0; i < attr_size/(num_layer+1); i++)
        {
            attr[attrStart+i] = lrand48()%2;    //odd(lrand48());
            if (i == (attr_size/(num_layer+1)-1))
            {
                weight[attrStart+i] = tmp;
            }
            else
            {
                weight[attrStart+i] = tmp-tmp*drand48();
                tmp = tmp-weight[attrStart+i];
            }
        }
    }
    else
    {
        float tmp = 1;
        for (int i = 0; i < 2*attr_size/(num_layer+1); i++)
        {
            attr[attrStart+i] = lrand48()%2;
            if (i == (2*attr_size/(num_layer+1)-1))
            {
                weight[attrStart+i] = tmp;
            }
            else
            {
                weight[attrStart+i] = tmp-tmp*drand48();
                tmp = tmp-weight[attrStart+i];
            }
        }
    }
}


string task_t::logTask(string seperator)
{
    string out;
    out = to_string(selectedClass) + seperator + to_string(sensingArea) + seperator + "[";
    for (unsigned int i = 0; i < area.size(); i++)
    {
        if (i < area.size()-1)
            out = out + to_string(area[i]) + " ";
        else
            out = out + to_string(area[i]) + "]";
    }
    out = out + seperator + "[";
    for (unsigned int i = 0; i < weight.size(); i++)
    {
        if (i < weight.size()-1)
            out = out + to_string(weight[i]) + " ";
        else
            out = out + to_string(weight[i]) + "]";
    }
    out = out + seperator + "[";
    for (unsigned int i = 0; i < attr.size(); i++)
    {
        if (i < attr.size()-1)
            out = out + to_string(attr[i]) + " ";
        else
            out = out + to_string(attr[i]) + "]";
    }
    return out;
}


vector<task_t> taskGenerator(int area_size, int attr_size, int num_task, int num_layer, 
    int HP1, int HP2)
{
    // Generate tasks
    vector<task_t> taskList;
    for (int j = 0; j < num_task; j++)
    {
        task_t task(area_size, attr_size, num_layer, HP1, HP2);
        taskList.push_back(task);
    }

    // Log tasks
    string seperator = ",";     // Change this parameter to change the seperator
    ofstream outFile("task_list.csv");  
    if (outFile.fail())
    {
        cout << "Cannot open the file 'task_list.csv'" << endl; 
    }
    outFile << "No." << seperator << "selectedClass" << seperator << "range" << seperator
        << "areas" << seperator << "weight" << seperator << "attributions" << endl;
    for (int j = 0; j < num_task; j++)
    {
        string line = taskList[j].logTask(seperator);
        outFile << j+1 << seperator << line << endl;
    }

    return taskList;
}