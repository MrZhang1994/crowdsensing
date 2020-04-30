#include "eval.h"
#include <iostream>

using namespace std;

// Whether dot product > 0
static bool has_intersection(vector<bool> v1, vector<bool> v2)
{
    if (v1.size() != v2.size())
    {
        cerr << "2 vectors with different dimensions." << endl;
        exit(3);
    }
    for (size_t i = 0; i < v1.size(); i++)
    {
        if (v1[i] == true && v2[i] == true)
            return true;
    }
    return false;
}

// Append attributions (server & task)
static void append_attr(server_t &server, const task_t &task)
{
    server.attr.push_back(task.attr);
}

// Append attributions (server & server)
static void append_attr(server_t &target, server_t &source)
{
    for (size_t i = 0; i < source.attr.size(); i++)
    {
        target.attr.push_back(source.attr[i]);
    }
}

// Append attributions (random)
static void append_attr(server_t &target)
{
    vector<bool> tmp;
    tmp.resize(target.weight.size());
    for (size_t i = 0; i < tmp.size(); i++)
    {
        tmp[i] = lrand48()%2;
    }
    target.attr.push_back(tmp);
}

// Returns number of ones in the vector
static int num_ones(vector<bool> v)
{
    int ret = 0;
    for (size_t i = 0; i < v.size(); i++)
    {
        if (v[i] == true)
            ret++;
    }
    return ret;
}

// Returns number of ones in the vector
static int num_ones(vector<float> v)
{
    int ret = 0;
    for (size_t i = 0; i < v.size(); i++)
    {
        if (v[i] == 1)
            ret++;
    }
    return ret;
}

// Returns true if 2 servers are connected
static bool is_connected(const server_t &upper_server, int lower_server_index)
{
    for (size_t i = 0; i < upper_server.lower_neighbours.size(); i++)
    {
        if (upper_server.lower_neighbours[i].first == lower_server_index)
            return true;
    }
    return false;
}

// Returns index of a neighbouring server in "lower_neighbours"
static int index_lower_neighbour(const server_t &upper_server, int lower_server_index)
{
    for (size_t i = 0; i < upper_server.lower_neighbours.size(); i++)
    {
        if (upper_server.lower_neighbours[i].first == lower_server_index)
            return i;
    }
    return -1;
}

// Returns index of a neighbouring server in "upper_neighbours"
static int index_upper_neighbour(const server_t &lower_server, int upper_server_index)
{
    for (size_t i = 0; i < lower_server.upper_neighbours.size(); i++)
    {
        if (lower_server.upper_neighbours[i].first == upper_server_index)
            return i;
    }
    return -1;
}


// System initialization
int systemInit(float HP4, vector<vector<server_t>> &servers, vector<task_t> &taskList)
{
    //cout << "SystemInit" << endl;
    // Assign attributions to the 1st layer
    for (size_t eachTask = 0; eachTask < taskList.size(); eachTask++)
    {
        for (size_t eachServer = 0; eachServer < servers[0].size(); eachServer++)
        {
            if (has_intersection(taskList[eachTask].area, servers[0][eachServer].area) == true)
                taskList[eachTask].server.push_back(servers[0][eachServer]);
            if (taskList[eachTask].server.size() > 0)
            { 
                server_t &target = taskList[eachTask].server[lrand48()%(taskList[eachTask].server.size())];
                append_attr(target, taskList[eachTask]);
            }
        }
    }

    // Generate remaining attribution of each server
    for (size_t eachServer = 0; eachServer < servers[0].size(); eachServer++)
    {
        while (   servers[0][eachServer].attr.size() <= 
            ((float)taskList.size()*HP4/(float)servers[0].size())   )
        {
            append_attr(servers[0][eachServer]);   
        }
    }

    // Transmission cost
    int attrTrans = 0;
    for (size_t layer = 1; layer < servers.size(); layer++)
    {
        for (size_t eachServer = 0; eachServer < servers[layer].size(); eachServer++)
        {
            for (size_t lowerServer = 0; lowerServer < servers[layer-1].size(); lowerServer++)
            {
                if (is_connected(servers[layer][eachServer], lowerServer) == true)
                {
                    append_attr(servers[layer][eachServer], servers[layer-1][lowerServer]);

                    int index1 = index_lower_neighbour(servers[layer][eachServer], lowerServer);
                    int index2 = index_upper_neighbour(servers[layer-1][lowerServer], eachServer);
                    int tmp = lrand48()%10;     // latency
                    servers[layer][eachServer].lower_neighbours[index1].second = tmp;
                    servers[layer-1][lowerServer].upper_neighbours[index2].second = tmp;
                    
                    attrTrans = attrTrans + servers[layer][eachServer].attr.size() *
                        num_ones(servers[layer][eachServer].weight);
                }
            }
        }
    }

    return attrTrans;
}


// Evaluator
result_t evaluator(vector<vector<server_t>> &servers, vector<task_t> &taskList, float HP4, float HP5)
{
    //cout << "Evaluator" << endl;

    int attrTrans = systemInit(HP4, servers, taskList);
    result_t ret;
    ret.attrTrans = attrTrans;
    ret.total_delay = 0;
    ret.total_transCost = 0;
    ret.num_failure = 0;

    for (size_t eachTask = 0; eachTask < taskList.size(); eachTask++)
    {
        // dispatcher
        dispatcher(taskList[eachTask], servers);

        // dispatcher failed
        if (taskList[eachTask].target.first == -1)
        {
            ret.num_failure++;
            cout << "Task " << eachTask+1 << " is not matched because of dispatching failure." << endl;
            continue;
        }

        server_t &target = servers[taskList[eachTask].target.first][taskList[eachTask].target.second];

        int matchFlag = matcher(taskList[eachTask], target, HP5);
        if (matchFlag > 0)
        {
            cout << "Task " << eachTask+1 << " is matched. Target layer: layer " <<  
                    target.layer << "." << endl;
        }
        else if (!matchFlag && target.layer == 1)
        {
            cerr << "Matching of task " << eachTask+1 << " failed. (layer 1)" << endl;
            exit(4);
        }
        else  //(!matchFlag)
        {
            for (size_t lowerServer = 0; lowerServer < servers[target.layer-2].size(); lowerServer++)
            {
                if (is_connected(target, lowerServer) == true)
                {
                    server_t &lowerTarget = servers[target.layer-2][lowerServer];
                    int matchTemp = matcher(taskList[eachTask], lowerTarget, HP5);
                    matchFlag = matchFlag + matchTemp;
                    taskList[eachTask].transCost++;
                    int ln_index = index_lower_neighbour(target, lowerServer);
                    int latency = target.lower_neighbours[ln_index].second;
                    if (latency > taskList[eachTask].delay)
                        taskList[eachTask].delay = latency;
                }
            }
            if (matchFlag > 0)
            {
                cout << "Task " << eachTask+1 << " is matched in lower layer: layer " <<  
                    target.layer-1 << "." << endl;
            }
            else
            {
                cout << "Task " << eachTask+1 << " is not matched even in lower layer. (Low similarity)" 
                    << endl;
            }
            
        }
        
        ret.total_delay = ret.total_delay + taskList[eachTask].delay;
        ret.total_transCost = ret.total_transCost +taskList[eachTask].transCost;
    }

    return ret;
}


// dispatcher
void dispatcher(task_t &task, vector<vector<server_t>> &servers)
{
    //cout << "Dispatcher" << endl;

    for (size_t layer = 0; layer < servers.size(); layer++)
    {
        for (size_t eachServer = 0; eachServer < servers[layer].size(); eachServer++)
        {
            // Check if the range of the servers covers the task
            vector<bool> tmp1;
            tmp1.resize(task.area.size());
            for (size_t i = 0; i < tmp1.size(); i++)
            {
                tmp1[i] = servers[layer][eachServer].area[i] && (!task.area[i]);
            }
            if (num_ones(tmp1) == (num_ones(servers[layer][eachServer].area) - num_ones(task.area)))
            {
                task.target = make_pair(layer, eachServer);
                return;
            }
        }
    }

    // target not found
    //cerr << "Target not found. Dispatching failure." << endl;
    //exit(5);
}


// matcher 
bool matcher(const task_t &task, const server_t &server, float HP5)
{
    bool matchFlag = 0;
    for (size_t eachAttr = 1; eachAttr < server.attr.size(); eachAttr++)
    {
        float sim = 0;
        for (size_t eachBit = 0; eachBit < task.weight.size(); eachBit++)
        {
            if (server.weight[eachBit] == 1 && task.attr[eachBit] == server.attr[eachAttr][eachBit])
                sim = sim + task.weight[eachBit];
        }
        if (sim >= HP5)
        {
            matchFlag = 1;
            break;
        }
    }
    return matchFlag;
}
