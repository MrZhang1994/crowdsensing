#include <iostream>
#include <vector>
#include <ctime>
#include <queue>
#include "task.h"
#include "edge.h"
using namespace std;


/* Sensor related parameters */
#define NUM_EDGES 200                       // Number of available sensors in total
#define ATTR_SIZE 5                         // Size of attribution vector
#define AREA_SIZE 10                        // Size of area vector
#define AVG_NEIGHBOURS 5                    // Average number of neighbours of each sensor

/* Task related parameters */
#define NUM_TASKS 10                        // Number of tasks in total
#define MAX_NUM_SENSOR (NUM_EDGES/3)        // Maximum number of sensors required by a task
#define MIN_NUM_SENSOR 1                    // Minimum number of sensors required by a task
#define THRESHOLD 0.5                       // Similarity threshold required by a task


/* Comment this line if verbose output is not needed */
#define VERBOSE


// Determine required number of sensors of a task
int generate_numSensor()
{
    int tmp = lrand48() % (MAX_NUM_SENSOR - MIN_NUM_SENSOR);
    return (tmp + MIN_NUM_SENSOR);
}

// Determine whether a sensor is qualified for a task
bool check_threshold(const task_t &task, const edge_t &sensor)
{
    // Calculate similarity: "Tanimoto coefficient". Range: [0, 1]
    float a = 0;
    float b = 0;
    float c = 0;
    for (int i = 0; i < ATTR_SIZE; i++)
    {
        a = a + task.attr[i] * sensor.attr[i];
        b = b + task.attr[i] * task.attr[i];
        c = c + sensor.attr[i] * sensor.attr[i];
    }
    float rst = a / (b + c - a);
    if (rst >= THRESHOLD)
        return true;
    else
        return false;
}

// Check area requirements ???
bool check_area(task_t &task, edge_t *psensor)
{
    for (int i = 0; i < AREA_SIZE; i++)
    {
        if (task.area[i] == true && psensor->area[i] == true)
            return true;
    }
    return false;
}

// Apply gossip algorithm
void get_sensors(task_t &task, edge_t *psensor, int &cost)
{
    queue<edge_t*> Q;
    Q.push(psensor);
    while (Q.empty() == false)
    {
        if (task.sensor_num == 0)
        {
            cout << "All sensors found." << endl;
            return;
        }

        edge_t *current = Q.front();
        Q.pop();

        // check this sensor
        cost = cost + 2;
        current->checked = true;
        bool rst = check_threshold(task, *current);
        if (rst == true && check_area(task, current) == true)
        {
            task.sensor_num--;
#ifdef VERBOSE
            cout << "Sensor " << current->index << " aquired. " << task.sensor_num 
                << " sensors still needed" << endl;
#endif
        }

        // push its neighbours into the queue
        for (unsigned int i = 0; i < current->neighbours.size(); i++)
        {
            if (current->neighbours[i]->checked == true)
                continue;
            else
            {
                Q.push(current->neighbours[i]);
            }
        }
    }
}

int main()
{
    srand48(time(NULL));


    // Establish the system
    graph_t system(NUM_EDGES, AVG_NEIGHBOURS, ATTR_SIZE, AREA_SIZE);
    cout << "=================== System attributions ===================" << endl;
    cout << "Number of available sensors: " << NUM_EDGES << endl;
    cout << "Size of attribution vector: " << ATTR_SIZE << endl;
    cout << "Size of area vector: " << AREA_SIZE << endl;
    cout << "Number of tasks: " << NUM_TASKS << endl;
    cout << "===========================================================" << endl;

    // Generate tasks
    vector<task_t> tasks;
    for (int i = 0; i < NUM_TASKS; i++)
    {
        int numSensor = generate_numSensor();
        task_t new_task(ATTR_SIZE, numSensor, AREA_SIZE, THRESHOLD);
        tasks.push_back(new_task);
    }

    int cost = 0;    // I_n

    // Assume that sensors without connection with other sensors exists ???
    for (int i = 0; i < NUM_TASKS; i++)
    {
        cout << "***Task " << i << "***" << endl;
        tasks[i].print_task();
        cout << "Task " << i << " search begins." << endl;
        // update connections
        system.update_connections();

        for (int j = 0; j < NUM_EDGES; j++)
        {
            if (system.edges[j]->checked == true)
                continue;
            else
            {
                // check from this sensor
                get_sensors(tasks[i], system.edges[j], cost);
                if (tasks[i].sensor_num == 0)
                {
                    cout << "Task " << i << " search finished." << endl;
                    break;
                }
            }
        }

        cout << "Cumulative cost now is " << cost << "." << endl;

        // Not enough sensors
        if (tasks[i].sensor_num > 0)
            cout << "Not enough sensors for task " << i << "." << endl;

        cout << "===========================================================" << endl;
    }

    cout << "The overall cost is " << cost << "." << endl;
    float avg_cost = float(cost) / NUM_TASKS;
    cout << "Average cost is " << avg_cost << endl;

    return 0;
}