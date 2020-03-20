#include <iostream>
#include <vector>
#include "task.h"
#include "server.h"
#include "cluster/kmeans.h"
using namespace std;

// Parameters
#define L1_SERVER_RANGE 2               // Range covered by L1 servers
#define L1_CAPACITY 50                  // capacity of L1 srevers
#define CAPACITY(n) n*L1_CAPACITY*0.7   // capacity of Ln servers
#define lb_u 0.5                        // lower bound of server utilization
#define ub_u 1                          // upper bound of server utilization
#define AREA_SIZE 20                    // size of area vector
#define ATTR_SIZE 10                    // size of  attribution vector
#define NUM_TASK 20                     // number of tasks
#define NUM_CLASS 3                     // number of classes of tasks (for generating tasks)
#define NUM_ITERS 100                   // number of iterations for k-means clustering
#define HP1 1
#define HP2 0
#define HP3 3

// Output options
#define show_cluster_results
#define show_server_configurations  // different cluster numbers


// Cluster the tasks
vector<int> cluster_tasks(int num_clusters, vector<task_t> &taskList)
{
    vector<Point_t> points;
    for (unsigned int i = 0; i < taskList.size(); i++)
    {
        double tmp = (double)taskList[i].sensingArea;
        Point_t point(tmp, i);
        points.push_back(point);
    }

    if ((int)points.size() < num_clusters)
    {
        cerr << "Error: clustering failed with too large number of clusters" << 
            num_clusters << endl;
        exit(1); 
    }

    KMeans_t kmeans(num_clusters, NUM_ITERS);
    kmeans.Run(points);

    // set clusters for each task
    for (unsigned int i = 0; i < taskList.size(); i++)
    {
        taskList[i].clusteredClass = points[i].getClusterId();
        /*cout << "cluster: " << taskList[i].clusteredClass << " range: " << 
            taskList[i].sensingArea << " selected class: " << taskList[i].selectedClass <<  endl; */
    }

    vector<double> tmp = kmeans.getCentroids();
    vector<int> ret;
    for (unsigned int i = 0; i < tmp.size(); i++)
    {
        ret.push_back(round(tmp[i]));
        /*cout << "centroids: " << tmp[i] << " " << ret[i] << endl;*/
    }
    
    return ret;
}


// Find index of n largest numbers in a vector (all numbers are positive)
vector<int> index_of_n_largest_numbers(vector<float> v, int n)
{
    vector<int> indices;
    for (int i = 0; i < n; i++)
    {
        int index_max = 0;
        for (unsigned int j = 0; j < v.size(); j++)
        {
            if (v[j] > v[index_max])
                index_max = j;
        }
        indices.push_back(index_max);
        v[index_max] = -10;
    }
    return indices;
}


// Check if 2 servers has intersection
bool has_intersection(server_t s1, server_t s2)
{
    for (unsigned int i = 0; i < s1.area.size(); i++)
    {
        if (s1.area[i] && s2.area[i])
            return true;
    }
    return false;
}


// Print information of all servers
void print_servers(vector<vector<server_t>> servers)
{
    cout << "Printing results of established server system..." << endl;
    // Each layer
    for (unsigned int i = 0; i < servers.size(); i++)
    {
        cout << "+++++++++++++++++++++++ Layer " << i+1 << " +++++++++++++++++++++++" << endl;
        cout << "Layer " << i+1 << " has " << servers[i].size() << " servers." << endl;
        // Each server
        for (unsigned int j = 0; j < servers[i].size(); j++)
        {
            cout << "Server No. " << j << " :" << endl;
            cout << "Layer: " << servers[i][j].layer << "; Range: " << servers[i][j].sensingArea
                << "; Capacity: " << servers[i][j].capacity << "; Utilization: " << 
                servers[i][j].utilization << endl;
            cout << "Sensing area: [ ";
            for (unsigned int k = 0; k < servers[i][j].area.size(); k++)
            {
                cout << servers[i][j].area[k] << " ";
            }
            cout << "]" << endl;
            cout << "Attribution weights: [ ";
            for (unsigned int k = 0; k < servers[i][j].weight.size(); k++)
            {
                cout << servers[i][j].weight[k] << " ";
            }
            cout << "]" << endl;
            cout << "Lower neighbours: [ ";
            for (unsigned int k = 0; k < servers[i][j].lower_neighbours.size(); k++)
            {
                cout << servers[i][j].lower_neighbours[k] << " ";
            }
            cout << "]" << endl;
            cout << "Upper neighbours: [ ";
            for (unsigned int k = 0; k < servers[i][j].upper_neighbours.size(); k++)
            {
                cout << servers[i][j].upper_neighbours[k] << " ";
            }
            cout << "]" << endl;
        }
    }
}


void hsd()
{
    vector<server_t> L1_servers = generate_L1_servers(L1_SERVER_RANGE, AREA_SIZE, 
        ATTR_SIZE, L1_CAPACITY, lb_u, ub_u);
    vector<task_t> taskList = taskGenerator(AREA_SIZE, ATTR_SIZE, NUM_TASK, 
        NUM_CLASS, HP1, HP2);
    vector<vector<server_t>> servers;
    servers.push_back(L1_servers);

    cout << "Task generation completed." << endl;

    // Calculate total sensing demand for each area (sm)
    vector<float> sm;
    sm.resize(AREA_SIZE);
    sm.assign(AREA_SIZE, 0);
    for (unsigned int i = 0; i < L1_servers.size(); i++)
    {
        float sm_thisServer = (L1_servers[i].capacity*L1_servers[i].utilization) /
            (ATTR_SIZE*L1_servers[i].sensingArea);
        for (int j = 0; j < AREA_SIZE; j++)
        {
            if (L1_servers[i].area[j] == 1)
                sm[j] = sm[j]+sm_thisServer;
        }
    }

    for (int clusterNum = 2; clusterNum <= HP3; clusterNum++)
    {
        // clustering
        vector<int> clusterCenters = cluster_tasks(clusterNum, taskList);
        vector<vector<task_t>> clusters;
        clusters.resize(clusterNum);
        for (unsigned int i = 0; i < taskList.size(); i++)
        {
            int index = taskList[i].clusteredClass;
            clusters[index-1].push_back(taskList[i]);
        }

#ifdef show_cluster_results
        cout << "==================================================================" << endl;
        cout << "Clustering results for " << clusterNum << " clusters:" << endl;
        cout << "Centroids: ";
        for (unsigned int i = 0; i < clusterCenters.size(); i++)
        {
            cout << clusterCenters[i] << " ";
        }
        cout << endl;
        for (unsigned int i = 0; i < clusters.size(); i++)
        {
            cout << "Cluster " << i+1 << " sensing ranges:" << endl;
            for (unsigned int j = 0; j < clusters[i].size(); j++)
            {
                cout << clusters[i][j].sensingArea << " ";
            }
            cout << endl;
        }
        cout << "==================================================================" << endl;
#endif
        

        int num_layer = 1;
        for (int cluster_index = 0; cluster_index < clusterNum; cluster_index++)
        {
            //  Check if L1 servers can cover the cluster
            int L1_avg_range = 0;
            for (unsigned int i = 0; i < L1_servers.size(); i++)
            {
                L1_avg_range = L1_avg_range+L1_servers[i].sensingArea;
            }
            L1_avg_range = L1_avg_range/L1_servers.size();
            if (clusterCenters[cluster_index] <= L1_avg_range)
                continue;

            // Add a layer
            num_layer++;
            vector<float> weight_sum;   // Sum of weight 
            weight_sum.resize(ATTR_SIZE);
            weight_sum.assign(ATTR_SIZE, 0);
            for (unsigned int i = 0; i < clusters[cluster_index].size(); i++)
            {
                for (int j = 0; j < ATTR_SIZE; j++)
                {
                    weight_sum[j] = weight_sum[j]+clusters[cluster_index][i].weight[j];
                }
            }

            // Set new servers
            int num_new_servers = AREA_SIZE/clusterCenters[cluster_index];
            for (int each_server = 0; each_server < num_new_servers; each_server++)
            {
                server_t new_server(num_layer, AREA_SIZE, ATTR_SIZE, clusterCenters[cluster_index],
                    CAPACITY(num_layer), lb_u, ub_u);
                // Set sensing areas
                float demand = 0;
                for (int k = each_server*new_server.sensingArea; 
                    k < ((each_server+1)*new_server.sensingArea); k++)
                {
                    new_server.area[k] = 1;
                    demand = demand+sm[k];
                }
                // Set attribution weights
                int num_stored_attr = new_server.capacity/demand;
                if (num_stored_attr < 1)
                {
                    cerr << "Capacity of L" << num_layer << " servers are too small" << endl;
                    exit(2); 
                }
                else if (num_stored_attr > ATTR_SIZE)
                {
                    cerr << "WARNING (occurred when cluster number is " << clusterNum << "):" << endl;
                    cerr << "Capacity of L" << num_layer << " servers are too big" << endl;
                    cerr << "All attributions are stored in " << num_layer << " servers" << endl;
                    num_stored_attr = ATTR_SIZE;
                }
                vector<int> weight_index = index_of_n_largest_numbers(weight_sum, num_stored_attr);
                for (int k = 0; k < num_stored_attr; k++)
                {
                    new_server.weight[weight_index[k]] = 1;
                }
                new_server.utilization = (demand*num_stored_attr)/new_server.capacity;
                // Determine connections
                for (unsigned int k = 0; k < servers[num_layer-2].size(); k++)
                {
                    if (has_intersection(servers[num_layer-2][k], new_server) == true)
                    {
                        // Record indicecs of connected servers
                        if ((int)servers.size() < num_layer)
                        {
                            servers[num_layer-2][k].upper_neighbours.push_back(0);
                        }
                        else
                        {
                            servers[num_layer-2][k].upper_neighbours.push_back(
                                int(servers[num_layer-1].size()));
                        }
                        new_server.lower_neighbours.push_back(k);
                    }
                }
                // Add new server into the list
                if ((int)servers.size() < num_layer)
                {
                    servers.resize(num_layer);
                }
                servers[num_layer-1].push_back(new_server);
            }
            if (num_new_servers*clusterCenters[cluster_index] != AREA_SIZE)
            {
                // 1 extra server
                server_t new_server(num_layer, AREA_SIZE, ATTR_SIZE, clusterCenters[cluster_index],
                    CAPACITY(num_layer), lb_u, ub_u);
                // Set sensing areas
                float demand = 0;
                for (int k = 0; k < new_server.sensingArea; k++)
                {
                    new_server.area[AREA_SIZE-k] = 1; 
                    demand = demand+sm[AREA_SIZE-k];
                }
                // Set attribution weights (same as above)
                int num_stored_attr = new_server.capacity/demand;
                if (num_stored_attr < 1)
                {
                    cerr << "Capacity of L" << num_layer << " servers are too small" << endl;
                    exit(2); 
                }
                else if (num_stored_attr > ATTR_SIZE)
                {
                    cerr << "WARNING (occurred when cluster number is " << clusterNum << "):" << endl;
                    cerr << "Capacity of L" << num_layer << " servers are too big" << endl;
                    cerr << "All attributions are stored in " << num_layer << " servers" << endl;
                    num_stored_attr = ATTR_SIZE;
                }
                vector<int> weight_index = index_of_n_largest_numbers(weight_sum, num_stored_attr);
                for (int k = 0; k < num_stored_attr; k++)
                {
                    new_server.weight[weight_index[k]] = 1;
                }
                new_server.utilization = (demand*num_stored_attr)/new_server.capacity;
                // Determine connections
                for (unsigned int k = 0; k < servers[num_layer-2].size(); k++)
                {
                    if (has_intersection(servers[num_layer-2][k], new_server) == true)
                    {
                        // Record indicecs of connected servers
                        if ((int)servers.size() < num_layer)
                        {
                            servers[num_layer-2][k].upper_neighbours.push_back(0);
                        }
                        else
                        {
                            servers[num_layer-2][k].upper_neighbours.push_back(
                                int(servers[num_layer-1].size()));
                        }
                        new_server.lower_neighbours.push_back(k);
                    }
                }
                // Add new server into the list
                if ((int)servers.size() < num_layer)
                {
                    servers.resize(num_layer);
                }
                servers[num_layer-1].push_back(new_server);

            }
        }

#ifdef show_server_configurations
    cout << "==================================================================" << endl;
    cout << "Following is the server configuration for " << clusterNum << " clusters" << endl;
    print_servers(servers);
#endif

    }

    // Evaluator
    

}




int main()
{
    hsd();

    return 0;
}