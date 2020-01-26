#include <fstream>  
#include <iostream> 
#include <random>
#include <ctime>
#include "task.h"
#define ATTR_MAX 10
#define DIM_SIZE_MAX 10 

using namespace std; 

void GenerateData(vector<vector<float> > &data, int each_cluster, int cluster_num, float std_div)
{
    data.resize(cluster_num); 
    for (int i = 0; i < cluster_num; i ++)
    {
        data[i].resize(each_cluster); 
    } // initialize our data space

    vector<float> centers;  
    centers.resize(cluster_num); 
    for (int i = 0; i < cluster_num; i ++)
    {
        centers[i] = float(rand() / double(RAND_MAX)) * ATTR_MAX * 0.9 + 0.1 * ATTR_MAX; 
    } // generate the cluster center for the "count" layer group

    default_random_engine generator;
    for (int i = 0; i < cluster_num; i ++)
    {
        normal_distribution<float> distribution(centers[i], std_div);
        for (int j = 0; j < each_cluster; j ++)
        {
            data[i][j] = distribution(generator); 
        }
    }
}

void FitIn(Grid *map, vector<vector<float> > &data, int rank, int each_cluster, int cluster_num)
{
    vector<int> cluster_cnt; 
    cluster_cnt.resize(cluster_num); 
    for (int i = 0; i < cluster_num; i ++)
    {
        cluster_cnt[i] = each_cluster; 
    } // generate a vector of counter recording the left numbers for each cluster

    cluster_cnt.push_back((map->GetWidth() * map->GetHeight()) - (each_cluster * cluster_num)); 
    // append a default counter to record the zeros

    for (int i = 0; i < map->GetWidth(); i ++)
    {
        for (int j = 0; j < map->GetHeight(); j ++)
        {
            int current_cluster = rand() % (cluster_num + 1) + 1;
            while (cluster_cnt[current_cluster - 1] <= 0)
            {
                current_cluster = rand() % (cluster_num + 1) + 1;
            } 
            cluster_cnt[current_cluster - 1] --; 
            map->city[i][j]->assigned = true; 
            if (current_cluster != cluster_num + 1)
            {
                (map->city[i][j]->attr)[rank - 1] = data[current_cluster - 1][cluster_cnt[current_cluster - 1]]; 
            }
        }
    }

}

int main()
{
    srand((unsigned int)(time(nullptr)));
    int grid_w = 10; 
    int grid_h = 10; 
    int grid_max_size = DIM_SIZE_MAX; 
    int cluster_number = 4; 
    int each_cluster = grid_w * grid_h / cluster_number; 
    float standard_div = 0.5; 

    Grid* cmap = new Grid(grid_w, grid_h, grid_max_size); 
    // we now have the space for gird_w * grid_h * grid_max_size 
    vector<vector<float> > cluster_data; 
    
    for (int i = 0; i < DIM_SIZE_MAX; i ++)
    {
        GenerateData(cluster_data, each_cluster, cluster_number, standard_div); 
        FitIn(cmap, cluster_data, i + 1, each_cluster, cluster_number); 
    }
    
    ofstream outFile("Map.log");  
    if (outFile.fail())
    {
        cout << "error when open output file" << endl; 
    }

    for (int k = 0; k < DIM_SIZE_MAX; k ++)
    {
        for (int i = 0; i < cmap->GetWidth(); i ++)
        {
            for (int j = 0; j < cmap->GetHeight(); j ++)
            {
                outFile << (cmap->city[i][j]->attr)[k] << " "; 
            }
        }
        outFile << endl; 
    }

    outFile.close();  

    delete cmap; 
    return 0; 
}
