#ifndef KMEANS_H
#define KMEANS_H

#include "cluster.h"

using namespace std; 

// mute the output
#define mute

class KMeans_t
{
private: 
    int K;
    int iters;
    int dimensions;
    int total_points; 
    vector<Cluster_t> clusters; 

    int getNearestClusterID(Point_t p); 

    // rearrange cluster order (ascending) (1 dimensional)
    void rearrange_order(vector<Point_t>& points_set);

public:
    KMeans_t(int K_in, int iterations); 
    void Run(vector<Point_t>& points_set); 

    // get centroids for 1 dimensional points
    vector<double> getCentroids();
};

int KMeans_t::getNearestClusterID(Point_t p)
{
    double sum = 0.0; 
    double min_dist; 
    int nearest_cluster_id; 

    for (int i = 0; i < dimensions; i ++)
    {
        sum += pow(clusters[0].getCentroid(i) - p.getValue(i), 2.0); 
    }
    min_dist = sqrt(sum); 
    nearest_cluster_id = clusters[0].getClusterId(); 

    for (int i = 1; i < K; i ++)
    {
        double dist; 
        sum = 0.0; 

        for (int j = 0; j < dimensions; j ++)
        {
            sum += pow(clusters[i].getCentroid(j) - p.getValue(j), 2.0); 
        }

        dist = sqrt(sum); 

        if (dist < min_dist)
        {
            min_dist = dist;
            nearest_cluster_id = clusters[i].getClusterId();
        }
    }

    return nearest_cluster_id; 
}

KMeans_t::KMeans_t(int K_in, int iterations)
{
    this->K = K_in; 
    this->iters = iterations; 
}

void KMeans_t::Run(vector<Point_t>& points_set)
{
    total_points = points_set.size(); 
    dimensions = points_set[0].getDimensions(); 

    // initialize clusters 
    vector<int> used_point_ids; 

    for (int i = 1; i <= K; i ++)
    {
        while (true)
        {
            int index = rand() % total_points; 
            if (find(used_point_ids.begin(), used_point_ids.end(), index) == used_point_ids.end())
            {
                used_point_ids.push_back(index);
                points_set[index].setCluster(i); 
                Cluster_t current_cluster(i, points_set[index]); 
                clusters.push_back(current_cluster); 
                break;
            }
        }
    }

#ifndef mute
    cout << "Clusters initialized = " << clusters.size() << endl; 
    cout << endl; 

    // run K-Means clustering process 
    cout << "Running K-Means .." << endl; 
#endif

    int current_iter = 1; 
    while (true)
    {
#ifndef mute
        cout << "Iter - " << current_iter << "/" << iters << endl; 
#endif
        bool done = true; 

        // add all points to their nearest cluster
        for (int i = 0; i < total_points; i ++)
        {
            int current_cluster_id = points_set[i].getClusterId(); 
            int nearest_cluster_id = getNearestClusterID(points_set[i]); 

            if (current_cluster_id != nearest_cluster_id)
            {
                if (current_cluster_id != 0)
                {
                    for (int j = 0; j < K; j ++)
                    {
                        if (clusters[j].getClusterId() == current_cluster_id)
                        {
                            clusters[j].removePoint(points_set[i].getPointId()); 
                        }
                    }
                }

                for (int j = 0; j < K; j ++)
                {
                    if (clusters[j].getClusterId() == nearest_cluster_id)
                    {
                        clusters[j].addPoint(points_set[i]);
                    }
                }
                points_set[i].setCluster(nearest_cluster_id);
                done = false; 
            }
        }

        // recalculate centroid for each cluster 
        for (int i = 0; i < K; i ++)
        {
            int ClusterSize = clusters[i].getSize(); 

            for (int j = 0; j < dimensions; j ++)
            {
                double sum = 0.0;
                if (ClusterSize > 0)
                {
                    for (int m = 0; m < ClusterSize; m ++)
                    {
                        sum += clusters[i].getPoint(m).getValue(j); 
                    }
                    clusters[i].setCentroid(j, sum / ClusterSize); 
                }
            }
        }

        if (done || current_iter >= iters)
        {
#ifndef mute
            cout << "Clustering process completed in iteration: " << current_iter << endl; 
            cout << endl; 
#endif
            break; 
        } 
        current_iter ++; 
    }

    // rearrange order
    rearrange_order(points_set);

#ifndef mute
    // print point_ids in each cluster
    for (int i = 0; i < K; i ++)
    {
        cout << "Points in cluster " << clusters[i].getClusterId() << " : "; 
        for (int j = 0; j < clusters[i].getSize(); j ++)
        {
            cout << clusters[i].getPoint(j).getPointId() << " ";
        }
        cout << endl; 
        cout << endl; 
    }
    cout << "========================" << endl;
    cout << endl;
#endif

#ifndef mute
    // write cluster centers to file
    ofstream outfile; 
    outfile.open("clusters_result");
    if (!outfile.is_open())
    {
        cout << "Error: cannot write to file clusters_result" << endl; 
    }
    else 
    {
        for (int i = 0; i < K; i ++)
        {
            cout << "Cluster " << clusters[i].getClusterId() << " centroid: "; 
            for (int j = 0; j < dimensions; j ++)
            {
                cout << clusters[i].getCentroid(j) << " ";
                outfile << clusters[i].getCentroid(j) << " ";
            }
            cout << endl; 
            outfile << endl;
        }
        outfile.close(); 
    }
#endif
}

vector<double> KMeans_t::getCentroids()
{
    vector<double> ret;
    for (unsigned int i = 0; i < clusters.size(); i++)
    {
        ret.push_back(clusters[i].getCentroid(0));
    }
    return ret;
}


void KMeans_t::rearrange_order(vector<Point_t>& points_set)
{
    vector<double> old_order;
    vector<double> new_order;
    vector<int> new_index;

    for (unsigned int i = 0; i < clusters.size(); i++)
    {
        old_order.push_back(clusters[i].getCentroid(0));
    }

    new_order = old_order;
    sort(new_order.begin(), new_order.end());   //ascending

    for (unsigned int i = 0; i < clusters.size(); i++)
    {
        for (unsigned int j = 0; j < clusters.size(); j++)
        {
            if (old_order[i] == new_order[j])
            {
                new_index.push_back(j);
                break;
            }
        }
    }

/*
    cout << "old order: " << endl;
    for (unsigned int i = 0; i < old_order.size(); i++)
    {
        cout << old_order[i] << " ";
    }
    cout << endl;

    cout << "new order: " << endl;
    for (unsigned int i = 0; i < old_order.size(); i++)
    {
        cout << new_order[i] << " ";
    }
    cout << endl;
*/

    //swap
    vector<Cluster_t> tmp = clusters;
    for (unsigned int i = 0; i < clusters.size(); i++)
    {
        tmp[new_index[i]] = clusters[i];
        tmp[new_index[i]].cluster_id = new_index[i]+1;
    }
    clusters = tmp;
/*
    cout << "new index: " << endl;
    for (unsigned int i = 0; i < new_index.size(); i++)
    {
        cout << new_index[i] << " ";
    }
    cout << endl;

    cout << "cluster: " << endl;
    for (unsigned int i = 0; i < new_index.size(); i++)
    {
        cout << clusters[i].getCentroid(0) << " ";
    }
    cout << endl;
*/
    for (unsigned int i = 0; i < points_set.size(); i++)
    {
        int old_cluster = points_set[i].getClusterId();
        points_set[i].setCluster(new_index[old_cluster-1]+1);
        /* cout << "point value: " << points_set[i].getValue(0) << " cluster:"
            << points_set[i].getClusterId() << endl; */
    }
}


#endif
