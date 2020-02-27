#include "kmeans.h"

using namespace std;

int main(int argc, char **argv)
{
    // Need 2 arguments (except filename) to run, else exit
    if(argc != 3)
    {
        cout<<"Error: wrong argument." << endl;
        return 1;
    }

    // Fetching number of clusters
    int K = atoi(argv[2]); 

    // Open file for fetching points
    string filename = argv[1];
    ifstream infile(filename.c_str()); 

    if (!infile.is_open())
    {
        cout << "Error: fail to open input file." << endl;
        return 1; 
    }

    int pointId = 1;
    vector<Point_t> points_set; 
    string line; 

    while (getline(infile, line))
    {
        Point_t current_point(line, pointId);
        points_set.push_back(current_point); 
        pointId ++; 
    }
    infile.close();
    cout << "Fetching data complete." << endl; 

    // Exception when cluster number > point number
    if (points_set.size() < K)
    {
        cout << "Error: big K." << endl;
        return 1; 
    }

    // Running process
    int iters = 100; 

    KMeans_t kmeans(K, iters);
    kmeans.Run(points_set); 

    return 0; 
}
