#ifndef CLUSTER_H
#define CLUSTER_H

#include "point.h"

using namespace std; 

class Cluster_t
{
private:  
    vector<double> centroid; 
    vector<Point_t> points; 

public:
    int cluster_id;
    Cluster_t(int id, Point_t centroid);
    void addPoint(Point_t p); 
    bool removePoint(int point_id); 
    int getClusterId(); 
    int getSize(); 
    Point_t getPoint(int index); 
    double getCentroid(int index);
    void setCentroid(int index, double value); 
};

Cluster_t::Cluster_t(int id, Point_t centroid)
{
    this->cluster_id = id; 
    for (int i = 0; i < centroid.getDimensions(); i ++)
    {
        this->centroid.push_back(centroid.getValue(i)); 
    }
    this->addPoint(centroid); 
}

void Cluster_t::addPoint(Point_t p)
{
    p.setCluster(this->cluster_id); 
    this->points.push_back(p); 
}

bool Cluster_t::removePoint(int point_id)
{
    bool justify = false; 
    int size = points.size(); 
    for (int i = 0; i < size; i ++)
    {
        if (points[i].getPointId() == point_id)
        {
            points.erase(points.begin() + i); 
            justify = true; 
            break; 
        }
    }
    return justify; 
}

int Cluster_t::getClusterId()
{
    return cluster_id;
}

int Cluster_t::getSize()
{
    return points.size(); 
}

Point_t Cluster_t::getPoint(int index)
{
    return points[index]; 
}

double Cluster_t::getCentroid(int index)
{
    return centroid[index]; 
}

void Cluster_t::setCentroid(int index, double value)
{
    this->centroid[index] = value;  
}

#endif 
