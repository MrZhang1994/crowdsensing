#ifndef POINT_H
#define POINT_H

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace std;

class Point_t 
{
private:
    int point_id;
    int cluster_id;
    int dimensions; 
    vector<double> values; 

public:
    Point_t(string line, int id);
    int getPointId();
    int getClusterId();
    int getDimensions();
    double getValue(int index);

    void setCluster(int c_id); 
};

Point_t::Point_t(string line, int id)
{
    point_id = id; 
    cluster_id = 0; // initially not assigned to any cluster
    dimensions = 0; 
    stringstream input(line); 
    values.clear(); 
    values.resize(0); 
    double tmp; 
    while (input >> tmp)
    {
        values.push_back(tmp);
        dimensions ++; 
    }
}

int Point_t::getPointId()
{
    return point_id; 
}

int Point_t::getClusterId()
{
    return cluster_id; 
}

int Point_t::getDimensions()
{
    return dimensions; 
}

double Point_t::getValue(int index)
{
    return values[index];  
}

void Point_t::setCluster(int c_id)
{
    cluster_id = c_id;  
}

#endif 
