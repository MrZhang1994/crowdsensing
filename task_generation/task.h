#ifndef TASK_H
#define TASK_H

#include <iostream>
#include <cstdlib>
#include <vector>
#include <utility> 

using namespace std; 

class Task
{
private: 
    int size; 
    pair<int, int> order; 
public: 
    bool assigned; 
    vector<float> attr; 

    Task(int size_in, int layer, int index);
    int GetSize();
    pair<int, int> GetOrder(); 
};

Task::Task(int size_in, int layer, int index)
{
    size = size_in; 
    order = make_pair(layer, index); 
    attr.resize(size); 
    assigned = false; 
    for (int i = 0; i < size; i ++)
    {
        attr[i] = 0; 
    }
}

int Task::GetSize()
{
    return size; 
}

pair<int, int> Task::GetOrder()
{
    return order; 
}

class Grid
{
private: 
    int width; 
    int height; 
    int max_size; 
public: 
    vector<vector<Task *> > city; 
    Grid(int width_in, int height_in, int max_size_in);
    ~Grid(); 
    int GetWidth();
    int GetHeight();
    int GetMaxSize();
};

Grid::Grid(int width_in, int height_in, int max_size_in)
// construct the whole grid and assign initial values for tasks inside using Task()
{
    width = width_in;
    height = height_in;
    max_size = max_size_in; 
    city.resize(width); 
    for (int i = 0; i < width; i ++) 
    {
        city[i].resize(height); 
    }
    int cnt = 0; 
    for (int i = 0; i < width; i ++)
    {
        for (int j = 0; j < height; j ++)
        {
            city[i][j] = new Task(max_size, 0, cnt); // initialize the tasks on layer 0, which means the original layer
            cnt ++; 
        }
    }
}

Grid::~Grid()
{
    for (int i = 0; i < width; i ++)
    {
        for (int j = 0; j < height; j ++)
        {
            delete city[i][j]; 
        }
    }
}

int Grid::GetWidth()
{
    return width; 
}

int Grid::GetHeight()
{
    return height; 
}

int Grid::GetMaxSize()
{
    return max_size; 
}

#endif
