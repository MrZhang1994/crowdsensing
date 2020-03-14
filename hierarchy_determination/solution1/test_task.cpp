#include "task.h"
#include <ctime>
#include <iostream>
using namespace std;


int main()
{
    srand48(time(NULL));
    /*
    task_t task(3,10,3,1,1);
    string a = task.logTask();
    cout << a << endl;
    */
    vector<task_t> taskList = taskGenerator(40, 20, 20, 3, 1, 0);
    return 0;
}