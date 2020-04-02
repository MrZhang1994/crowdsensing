#ifndef _EVAL_H_
#define _EVAL_H_

#include <vector>
#include "task.h"
#include "server.h"
using namespace std;

class result_t
{
public:
    int total_delay;
    int total_transCost;
    int attrTrans;
    int num_failure;
};

// System initialization (returns attrTrans)
int systemInit(float HP4, vector<vector<server_t>> &servers, vector<task_t> &taskList);

// Evaluator
result_t evaluator(vector<vector<server_t>> &servers, vector<task_t> &taskList, float HP4, float HP5);

//determine suitable layer
void dispatcher(task_t &task, vector<vector<server_t>> &servers);

// matchFlag=matcher(task(eachTask), task(eachTask).target,Hp5)
bool matcher(const task_t &task, const server_t &server, float HP5);

#endif