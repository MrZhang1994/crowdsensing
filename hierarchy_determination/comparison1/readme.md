# Comparison 1 Update 1

Some questions still remain in this version.

## Parameters
Parameters can be changed in `comparison.cpp`. They are currently defined by 
``` c++
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
```

## Compile
```bash
g++ -Wall -o comparison1 comparison.cpp edge.cpp task.cpp
```

## Execute
```bash
./comparison1
```

## Remaining Questions
1. Do isolated vertices exist in the system?
    (I assume they exist for now.)
2. Only `number of requests and responses` are counted towards `cost` for now (+2 for each search). 
3. Generation of tasks does not match the hierarchical method for now.
4. How to check area requirements? (For now a sensor with covering-area intersection with the task is accepted if the attributions meet the similarity threshold.)
5. Method of calculating similarity of attributions. For now I use `Tanimoto coefficient`. https://docs.tibco.com/pub/spotfire/6.5.2/doc/html/hc/hc_tanimoto_coefficient.htm