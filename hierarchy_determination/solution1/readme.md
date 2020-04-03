This is the archive containing solution to hierarchy_determination

## Server attributions updated
Server attributions are 2 dimensional now.

# First complete draft updated
## Project hierachy
``` 
hsd.cpp
  |- task.h / task.cpp
      |- taskGenerator
  |- srever.h / server.cpp
  |- eval.h / eval.cpp
      |- evaluator
      |- dispatcher
      |- matcher
```

## Compile
```bash
g++ -Wall -o hsd hsd.cpp task.cpp server.cpp eval.cpp
```
## Execute
```bash
./hsd
```
## Parameters
All relavent parameters can be changed in `hsd.cpp`.

## Output files
Relavent output files are all in `.csv` format. `config_#.csv` records configurations of server system. Each cluster number corresponds to one configuration result. `task_list.csv` records generated task list.

# Previous Updates
## Task test
### Compile
```bash
g++ -Wall -o test_task test_task.cpp task.cpp
```
### Execute
```bash
./test_task
```
Check the output file `task_list.csv` using database software, Excel, etc. The seperator is `","` for now.

## HSD test
HSD without evalutation updated. Clustering results and server configuration results (for all differnent number of clusters) are ouputted in standard output. `main` function is in `hsd.cpp`. Necessary changes are applied to code in the folder `cluster` to accommodate the system.

### Compile
```bash
g++ -Wall -o hsd hsd.cpp task.cpp server.cpp
```
### Execute
```bash
./hsd
```
### Parameters
All relavent parameters can be changed in `hsd.cpp`.

```cpp
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
#define show_server_configurations      // different cluster numbers
```
### Errors
If parameters are not appropriately selected, possible errors/warnings would occur:
1. Too large cluster number
```cpp
cerr << "Error: clustering failed with too large number of clusters" << endl;
```
2. Too small capacity 

Cannot even store a single attribution
```cpp
cerr << "Capacity of L" << num_layer << " servers are too small" << endl;
```
3. Too large capacity

Capacity is too large so that all the attributions can be stored in servers of upper layers (layer No. > 1)
```cpp
cerr << "WARNING (occurred when cluster number is " << clusterNum << "):" << endl;
cerr << "Capacity of L" << num_layer << " servers are too big" << endl;
cerr << "All attributions are stored in " << num_layer << " servers" << endl;
```
