# LBSC+

## Introduction
LBSC+, a learning-based cost-aware caching framework that jointly optimizes ad-mission and eviction for cloud databases.

we evaluate the performance of LBSC+ on synthetic datasets based on real-world workloads, this part is implemented based on the cache framework webcachesim(https://github.com/sunnyszy/lrb).


## Dependencies and Build
```
See scripts/install.sh
```

## Dataset
```
The CDN database can be downloaded from http://lrb.cs.princeton.edu/wiki2018.tr.tar.gz.
```

## Quick Start
- Generating cost for the real-world dataset. caching algorithm(LRU) and cache size(4294967296) are arbitrary. The last three parameters are used to adjust whether transfer cost dominates or computation cost dominates.
```
webcachesim_cli data_path LRU 4294967296 --delta_ratio=xx --fixed_byte=xx --min_ratio=xx 
```
run the following cmd:

```
webcachesim_cli data_path LBSCP 4294967296 
