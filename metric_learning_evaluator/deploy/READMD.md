# Auto Verification

## Introduction

Auto verification system which integrates both metric_learning_evaluator & db_verification.

## Quick start

### Installation
1. install [`metric_learning_evaluator`](http://awsgit.viscovery.co/Cybertron/metric-learning-evaluator/tree/feat/auto-verification#installation-1)
2. install [`libhnsw`](http://awsgit.viscovery.co/Cybertron/metric-learning-evaluator/tree/feat/auto-verification#intallation-of-hnswlib)
3. install [`db_verification`](http://awsgit.viscovery.co/algo_experiment/db-verification)

### Application
The application code in path **metric_learning_evaluator/tools/auto_verification_report.py**

#### Example
Path of embedding_db

```bash
export DB_PATH=/vol/08812401/kevin.zhao/cybertron/metric-learning-evaluator/Ijysheng/embedding_db_extracted_features_0801_anchor_gt_query_srmega_cntbase_es256/
```

Run command
```
python3.6 metric_learning_evaluator/tools/auto_verification_report.py -sc metric_learning_evaluator/deploy/sys_config.yaml  -dd $DB_PATH/embedding_db.pkl
```

Terminal would show...

```

Container initialized.
=============== embedding_container ===============
embeddings: (9759, 256)
internals: label_ids, label_names, filename_strings,
attributes: query, anchor
==================================================

7560 anchors, 2199 queries
HNSW Index Agent is initialized with 7560 features
Start indexing...
Indexing finished, 219900 retrieved events
Start exporting results...
Preprocessing done
Start join two data frames...
dataframe joined
Start split dataframes into query & anchor...
query_df & anchor_df are splitted
Save results and events into '/tmp/auto_verification_workspace/temp_results'
Start split dataframes into query & anchor...
query_df & anchor_df are splitted
Select #of anchors = 7560
    label_name  top 1 accuracy  top 3 accuracy  top 5 accuracy
0      黑鑽巧克力麵包        1.000000        1.000000        1.000000
1         白醬燻雞        1.000000        1.000000        1.000000
2      黑糖穀物小餐包        1.000000        1.000000        1.000000
3       烤鮪魚三明治        1.000000        1.000000        1.000000
4      火腿蛋沙拉可頌        1.000000        1.000000        1.000000
5       港式奶皇菠蘿        1.000000        1.000000        1.000000
6         海鹽羅宋        1.000000        1.000000        1.000000
7       海苔肉鬆沙拉        1.000000        1.000000        1.000000
8         法國大蒜        1.000000        1.000000        1.000000
9      歐式核桃蔓越莓        1.000000        1.000000        1.000000
10         椰子塔        1.000000        1.000000        1.000000
11     明太子法國麵包        1.000000        1.000000        1.000000
12       明太子可頌        1.000000        1.000000        1.000000
13    抹茶紅豆麻糬麵包        1.000000        1.000000        1.000000
14       德國鹹乳酪        1.000000        1.000000        1.000000
15       帕瑪森起士        1.000000        1.000000        1.000000
16        巨蛋核果        1.000000        1.000000        1.000000
17        大綠豆凸        1.000000        1.000000        1.000000
18     夢幻火腿三明治        1.000000        1.000000        1.000000
19    喜馬拉雅岩鹽可頌        1.000000        1.000000        1.000000
20      千層歐蕾蛋塔        1.000000        1.000000        1.000000
21       冰心維也納        1.000000        1.000000        1.000000
22       克林姆麵包        1.000000        1.000000        1.000000
23      丹麥蜂蜜年輪        1.000000        1.000000        1.000000
24        丹麥菠蘿        1.000000        1.000000        1.000000
25       丹麥熱狗排        1.000000        1.000000        1.000000
26       丹麥奶油卷        1.000000        1.000000        1.000000
27         甜甜圈        1.000000        1.000000        1.000000
28    4入桂圓核桃蛋糕        1.000000        1.000000        1.000000
29  蛋沙拉火腿起士三明治        1.000000        1.000000        1.000000

```

### Components

#### EmbeddingDB or EmbeddingContainer

#### ResultContainer

#### joined_df
