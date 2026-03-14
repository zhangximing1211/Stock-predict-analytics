# Future 5D Decision Tree Flowchart

```mermaid
flowchart TD
    N0{"Node 0<br/>close_to_SMA <= -0.0284 ?<br/>node avg return: +0.27%<br/>samples: 481867"}
    N1{"Node 1<br/>price_range <= 0.0371 ?<br/>node avg return: +0.72%<br/>samples: 74770"}
    N2{"Node 2<br/>return_lag_1 <= -0.0236 ?<br/>node avg return: +0.57%<br/>samples: 60445"}
    N3{"Node 3<br/>volatility_20 <= 0.0206 ?<br/>node avg return: +0.90%<br/>samples: 9181"}
    N4["Leaf 4<br/>predict 5d return: +0.76%<br/>samples: 6582"]
    N5["Leaf 5<br/>predict 5d return: +1.25%<br/>samples: 2599"]
    N3 -- yes --> N4
    N3 -- no --> N5
    N6{"Node 6<br/>volatility_20 <= 0.0092 ?<br/>node avg return: +0.51%<br/>samples: 51264"}
    N7["Leaf 7<br/>predict 5d return: +0.10%<br/>samples: 4247"]
    N8["Leaf 8<br/>predict 5d return: +0.55%<br/>samples: 47017"]
    N6 -- yes --> N7
    N6 -- no --> N8
    N2 -- yes --> N3
    N2 -- no --> N6
    N9{"Node 9<br/>close_to_SMA <= -0.1501 ?<br/>node avg return: +1.38%<br/>samples: 14325"}
    N10["Leaf 10<br/>predict 5d return: +3.23%<br/>samples: 907"]
    N11{"Node 11<br/>price_momentum_20 <= -0.2055 ?<br/>node avg return: +1.26%<br/>samples: 13418"}
    N12["Leaf 12<br/>predict 5d return: -0.46%<br/>samples: 619"]
    N13["Leaf 13<br/>predict 5d return: +1.34%<br/>samples: 12799"]
    N11 -- yes --> N12
    N11 -- no --> N13
    N9 -- yes --> N10
    N9 -- no --> N11
    N1 -- yes --> N2
    N1 -- no --> N9
    N14{"Node 14<br/>volatility_20 <= 0.0093 ?<br/>node avg return: +0.18%<br/>samples: 407097"}
    N15{"Node 15<br/>close_position <= 0.9213 ?<br/>node avg return: -0.04%<br/>samples: 90878"}
    N16{"Node 16<br/>BB_width <= 0.0399 ?<br/>node avg return: -0.01%<br/>samples: 77568"}
    N17["Leaf 17<br/>predict 5d return: -0.12%<br/>samples: 27242"]
    N18["Leaf 18<br/>predict 5d return: +0.06%<br/>samples: 50326"]
    N16 -- yes --> N17
    N16 -- no --> N18
    N19{"Node 19<br/>MACD <= 1.9948 ?<br/>node avg return: -0.26%<br/>samples: 13310"}
    N20["Leaf 20<br/>predict 5d return: -0.23%<br/>samples: 11995"]
    N21["Leaf 21<br/>predict 5d return: -0.61%<br/>samples: 1315"]
    N19 -- yes --> N20
    N19 -- no --> N21
    N15 -- yes --> N16
    N15 -- no --> N19
    N22{"Node 22<br/>RSI_14 <= 57.7255 ?<br/>node avg return: +0.25%<br/>samples: 316219"}
    N23{"Node 23<br/>volatility_20 <= 0.0262 ?<br/>node avg return: +0.34%<br/>samples: 169020"}
    N24["Leaf 24<br/>predict 5d return: +0.31%<br/>samples: 160265"]
    N25["Leaf 25<br/>predict 5d return: +1.01%<br/>samples: 8755"]
    N23 -- yes --> N24
    N23 -- no --> N25
    N26{"Node 26<br/>return <= 0.0005 ?<br/>node avg return: +0.14%<br/>samples: 147199"}
    N27["Leaf 27<br/>predict 5d return: +0.23%<br/>samples: 60568"]
    N28["Leaf 28<br/>predict 5d return: +0.07%<br/>samples: 86631"]
    N26 -- yes --> N27
    N26 -- no --> N28
    N22 -- yes --> N23
    N22 -- no --> N26
    N14 -- yes --> N15
    N14 -- no --> N22
    N0 -- yes --> N1
    N0 -- no --> N14
    classDef decision fill:#f7f3d6,stroke:#7a6a00,color:#2f2a00,stroke-width:1px;
    classDef leaf fill:#e3f4ea,stroke:#1b6b45,color:#123524,stroke-width:1px;
    class N0 decision;
    class N1 decision;
    class N2 decision;
    class N3 decision;
    class N4 leaf;
    class N5 leaf;
    class N6 decision;
    class N7 leaf;
    class N8 leaf;
    class N9 decision;
    class N10 leaf;
    class N11 decision;
    class N12 leaf;
    class N13 leaf;
    class N14 decision;
    class N15 decision;
    class N16 decision;
    class N17 leaf;
    class N18 leaf;
    class N19 decision;
    class N20 leaf;
    class N21 leaf;
    class N22 decision;
    class N23 decision;
    class N24 leaf;
    class N25 leaf;
    class N26 decision;
    class N27 leaf;
    class N28 leaf;
```

## Notes

- `yes` means the condition is true and the path goes left.
- `no` means the condition is false and the path goes right.
- `samples` is the number of training rows that reached that node.
- `predict 5d return` is the average target return for the training rows in that leaf.

## Summary

```json
{
  "model_file": "/Users/zhangximing/Desktop/eda_plots/model_outputs/future_5d_decision_tree/model.pkl",
  "model_type": "DecisionTreeRegressor",
  "max_depth": 4,
  "n_leaves": 15,
  "feature_names": [
    "return",
    "log_return",
    "close_to_SMA",
    "RSI_14",
    "MACD",
    "MACD_hist",
    "BB_width",
    "volatility_20",
    "price_momentum_20",
    "volume_ratio",
    "close_position",
    "return_lag_1",
    "daily_return",
    "price_range"
  ]
}
```
