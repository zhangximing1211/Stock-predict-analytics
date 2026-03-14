# Next Day Decision Tree Flowchart

```mermaid
flowchart TD
    N0{"Node 0<br/>return <= -0.0001 ?<br/>node p(up): 50.0%<br/>samples: 483867"}
    N1{"Node 1<br/>MACD_hist <= -0.1595 ?<br/>node p(up): 51.8%<br/>samples: 228648"}
    N2{"Node 2<br/>daily_return <= 0.0029 ?<br/>node p(up): 53.4%<br/>samples: 67596"}
    N3{"Node 3<br/>return_lag_1 <= -0.0344 ?<br/>node p(up): 53.9%<br/>samples: 62895"}
    N4["Leaf 4<br/>predict: UP<br/>p(up): 61.8%<br/>samples: 2916"]
    N5["Leaf 5<br/>predict: UP<br/>p(up): 53.6%<br/>samples: 59979"]
    N3 -- yes --> N4
    N3 -- no --> N5
    N6{"Node 6<br/>return <= -0.0222 ?<br/>node p(up): 46.8%<br/>samples: 4701"}
    N7["Leaf 7<br/>predict: DOWN<br/>p(up): 31.5%<br/>samples: 621"]
    N8["Leaf 8<br/>predict: DOWN<br/>p(up): 49.1%<br/>samples: 4080"]
    N6 -- yes --> N7
    N6 -- no --> N8
    N2 -- yes --> N3
    N2 -- no --> N6
    N9{"Node 9<br/>volume_ratio <= 0.5529 ?<br/>node p(up): 51.1%<br/>samples: 161052"}
    N10{"Node 10<br/>BB_width <= 0.0992 ?<br/>node p(up): 49.0%<br/>samples: 13352"}
    N11["Leaf 11<br/>predict: DOWN<br/>p(up): 47.7%<br/>samples: 7068"]
    N12["Leaf 12<br/>predict: UP<br/>p(up): 50.4%<br/>samples: 6284"]
    N10 -- yes --> N11
    N10 -- no --> N12
    N13{"Node 13<br/>RSI_14 <= 76.5259 ?<br/>node p(up): 51.3%<br/>samples: 147700"}
    N14["Leaf 14<br/>predict: UP<br/>p(up): 51.5%<br/>samples: 136719"]
    N15["Leaf 15<br/>predict: DOWN<br/>p(up): 49.0%<br/>samples: 10981"]
    N13 -- yes --> N14
    N13 -- no --> N15
    N9 -- yes --> N10
    N9 -- no --> N13
    N1 -- yes --> N2
    N1 -- no --> N9
    N16{"Node 16<br/>RSI_14 <= 60.6087 ?<br/>node p(up): 48.4%<br/>samples: 255219"}
    N17{"Node 17<br/>close_to_SMA <= -0.0309 ?<br/>node p(up): 49.5%<br/>samples: 154544"}
    N18{"Node 18<br/>return <= 0.0265 ?<br/>node p(up): 51.7%<br/>samples: 22097"}
    N19["Leaf 19<br/>predict: UP<br/>p(up): 50.8%<br/>samples: 19907"]
    N20["Leaf 20<br/>predict: UP<br/>p(up): 59.7%<br/>samples: 2190"]
    N18 -- yes --> N19
    N18 -- no --> N20
    N21{"Node 21<br/>log_return <= 0.0130 ?<br/>node p(up): 49.1%<br/>samples: 132447"}
    N22["Leaf 22<br/>predict: DOWN<br/>p(up): 49.7%<br/>samples: 94599"]
    N23["Leaf 23<br/>predict: DOWN<br/>p(up): 47.6%<br/>samples: 37848"]
    N21 -- yes --> N22
    N21 -- no --> N23
    N17 -- yes --> N18
    N17 -- no --> N21
    N24{"Node 24<br/>MACD_hist <= 0.0495 ?<br/>node p(up): 46.7%<br/>samples: 100675"}
    N25{"Node 25<br/>close_to_SMA <= 0.0032 ?<br/>node p(up): 44.2%<br/>samples: 22349"}
    N26["Leaf 26<br/>predict: UP<br/>p(up): 53.0%<br/>samples: 515"]
    N27["Leaf 27<br/>predict: DOWN<br/>p(up): 44.0%<br/>samples: 21834"]
    N25 -- yes --> N26
    N25 -- no --> N27
    N28{"Node 28<br/>RSI_14 <= 68.7723 ?<br/>node p(up): 47.4%<br/>samples: 78326"}
    N29["Leaf 29<br/>predict: DOWN<br/>p(up): 49.2%<br/>samples: 27740"]
    N30["Leaf 30<br/>predict: DOWN<br/>p(up): 46.4%<br/>samples: 50586"]
    N28 -- yes --> N29
    N28 -- no --> N30
    N24 -- yes --> N25
    N24 -- no --> N28
    N16 -- yes --> N17
    N16 -- no --> N24
    N0 -- yes --> N1
    N0 -- no --> N16
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
    class N10 decision;
    class N11 leaf;
    class N12 leaf;
    class N13 decision;
    class N14 leaf;
    class N15 leaf;
    class N16 decision;
    class N17 decision;
    class N18 decision;
    class N19 leaf;
    class N20 leaf;
    class N21 decision;
    class N22 leaf;
    class N23 leaf;
    class N24 decision;
    class N25 decision;
    class N26 leaf;
    class N27 leaf;
    class N28 decision;
    class N29 leaf;
    class N30 leaf;
```

## Notes

- `yes` means the condition is true and the path goes left.
- `no` means the condition is false and the path goes right.
- `samples` is the number of training rows that reached that node.
- `p(up)` is the share of training samples labeled as up in that node.

## Summary

```json
{
  "model_file": "/Users/zhangximing/Desktop/eda_plots/model_outputs/next_day_decision_tree/model.pkl",
  "model_type": "DecisionTreeClassifier",
  "max_depth": 4,
  "n_leaves": 16,
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
