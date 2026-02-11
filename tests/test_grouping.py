import pandas as pd
from rl_fair.grouping import make_intersectional_groups, encode_groups

def test_make_groups():
    df = pd.DataFrame({"a":[0,0,1], "b":[1,1,0]})
    g = make_intersectional_groups(df, ["a","b"])
    assert len(g) == 3
    enc, labels = encode_groups(g)
    assert enc.min() == 0
    assert enc.max() <= 2
