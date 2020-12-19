import pprint
from collections import Counter
from typing import List, Tuple


# sex, age, survived
data = [['male', 20, False],
        ['female', 20, True],
        ['male', 8, True],
        ['female', 30, True],
        ['male', 16, False],
        ['female', 50, False],
        ['male', 8, True],
        ['female', 18, True],
        ['male', 10, True],
        ['female', 50, False],
        ]

def divide_set(rows: List, column: int, value: float) -> Tuple:
    split_function = lambda row:row[column] >= value
    set1 = []
    set2 = []
    for row in rows:
        if split_function(row):
            set1.append(row)
        else:
            set2.append(row)
    return (set1, set2)

def get_count(data: tuple) -> List:
    resulting_data = []
    counter = [Counter() for _ in range(len(data))]
    for elem, ct in zip(data, counter):
        ct.update(elem[2])
        resulting_data.append((elem, ct))
    return resulting_data

resulting_data = get_count(divide_set(data, 1, 15))
pp = pprint.PrettyPrinter()
pp.pprint(resulting_data)
