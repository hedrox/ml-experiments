from collections import Counter
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

def divide_set(rows, column, value):
    split_function = lambda row:row[column] >= value
    set1 = []
    set2 = []
    for row in rows:
        if split_function(row):
            set1.append(row)
        else:
            set2.append(row)
    return (set1, set2)

# print(divide_set(data, 1, 15))

def get_count(data):
    resulting_data = []
    counter = [Counter() for _ in range(len(data))]
    for elem, counter in zip(data,counter):
        counter.update(elem[2])
        resulting_data.append((elem,counter))
    return resulting_data

resulting_data = get_count(divide_set(data,1,15))
print(resulting_data)

# for elem in divide_set(data, 1, 15):
#     for el in elem:
#         counter.update([el[2]])
# print(counter)