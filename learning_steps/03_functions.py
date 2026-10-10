passengers = [
    {"name": "Owen Harris", "age": 22, "class": 3, "survived": False},
    {"name": "Florence Briggs", "age": 38, "class": 1, "survived": True},
    {"name": "Laina Heikkinen", "age": 26, "class": 3, "survived": True},
    {"name": "Lily Futrelle", "age": 35, "class": 1, "survived": True},
    {"name": "William Allen", "age": 35, "class": 3, "survived": False},
]

for p in passengers:
    print(f"{p['name']} ({p['age']} years old) - class {p['class']}")


def count_survivors(passenger_list):
    count = 0
    for p in passenger_list:
        if p["survived"]:   
            count = count + 1
    return count

def average_age(passenger_list):
    total = 0
    for p in passenger_list:
        total = total + p["age"]
    return total / len(passenger_list)

result = average_age(passengers)
print(f"Average age: {result}")
result = count_survivors(passengers)
print(f"Number of survivors: {result}")