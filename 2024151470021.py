a = 100
b_list = [23, 5, 0, 76, 'A', 66]

for b in b_list:
    try:
        result = a / b
        print(f"{result:.4f}")
    except Exception as e:
        print(f"{e.__class__.__name__}: {e}")
