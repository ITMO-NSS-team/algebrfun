

def mape(a, b):
    try:
        return abs(a-b)/abs(a+b)
    except RuntimeWarning:
        return abs(a - b) / abs(a + b)