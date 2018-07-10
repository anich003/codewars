
def id_best_users(*args):
    customer_counts = {}
    for month in args:
        for customer in month:
            customer_counts[customer] = customer_counts.get(customer,0) + 1
    return [[]]
