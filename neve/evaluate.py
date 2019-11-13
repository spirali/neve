def bound_difference(old_bounds, new_bounds):
    old_size = old_bounds[1] - new_bounds[0]
    new_size = old_bounds[1] - new_bounds[0]
    return new_size / old_size
