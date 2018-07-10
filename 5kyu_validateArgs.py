from functools import wraps

def validate_args(*arg_types):
    def check_arg_type(fun):
        @wraps(fun)
        def wrapper(*args,**kwargs):
            for type,arg in zip(arg_types,args):
                if not isinstance(arg,type):
                    raise ValueError(f'Arg {arg} is not of type {type}')
            return fun(*args)
        return wrapper
    return check_arg_type

@validate_args(str)
def say_hello(name):
    """Greets person before executing. You know, to be kind"""
    return f'Hello, {name}'

print('Func name:', say_hello.__name__)
print(' Func doc:', say_hello.__doc__)

try:
    say_hello(5)
    say_hello('Greg')
except Exception as e:
    print(f'ERROR: {e}')
