import functools
import traceback


def find_executable_batch_size(function: callable = None, starting_batch_size: int = 128, auto_find_batch_size: bool = True):
    """
    This is the implementation of 

    Args:
        function (callable, optional): A function to wrap.
        starting_batch_size (int, optional): The batch size to try and fit into memory.
        auto_find_batch_size (bool, optional): If True, will try to find an executable batch size by reducing it on failures.

    A decorator that will try to execute `function`. If it fails from exceptions related to out-of-memory, the batch size
    is cut in half and passed to `function`. `function` must take in a `batch_size` parameter as its first argument.
    """

    if function is None:
        return functools.partial(
            find_executable_batch_size,
            starting_batch_size=starting_batch_size,
            auto_find_batch_size=auto_find_batch_size,
        )

    def wrapper(*args, **kwargs):
        if not auto_find_batch_size:
            return function(starting_batch_size, *args, **kwargs)

        batch_size = starting_batch_size
        while batch_size > 0:
            try:
                return function(batch_size, *args, **kwargs)
            except (MemoryError, RuntimeError) as e:
                # Check if it's a CUDA out-of-memory error
                if 'CUDA out of memory' in str(e):
                    print(f"Reducing batch size to {batch_size // 2} due to memory error.")
                    batch_size //= 2
                    print(f"New batch size: {batch_size}")
                else:
                    # If the exception is not memory related, re-raise it
                    raise e
            except Exception as e:
                # For any other exception, print the traceback and raise the exception
                traceback.print_exc()
                raise e

        raise ValueError("Could not find an executable batch size")

    return wrapper


def get_nested_attr(obj, attr):
    """Get a nested attribute."""
    for a in attr.split('.'):
        obj = getattr(obj, a)
    return obj

def set_nested_attr(obj, attr, value):
    """Set a nested attribute."""
    attrs = attr.split('.')
    for a in attrs[:-1]:
        obj = getattr(obj, a)
    setattr(obj, attrs[-1], value)

def has_nested_attr(obj, attr):
    """Check if an object has a nested attribute."""
    try:
        get_nested_attr(obj, attr)
        return True
    except AttributeError:
        return False