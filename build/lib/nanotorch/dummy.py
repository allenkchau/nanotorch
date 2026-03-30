from nanotorch.value import Value


def run_demo() -> Value:
    """Create two Values and return their sum."""
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    return c


if __name__ == "__main__":
    result = run_demo()
    print(f"Dummy demo result: {result.data}")
