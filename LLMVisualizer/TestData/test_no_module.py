# Edge case: has class but no nn.Module/Model inheritance
class MyHelper:
    def __init__(self):
        self.data = []

    def process(self):
        pass

# Some random code
x = 42
y = x * 2
print(f"Hello {y}")
