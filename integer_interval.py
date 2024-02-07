

class IntegerInterval:

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __add__(self, other: [int, float, 'IntegerInterval']):
        if isinstance(other, IntegerInterval):
            return IntegerInterval(self.min + other.min, self.max + other.max)
        elif isinstance(other, (int, float)):
            return IntegerInterval(self.min + other, self.max + other)
        else:
            raise NotImplementedError("You can only add an Interval or an integer to an Interval")

    def __sub__(self, other: [int, float, 'IntegerInterval']):
        return self + (-other)

    def __neg__(self):
        return IntegerInterval(-self.max, -self.min)

    def __mul__(self, other: [int, float, 'IntegerInterval']):
        if isinstance(other, IntegerInterval):
            return IntegerInterval(self.min * other.min, self.max * other.max)
        elif isinstance(other, (int, float)):
            return IntegerInterval(self.min * other, self.max * other)
        else:
            raise NotImplementedError("You can only multiply an Interval by an integer or a float")

    def __str__(self):
        return f"[{self.min}, ..., {self.max}]"
