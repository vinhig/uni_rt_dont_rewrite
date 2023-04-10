import random
samples = []

for i in range(32):
    sample = [random.random(), random.random()]
    print("vec2(", sample[0], ",", sample[1], "),")

    samples.append(sample)