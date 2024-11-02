inputs = [1,2,3,2.5]
weights1 = [[0.2,0.8, -0.5, 1.0],
            [0.5,-0.91, 0.26, -0.5],
            [-0.26,-0.27, 0.17, 0.87]]

bias1 = [2,3,0.5]
output = []
for i in range(3):
    output.append(sum([inputs[x]*weights1[i][x] for x in range(len(inputs))]) + bias1[i])
print(output)