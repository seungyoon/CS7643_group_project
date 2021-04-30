import generators

# Add - Long binary addition (badd)
# least-significant bit left.
# 2 -> 1, 1 -> 0, 11 -> +, 0 -> end_token

length = 40
test_length = 400

# Binary Add
#example = generators.generators['badd'].get_batch(length,20480*10)
#example = generators.generators['badd'].get_batch(length,2560*10)
example = generators.generators['badd'].get_batch(test_length,2560*10)

# Add
#example = generators.generators['add'].get_batch(8,20480)
#example = generators.generators['add'].get_batch(8,2560)
#example = generators.generators['add'].get_batch(8,2560)

# Reverse
#example = generators.generators['rev'].get_batch(8,20480)
#example = generators.generators['rev'].get_batch(8,2560)
#example = generators.generators['rev'].get_batch(8,2560)

# Copy
#example = generators.generators['scopy'].get_batch(8,20480)
#example = generators.generators['scopy'].get_batch(8,2560)
#example = generators.generators['scopy'].get_batch(8,2560)

input, target, taskid = example
input = input.astype(int)

#print(input[0])
#print(target[0])
#print(input[1])
#print(target[1])
#exit(0)


print("input,target")
length = len(input)

for i in range(length):
#    print("<sos> ", end="")
    for num in input[i][0]:
        print(num, end=' ')
    
#    print('<eos>, <sos>', end=' ')
    print(',', end=' ')

    for num in target[i]:
        print(num, end=' ')

#    print("<eos>")
    print("")

#print(input)
#print(target)
