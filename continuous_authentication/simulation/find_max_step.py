import statistics
test = [0,2,3,4,5,10,20,40,100,100000]

def find_max(y):
    max_thresh = max(y)
    #find a way to graph more of the smaller steps instead of the larger steps
    step = False
    return max_thresh , step

if __name__ == "__main__":
    x = find_max(test)
    print(x)
