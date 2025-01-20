import multiprocessing as mp

def collector(numbers, result, lock):
    """Calculate the sum of a list of numbers"""
    local_result =  [n * 2     for n in numbers]
    local_result += [n * 2 + 1 for n in numbers]
    with lock:
        result.value += local_result

def main():
    # list of numbers to sum
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # create a manager to share the result between processes
    manager = mp.Manager()
    result = manager.Value('i', [])
    lock = mp.Lock()

    # create a list of processes
    processes = []
    chunk = len(numbers) // 2
    for i in range(0, len(numbers), chunk):
        p = mp.Process(target=collector,
                       args=(numbers[i:i+chunk], result, lock))
        processes.append(p)
        p.start()

    # wait for all processes to finish
    for p in processes:
        p.join()

    # print the result
    print(result.value)

if __name__ == "__main__":
    main()
