def GPU_selection(gpu_use:int = 70, min_memory:int=8000)->int:
    from os import name
    if name == 'nt': # When running at NIN
        return 0
    if name == 'posix':
        if type(gpu_use)!=int: print('gpu_use must be an int');       return -1
        if type(min_memory)!=int: print('min_memory must be an int'); return -1

        from subprocess import check_output
        from torch.cuda import device_count
        from numpy import argsort, all

        d_count = device_count()

        utilization_gpu = []
        free_memory     = []

        for device_id in range(0, d_count+1):
            #print('Device ID:  ', device_id)
            try:
                utilization = check_output(['nvidia-smi', '-i', str(device_id), '--query-gpu=utilization.gpu', '--format=csv'])
                utilization = int(str(utilization).split('\\n')[-2].split(' ')[0])
            except:   #sp.CalledProcessError
                utilization = -1
            #print('     Utilization  ', utilization)
            utilization_gpu.append(utilization)

            try:
                memory = check_output(['nvidia-smi', '-i', str(device_id), '--query-gpu=memory.free', '--format=csv'])
                memory = int(str(memory).split('\\n')[-2].split(' ')[0])
            except:   #subprocess.CalledProcessError
                memory = -1
            #print('     Memory:  ', memory)
            free_memory.append(memory)

        #print('GPU utilization:  ', utilization_gpu)
        #print('Memory:           ', free_memory)

        ##########

        # Conditions to select a GPU
        for gpus in argsort(free_memory)[::-1]:
            if utilization_gpu[gpus] < gpu_use and free_memory[gpus]>min_memory:
                return gpus
        print('No available GPUs with the requirements...')
        return -1