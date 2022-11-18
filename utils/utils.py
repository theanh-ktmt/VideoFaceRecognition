# Convert frame index that celebs occur to range of time
def convert_findex_to_time(index_list, fps, capture_every):
    start = 0
    result = []
    
    if len(index_list) == 0:
        return result
    
    for i in range(1, len(index_list)+1):
        if i == len(index_list):
            result.append((index_list[start]/fps, index_list[i - 1]/fps))
            continue
        
        if index_list[i] != index_list[i - 1] + capture_every * fps:
            result.append((index_list[start]/fps, index_list[i - 1]/fps))
            start = i
            
    return result