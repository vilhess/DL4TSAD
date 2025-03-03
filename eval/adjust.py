# Adjust predict but just on the next observations: if I detect an anomaly at time t and t+1 is also abnormal, then I also detect it.

def adjust_predicts(actual, predict):
    assert len(actual)==len(predict), "labels and detections should have the same shape"
    for i in range(len(predict)):
        if actual[i] and predict[i]:
            for j in range(i, len(predict), 1):
                if not actual[j]:
                    break
                else:
                    predict[j]=1
    return predict


# Compute prediction latency: the time needed to detect the begin of an anomaly period

def find_anomaly_periods(actual):
    periods = []
    in_anomaly = False
    start = None
    
    for i, val in enumerate(actual):
        if val == 1 and not in_anomaly:
            start = i
            in_anomaly = True
        elif val == 0 and in_anomaly:
            periods.append((start, i - 1))
            in_anomaly = False
    
    if in_anomaly:
        periods.append((start, len(actual) - 1))
    
    return periods

def compute_latencies(actual, predic):
    assert len(actual)==len(predic), "labels and detections should have the same shape" 
    anomaly_periods = find_anomaly_periods(actual)
    latencies = []
    sizes = []
    
    for start, end in anomaly_periods:
        detection_time = next((i for i in range(start, end + 1) if predic[i] == 1), None)
        latency = detection_time - start if detection_time is not None else None
        latencies.append(latency)
        sizes.append(end - start + 1)
    
    return latencies, sizes