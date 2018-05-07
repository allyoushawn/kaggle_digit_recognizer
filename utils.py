import numpy as np
def read_csv_data(file_name, data_type='test'):
    labels = []
    data = []
    with open(file_name) as fp:
        for line_id, line in enumerate(fp.readlines()):
            if line_id == 0:
                # The example line
                continue
            tokens = line.rstrip().split(',')
            if data_type == 'train' or data_type == 'dev':
                labels.append(int(tokens[0]))
                data.append( [float(x) / 255. for x in tokens[1:]])
            elif data_type == 'test':
                labels.append(0)
                data.append( [float(x) / 255. for x in tokens])

    return np.array(data, dtype=np.float32), np.array(labels)
