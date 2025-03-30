import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import kagglehub

with open('./data.yml') as f:
    config = yaml.safe_load(f)
    path = config['path']
    draw_raw_data = config['draw_raw_data']
    draw_processed_data = config['draw_processed_data']


if path is None:
    path = kagglehub.dataset_download("piyanit/wisdm-ar-v11")
print("Path to dataset files:", path)


action2id = {
    'Walking': 0,
    'Jogging': 1,
    'Upstairs': 2,
    'Downstairs': 3,
    'Sitting': 4,
    'Standing': 5,
}

id2action = {
    0: 'Walking',
    1: 'Jogging',
    2: 'Upstairs',
    3: 'Downstairs',
    4: 'Sitting',
    5: 'Standing',
}

processed_list = []

'''
Data Clean
'''

with open(path + '/WISDM_ar_v1.1_raw.txt') as f:
    for ix, line in enumerate(f.read().strip().split(';')):
        if not line:
            print(f'ended at line: {ix}')
            break

        line = line.split(',')

        line[0] = int(line[0])
        line[1] = action2id[line[1]]
        line[2] = int(line[2])
        line[3] = float(line[3])
        line[4] = float(line[4])
        line[5] = float(line[5])

        if ix == 343419:
            '''
            error while reading: 343419: [11, 0, 1867172313000, 4.4, 4.4, 11.0, 'Walking', '1867222270000', '5.48', '8.43', '9.724928']
            '''

            # line.append([11, 0, 1867172313000, 4.4, 4.4, 11.0])
            line.append([11, 0, 1867222270000, 5.48, 8.43, 9.724928])
            continue
        if len(line) == 7 and line[-1] == '':
            '''
            error while reading: 844623: [21, 1, 117687383612000, -0.38, 7.59, 2.49, '']
            error while reading: 844624: [21, 1, 117687421515000, 0.8, 9.43, 0.08, '']
            error while reading: 844625: [21, 1, 117687461707000, 0.5, 8.16, 0.95, '']
            error while reading: 844626: [21, 1, 117687541571000, 1.76, 9.43, 1.92, '']
            error while reading: 844627: [21, 1, 117687581519000, 0.91, 8.66, 1.08, '']
            error while reading: 844628: [21, 1, 117687621527000, 0.89, 8.54, 2.49, '']
            error while reading: 844629: [21, 1, 117687701514000, 3.17, 9.0, 1.23, '']
            error while reading: 844630: [21, 1, 117687741522000, 0.65, 9.11, 1.5, '']
            error while reading: 844631: [21, 1, 117687782111000, -1.04, 10.15, 1.65, '']
            error while reading: 844632: [21, 1, 117687861578000, 3.21, 7.86, 0.04, '']
            error while reading: 844633: [21, 1, 117687861578000, 3.21, 9.81, -0.8, '']
            '''

            line.pop(-1)
        if line[2] == 0:
            '''
            error while reading: 316349: [18, 2, 0, 0.0, 0.0, 0.0]
            error while reading: 316350: [18, 2, 0, 0.0, 0.0, 0.0]
            error while reading: 316360: [18, 2, 0, 0.0, 0.0, 0.0]
            error while reading: 400441: [10, 0, 0, 0.0, 0.0, 0.0]
            error while reading: 413090: [10, 0, 0, 0.0, 0.0, 0.0]
            error while reading: 426628: [10, 3, 0, 0.0, 0.0, 0.0]
            error while reading: 429524: [10, 2, 0, 0.0, 0.0, 0.0]
            error while reading: 430607: [10, 3, 0, 0.0, 0.0, 0.0]
            error while reading: 431569: [10, 2, 0, 0.0, 0.0, 0.0]
            error while reading: 432678: [10, 3, 0, 0.0, 0.0, 0.0]
            error while reading: 433664: [10, 5, 0, 0.0, 0.0, 0.0]
            error while reading: 684889: [4, 3, 0, 0.0, 0.0, 0.0]
            error while reading: 684890: [4, 3, 0, 0.0, 0.0, 0.0]
            error while reading: 780744: [8, 3, 0, 0.0, 0.0, 0.0]
            error while reading: 882380: [3, 0, 0, 0.0, 0.0, 0.0]
            error while reading: 882381: [3, 0, 0, 0.0, 0.0, 0.0]
            error while reading: 882382: [3, 0, 0, 0.0, 0.0, 0.0]
            error while reading: 938217: [22, 3, 0, 0.0, 0.0, 0.0]
            error while reading: 938218: [22, 3, 0, 0.0, 0.0, 0.0]
            error while reading: 1091332: [19, 3, 0, 0.0, 0.0, 0.0]
            '''
            # print(f'error while reading: {ix}: {line}')
            continue
        elif len(line) != 6:
            # print(f'error while reading: {ix}: {line}')
            continue

        processed_list.append(line)


processed_list = pd.DataFrame(processed_list)
processed_list.columns = ['user_id', 'label', 'time', 'acc-x', 'acc-y', 'acc-z']
print('>> info: ')
processed_list.info()


'''
Plot
'''

processed_list = processed_list.to_numpy()

if draw_raw_data:
    # TIME_LEN = 128
    # START_FRAME = 2344
    for usr in np.unique(processed_list[:, 0]):
        usr_list = processed_list[processed_list[:, 0] == usr]

        plt.figure(figsize=(18, 6))
        for act in np.unique(usr_list[:, 1]):
            act_list = usr_list[usr_list[:, 1] == act]
            # np.save(path + f'/processed/user_{usr}/action_{act}.npy', act_list)

            plt.subplot(3, 2, int(act + 1))

            # single
            # plt.plot(act_list[START_FRAME:START_FRAME + TIME_LEN, 2], act_list[START_FRAME:START_FRAME + TIME_LEN, 3], label='acc-x')
            # plt.plot(act_list[START_FRAME:START_FRAME + TIME_LEN, 2], act_list[START_FRAME:START_FRAME + TIME_LEN, 4], label='acc-y')
            # plt.plot(act_list[START_FRAME:START_FRAME + TIME_LEN, 2], act_list[START_FRAME:START_FRAME + TIME_LEN, 5], label='acc-z')

            # all
            plt.plot(act_list[:, 2], act_list[:, 3], label='acc-x')
            plt.plot(act_list[:, 2], act_list[:, 4], label='acc-y')
            plt.plot(act_list[:, 2], act_list[:, 5], label='acc-z')

            plt.title(f'User {usr}, Action {act}')
            plt.xlabel('time')
            plt.ylabel('acc')
            plt.title(f'User {usr}, Action {act}')
            plt.tight_layout()
            plt.legend()

        plt.tight_layout()
        plt.savefig(f'./plots/raw/raw/user_{usr}.png')
        plt.close()


'''
Split
'''


def get_data(time_steps=128, tolerance=20):

    def convert(usr, act, act_list):
        res = []
        seq = []
        last_time = None

        for ix, input in enumerate(act_list):
            if len(seq) == 0:
                rate = 1
            elif len(seq) == 1:
                rate = 1
                last_time = input[2] - seq[-1][2]
            else:
                rate = (input[2] - seq[-1][2]) / last_time
                last_time = input[2] - seq[-1][2]

            if (1. / tolerance) <= rate and rate <= tolerance:
                seq.append(input)
            else:
                res += sliding_window(seq, time_steps, time_steps // 2)  # overlap 50%
                last_time = None
                seq = [input]

        return res

    def sliding_window(x, step, stride):
        return [x[i:i + step] for i in range(0, len(x) - step, stride)]  # drop the last one

    data_dict = {}
    for usr in np.unique(processed_list[:, 0]):
        usr_list = processed_list[processed_list[:, 0] == usr]

        for act in np.unique(usr_list[:, 1]):

            act_list = usr_list[usr_list[:, 1] == act]
            data_dict[f'u_{int(usr)}_a_{int(act)}'] = np.array(convert(usr, act, act_list)).reshape(-1, time_steps, 6)

    return data_dict


if draw_processed_data['isdraw']:
    TIME_LEN = draw_processed_data['time_len']
    TOLERANCE = draw_processed_data['tolerance']
    sum = 0

    data_dict = get_data(TIME_LEN, TOLERANCE)

    for k, v in data_dict.items():
        sum += v.shape[0]
        print(f'{k}: {v.shape}')

    print(f'total: {sum}')

    for k, v in data_dict.items():
        user, action = k.split('_')[1], k.split('_')[3]

        if not os.path.exists('./plots/raw/splited'):
            os.makedirs('./plots/raw/splited')

        if v.shape[0] == 0:
            continue

        for ix, x in enumerate(v):
            if ix <= 3:
                plt.figure(figsize=(18, 6))
                plt.plot(x[:, 2], x[:, 3], label='acc-x')
                plt.plot(x[:, 2], x[:, 4], label='acc-y')
                plt.plot(x[:, 2], x[:, 5], label='acc-z')
                plt.title(f'User {user}, Action {action}')
                plt.xlabel('time')
                plt.ylabel('acc')
                plt.title(f'User {user}, Action {action}')
                plt.tight_layout()
                plt.legend()
                plt.savefig(f'./plots/raw/splited/{k}_{ix}.svg', format='svg')
                plt.close()


'''
Make Train, Test Dataset
'''

data_dict = get_data(
    time_steps=64,
    tolerance=20
)


static_list = [0] * 6
for k, v in data_dict.items():
    user, action = k.split('_')[1], k.split('_')[3]

    static_list[int(action)] += v.shape[0]

for ix, num in enumerate(static_list):
    print(f'Action {id2action[ix]}: {num}')

data = np.concatenate(list(data_dict.values()), axis=0)  # (num, 128, 6)

rate = 0.7
train, test = data[:int(data.shape[0] * rate)], data[int(data.shape[0] * rate):]

X_train, y_train = np.transpose(train[:, :, 3:], (0, 2, 1)), train[:, 1, 1]
X_test, y_test = np.transpose(test[:, :, 3:], (0, 2, 1)), test[:, 1, 1]


'''
Convert into Tensor
'''

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).long()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).long()

y_train = torch.functional.F.one_hot(y_train, num_classes=6).float()
# y_test = F.one_hot(y_test, num_classes=6).float()

print(f'Shape of X_train: {X_train.shape}')
print(f'Shape of y_train: {y_train.shape}')
print(f'Shape of X_test: {X_test.shape}')
print(f'Shape of y_test: {y_test.shape}')
