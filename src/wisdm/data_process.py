'''
csv like.

data_dir/
  phone/
    gyro/
        data_16{$id}_gyro_phone.txt
    accel/
        data_16{$id}_accel_phone.txt
  watch/
    gyro/
        data_16{$id}_gyro_phone.txt
    accel/
        data_16{$id}_accel_phone.txt


Parameters
----------
$id : 0 ~ 50
'''


import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO


labels = {
    "A": "walking",
    "B": "jogging",
    "C": "stairs",
    "D": "sitting",
    "E": "standing",
    "F": "typing",
    "G": "teeth",
    "H": "soup",
    "I": "chips",
    "J": "pasta",
    "K": "drinking",
    "L": "sandwich",
    "M": "kicking",
    "O": "catch",
    "P": "dribbling",
    "Q": "writing",
    "R": "clapping",
    "S": "folding",
}

key2num = {v: k for k, v in enumerate(labels.keys())}  # key -> number
num2value = {k: v for k, v in enumerate(labels.items())}  # number -> value

gyro_column = ['usr id', 'label', 'time stamp', 'gyro-x', 'gyro-y', 'gyro-z']
accel_column = ['usr id', 'label', 'time stamp', 'acc-x', 'acc-y', 'acc-z']

data_dir = './data/wisdm-dataset/raw/'
phone = data_dir + 'phone/'
watch = data_dir + 'watch/'

# phone
k_x_phone = []
k_y_phone = []
k_timestamp_phone = []
for id in range(50):
    with open(phone + f'gyro/data_16{id:02d}_gyro_phone.txt', 'r') as file:
        data = file.readlines()
    cleaned_data = [line.rstrip(';\n') for line in data]
    cleaned_data_str = "\n".join(cleaned_data)

    gyro_df = pd.read_csv(StringIO(cleaned_data_str), header=None)
    gyro_df.columns = gyro_column
    gyro_y = gyro_df['label']

    with open(phone + f'accel/data_16{id:02d}_accel_phone.txt', 'r') as file:
        data = file.readlines()
    cleaned_data = [line.rstrip(';\n') for line in data]
    cleaned_data_str = "\n".join(cleaned_data)

    acc_df = pd.read_csv(StringIO(cleaned_data_str), header=None)
    acc_df.columns = accel_column
    acc_y = acc_df['label']
    timestamp = acc_df['time stamp']
    k_timestamp_phone.append(timestamp)

    if True:
        print(acc_df)
        print(gyro_df)
        # print(gyro_y.equals(acc_y))  # 两个label不是一样的

    continue
    k_x_phone.append(pd.concat([acc_df[accel_column[-3:]], gyro_df[gyro_column[-3:]]], axis=1))  # 合并两张表

    acc_y = acc_y.map(key2num).astype(int)  # 从0到17一共18类
    k_y_phone.append(acc_y)

    if True:
        print(k_x_phone[-1])
        print(k_y_phone[-1])

    x = k_x_phone[-1].to_numpy()
    timestamp = timestamp.to_numpy()
    timestamp = timestamp - timestamp[0]
    print(timestamp)
    exit(0)
    for axis in range(5):
        plt.plot(timestamp, x[:, axis].reshape(-1, 1))
    plt.show()
