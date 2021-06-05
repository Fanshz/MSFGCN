import numpy as np
import json
import matplotlib.pyplot as plt


s_data = './data/20140816_train_F'
flete = open(s_data + '.txt', "r")
lines = flete.readlines()
speed = []
weather = []
count = 0
for line in lines:
    if count % 300 == 0:
        elements = json.loads(line, encoding="utf-8")
        weather.append(elements["weather"])
        speed.append(elements["speed"])

speed_mean = 0
mean_list = []
speed_std = 0
std_list = []
for i in range(len(speed)):
    mean_list.append(np.mean(speed[i]))
    std_list.append(np.std(speed[i], ddof=1))
    pass
speed_mean = np.mean(mean_list)
speed_std = np.mean(std_list)

weather_mean = 0
mean_list = []
weather_std = 0
std_list = []
for i in range(len(weather)):
    mean_list.append(np.mean(weather[i]))
    std_list.append(np.std(weather[i], ddof=1))
    pass
weather_mean = np.mean(mean_list)
weather_std = np.mean(std_list)

data_evaluation = {
    "speed_mean":speed_mean,
    "speed_std":speed_std,
    "weather_mean":weather_mean,
    "weather_std":weather_std
}
print(data_evaluation)
