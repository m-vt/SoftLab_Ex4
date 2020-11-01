
import numpy
import random
import time


class PERSON:
    def __init__(self, has_corona, bored_time, arrival_time, service_time):
        self.has_corona = has_corona
        self.bored_time = bored_time
        self.arrival_time = arrival_time
        self.bored_reception_clock = arrival_time + bored_time
        self.respons_time = service_time
        # reception
        self.service_time = service_time
        self.end_time = None
        self.wait_in_reception_queue = None
        self.total_wait = None


n = 10_000_000  # number of patinets
# M = int(input("number of rooms: "))
alpha = int(input("mean time to bored: "))
u = float(input("mean reception service rate: "))
landa = float(input("mean arrival rate: "))

"""
# M = 2  # number of rooms
# alpha = 20  # mean time to bored
# u = 0.8  # service rate for paziresh
# landa = 2  # arrival rate
# 
# number_of_doctors_per_room = [2, 3]  # , 2, 3 ]
# mean_check_up_time = [[10, 7], [6, 18, 9]]
"""
start_time = time.time()

perosons = []
perosons_corona = []
perosons_normal = []

service_time_for_paziresh = [i + 1 for i in list(numpy.random.poisson(u, n))]
arrival_time = list(numpy.random.poisson(landa, n))

for i in range(len(arrival_time)):
    if i == 0:
        continue
    arrival_time[i] = arrival_time[i - 1] + arrival_time[i]

corona_totals = int(0.1 * n)

type_1_list_idx = random.sample(range(n), corona_totals)
type_1_list_idx.sort()
corona_temp_index = 0

for i in range(n):
    if i == type_1_list_idx[corona_temp_index]:
        perosons_corona.append(PERSON(True, alpha, arrival_time[i], service_time_for_paziresh[i]))
        if corona_temp_index < corona_totals - 1:
            corona_temp_index += 1
    else:
        perosons_normal.append(PERSON(False, alpha, arrival_time[i], service_time_for_paziresh[i]))