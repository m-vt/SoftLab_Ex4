
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

class Reception_class:
    def __init__(self, persons_corona, persons_normal):
        self.Queue_to_room = []
        self.corona_patient_queue = persons_corona.copy()
        self.corona_pat_q_index = 0
        self.normal_patient_queue = persons_normal.copy()
        self.normal_pat_q_index = 0
        self.reception_clock_finish_service = -1
        self.reception_busy = False
        self.patient_in_reception = None

        self.total_time_wait_in_q_corona_pats = 0
        self.total_time_wait_in_q_normal_pats = 0

        self.time_tired = 0
        self.time_while = 0
        self.time_add = 0
        self.counter = 0

    def Reception(self, clock, number_of_patients):
        left_reception_corona_pats = 0
        left_reception_normal_pats = 0
        time2 = time.time()
        if clock >= self.reception_clock_finish_service:

            if clock == self.reception_clock_finish_service:
                self.reception_busy = False
                self.Queue_to_room.append(self.patient_in_reception)

            # if  pat is tired -- > left qu
            time2 = time.time()

            while True:
                if (len(self.corona_patient_queue) - self.corona_pat_q_index) and self.corona_patient_queue[
                    self.corona_pat_q_index].bored_reception_clock <= clock:
                    self.corona_pat_q_index += 1
                    number_of_patients -= 1
                    left_reception_corona_pats += 1
                else:
                    break

            while True:
                if (len(self.normal_patient_queue) - self.normal_pat_q_index) and self.normal_patient_queue[
                    self.normal_pat_q_index].bored_reception_clock <= clock:
                    self.normal_pat_q_index += 1
                    number_of_patients -= 1
                    left_reception_normal_pats += 1
                else:
                    break

            time3 = time.time()
            self.time_tired += (time3 - time2)

            # next pat come to reception

            if len(self.corona_patient_queue) - self.corona_pat_q_index and self.corona_patient_queue[
                self.corona_pat_q_index].arrival_time <= clock:
                self.patient_in_reception = self.corona_patient_queue[self.corona_pat_q_index]
                self.corona_pat_q_index += 1
                self.reception_busy = True
                self.reception_clock_finish_service = clock + self.patient_in_reception.service_time
                # wait_in_reception_queue = \
                self.patient_in_reception.wait_in_reception_queue = clock - self.patient_in_reception.arrival_time
                self.total_time_wait_in_q_corona_pats += self.patient_in_reception.wait_in_reception_queue

            elif len(self.normal_patient_queue) - self.normal_pat_q_index and self.normal_patient_queue[
                self.normal_pat_q_index].arrival_time <= clock:

                self.patient_in_reception = self.normal_patient_queue[self.normal_pat_q_index]
                self.normal_pat_q_index += 1
                self.reception_clock_finish_service = clock + self.patient_in_reception.service_time
                self.reception_busy = True
                self.patient_in_reception.wait_in_reception_queue = clock - self.patient_in_reception.arrival_time
                self.total_time_wait_in_q_normal_pats += self.patient_in_reception.wait_in_reception_queue

            time4 = time.time()
            self.time_add += time4 - time3

        return number_of_patients, left_reception_corona_pats, left_reception_normal_pats


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