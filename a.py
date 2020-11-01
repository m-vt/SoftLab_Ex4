
import numpy
import random
import time
from numpy import random as rn


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

class DOCTOR:
    def __init__(self, mean_service_rate):
        self.check_up_mean_service_rate = mean_service_rate
        self.cur_pat_type_corona = None
        self.finish_check_up_clock = -1

class ROOM:
    def __init__(self, room_num, number_of_doctors, mean_service_rates):
        self.Doctors = []
        for i in range(number_of_doctors):
            self.Doctors.append(DOCTOR(mean_service_rates[i]))
        self.corona_patients_queue = []
        self.normal_patients_queue = []
        self.number_of_corona_pat_in_room = 0
        self.number_of_normal_pat_in_room = 0
        self.room_is_full = False
        self.total_finished = 0
        self.total_time_in_sys_corona_pats = 0
        self.total_time_in_sys_normal_pats = 0
        self.total_time_wait_in_room_q_corona = 0
        self.total_time_wait_in_room_q_normal = 0

    def check_up(self, clock):

        number_of_busy_doctors = 0
        number_of_pat_finished_check_up = 0

        for doctor in self.Doctors:

            if doctor.finish_check_up_clock > clock:
                number_of_busy_doctors += 1
                continue

            if doctor.finish_check_up_clock <= clock:
                if doctor.finish_check_up_clock == clock:
                    if doctor.cur_pat_type_corona:
                        self.number_of_corona_pat_in_room -= 1
                    else:
                        self.number_of_normal_pat_in_room -= 1
                    self.total_finished += 1
                    number_of_pat_finished_check_up += 1
                    number_of_busy_doctors -= 1

                if (len(self.corona_patients_queue) or len(self.normal_patients_queue)):

                    number_of_busy_doctors += 1
                    service_rate = int(rn.exponential(doctor.check_up_mean_service_rate)) + 1

                    doctor.finish_check_up_clock = clock + service_rate

                    if len(self.corona_patients_queue):
                        doctor.cur_pat_type_corona = True
                        patient = self.corona_patients_queue.pop(0)
                        self.number_of_corona_pat_in_room += 1
                        wait = clock - (
                                patient.arrival_time + patient.service_time + patient.wait_in_reception_queue)
                        self.total_time_wait_in_room_q_corona += wait
                        patient.total_wait = patient.wait_in_reception_queue + wait
                        self.total_time_in_sys_corona_pats += (clock + service_rate - patient.arrival_time)

                    else:
                        doctor.cur_pat_type_corona = False
                        patient = self.normal_patients_queue.pop(0)
                        self.number_of_normal_pat_in_room += 1
                        wait = clock - (
                                patient.arrival_time + patient.service_time + patient.wait_in_reception_queue)
                        self.total_time_wait_in_room_q_normal += wait
                        patient.total_wait = patient.wait_in_reception_queue + wait
                        self.total_time_in_sys_normal_pats += (clock + service_rate - patient.arrival_time)

                    patient.respons_time += service_rate

        self.room_is_full = number_of_busy_doctors == len(self.Doctors)

        return number_of_pat_finished_check_up


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

class hospital:
    def __init__(self, M, number_of_doctors_per_room, mean_check_up_time, persons_corona, persons_normal, corona_totals):
        self.number_of_patients = len(persons_corona) + len(persons_normal)
        self.reception = Reception_class(perosons_corona, perosons_normal)
        self.number_of_doctors_per_room = number_of_doctors_per_room
        self.mean_check_up_time = mean_check_up_time
        self.Rooms = []
        self.time_reception_took = 0
        for i in range(M):
            self.Rooms.append(ROOM(i, self.number_of_doctors_per_room[i], self.mean_check_up_time[i]))

    def start_simulation(self):
        clock = 0
        left_reception_normal_pats = 0
        left_reception_corona_pats = 0
        while self.number_of_patients >= 0:

            left_reception_normal_pats_temp = 0
            left_reception_corona_pats_temp = 0

            time1 = time.time()
            if len(self.reception.corona_patient_queue) or len(
                    self.reception.normal_patient_queue) or self.reception.patient_in_reception:
                self.number_of_patients, left_reception_corona_pats_temp, left_reception_normal_pats_temp = self.reception.Reception(
                    clock, self.number_of_patients)

            # add number of pat left system in reception q
            left_reception_normal_pats += left_reception_normal_pats_temp
            left_reception_corona_pats += left_reception_corona_pats_temp

            time2 = time.time()
            self.time_reception_took += time2 - time1

            # send pat to shortest q
            while len(self.reception.Queue_to_room):
                qu_len = [
                    len(self.Rooms[i].corona_patients_queue) + len(self.Rooms[i].normal_patients_queue) + self.Rooms[
                        i].room_is_full for i in range(M)]
                index_min_qu = qu_len.index(min(qu_len))
                if self.reception.Queue_to_room[0].has_corona:
                    self.Rooms[index_min_qu].corona_patients_queue.append(self.reception.Queue_to_room.pop(0))
                else:
                    self.Rooms[index_min_qu].normal_patients_queue.append(self.reception.Queue_to_room.pop(0))

n = 10_000_000  # number of patinets
M = int(input("number of rooms: "))
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
