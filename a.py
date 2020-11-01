





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