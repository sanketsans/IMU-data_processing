import math

class Helpers:
    def __init__(self):
        self.value = 0

    def floor(self, value):
        return math.floor(value*100)/100.0

    def get_sample_rate(self, samples):
        total_sample = 0.0
        not_consistent = 0
        curr_bin = math.floor(samples[0])
        count = 0
        sample_rate = {}
        not_cons_sample_rate = {}
        for sample in samples:
            total_sample += sample - total_sample
            if total_sample > float(curr_bin)+0.99:
                sample_rate[curr_bin] = count
                # if (count != 100):
                #     not_consistent += 1
                #     not_cons_sample_rate[curr_bin] = count
                curr_bin = math.floor(total_sample)
                count = 0
            count += 1

        sample_rate[curr_bin] = count
        # if (count != 100):
        #     not_consistent += 1
        #     not_cons_sample_rate[curr_bin] = count

        return sample_rate ##if you want all the sample rates.
        # return not_cons_sample_rate, not_consistent

    def get_average_remove_dup(self, samples, avg_ind, rm_ind):
        samples[avg_ind] = (samples[avg_ind] + samples[rm_ind]) / 2.0
        samples.pop(len(samples) - abs(rm_ind))


if __name__ == "__main__":
    utils = Helpers()
