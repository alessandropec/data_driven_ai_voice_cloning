import random
class RandomCycler:
    def __init__(self, items,sample_lenght=None):
        self.items = list(items)
        if self.items==None or len(self.items)==0:
            print("ERROR: cannote use an empty sequence for random cycler")
        self.seen_items = list()
        self.cut=len(self.items)%sample_lenght

      

    def __next__(self):     
        if len(self.seen_items) == len(self.items)-self.cut:
            self.seen_items = list()
        item = random.choice(list(set(self.items)-set(self.seen_items)))
  
        self.seen_items.append(item)
        return item

    def sample(self, n):
        """Returns a list of n items sampled from the cycler."""
        samples = []
        for _ in range(n):
            samples.append(next(self))
        return samples