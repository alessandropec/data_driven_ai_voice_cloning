from pathlib import Path
from .random_cycler import RandomCycler
from .audio_tools import get_sig


class Speaker:
    def __init__(self, root: Path,utterances_per_speaker:int,multi_rec_sessions=False,model_sample_rate=16000):
        self.root = root
        self.multi_rec_sessions=multi_rec_sessions
        self.file_name = root.name
        self.rec_sessions=[]
        if self.multi_rec_sessions: #VoxCeleb
            for s in self.root.glob("*"):
                self.rec_sessions.append(s)
        self.utterances_per_speaker=utterances_per_speaker
        self.utterances = None
        self.utterance_cycler = None    
        self.model_sample_rate=model_sample_rate 


    def _load_utterances(self):

        sources=[]
        if self.multi_rec_sessions:
            for s in self.rec_sessions:
                sources+= s.glob("*")
        else:
            sources=self.root.glob("*")

        self.utterances = [Utterance(f,self.model_sample_rate) for f in sources]
        self.utterance_cycler = RandomCycler(self.utterances,self.utterances_per_speaker)
               
    def random_partial(self, count):
        """
        Samples a batch of <count> unique utterances from the disk in a way that all 
        utterances come up at least once every two cycles and in a random order every time.
        
        :param count: The number of partial utterances to sample from the set of utterances from 
        that speaker. Utterances are guaranteed not to be repeated if <count> is not larger than 
        the number of utterances available.
        """
        if self.utterances is None:
            self._load_utterances()

        return self.utterance_cycler.sample(count)




class Utterance:
    def __init__(self, fpath,sample_rate):
        self.path = fpath
        self.sample_rate=sample_rate


    def get_sig(self):
        return get_sig(self.path,sr=self.sample_rate)