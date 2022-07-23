'''
Free Association
'''
import os
from cmd import Cmd
import numpy as np
from gensim.models import KeyedVectors


wordvectors_source_path = os.path.expanduser("~/.local/share/yaspace/sgns.wiki.word.top50000")
wordvectors_path = os.path.expanduser("~/.local/share/yaspace/sgns.wiki.word.top50000.kv")


def prepare_vector(source, dest):
    word_vectors = KeyedVectors.load_word2vec_format(source, binary=False)
    word_vectors.save(dest)


class Agent:

    def __init__(self, wordvectors_path):
        self.wv = KeyedVectors.load(wordvectors_path, mmap='r')
        self.random = np.random.RandomState()

    def associate(self, word):
        hop_cnt = self.random.randint(1, 3)
        current = word
        for _ in range(hop_cnt):
            current = self.associate_hop(word)
        return current

    def associate_hop(self, word):
        try:
            result = self.wv.similar_by_word(word)
        except KeyError:
            return self.random.choice(self.wv.index_to_key)
        idx = self.random.randint(0, len(result))
        return result[idx][0]


class AgentShell(Cmd):
    prompt = 'Player> '

    def __init__(self, agent):
        super(AgentShell, self).__init__()
        self.agent = agent
        self.history = []

    def do_exit(self, inp):
        print("Bye")
        return True

    def default(self, inp):
        if inp == 'x' or inp == 'q':
            return self.do_exit(inp)
        recv = inp.strip()
        self.history.append(('recv', recv))

        reply = self.agent.associate(recv)
        print("Agent> {}".format(reply))
        self.history.append(('send', reply))


if __name__ == "__main__":
    shell = AgentShell(Agent(wordvectors_path))
    shell.cmdloop("Let's Play Word Ball !")