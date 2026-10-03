import urllib.request
class BPETokenizer:
    #creates empty dictionary (self.merges) and tokenID map (self.vocab built by build_vocabulary)
    def __init__(self):
        self.merges = {}
        self.build_vocabulary()
    
    #pairs each token with the neighbor to its right and counts how often pair occurs
    @staticmethod
    def get_pair_counts (token_ids):
        counts = {}

        for first, second in zip(token_ids, token_ids[1:]):
            pair = (first,second)
            counts[pair] = counts.get(pair, 0) + 1

        return counts
    
    #builds new token list from old, checks current + next = pair, and appends new_token_id. otherwise, copies current
    @staticmethod
    def merge_pair(token_ids, pair, new_token_id):
        merged_ids = []
        position = 0

        while position < len(token_ids):
            pair_is_present = (
                position < len(token_ids) - 1
                and token_ids[position] == pair[0]
                and token_ids[position + 1] == pair[1]
            )

            if pair_is_present:
                merged_ids.append(new_token_id)
                position += 2
            else:
                merged_ids.append(token_ids[position])
                position += 1
        return merged_ids

    #converts training text to bytes, iterates up to number_of_merges, counts neighbouring pairs, and picks most frequent pair, assigns it next available ID. rewrites token list with pair merged, records new token in tokenID map.
    def train(self, text, number_of_merges):
        token_ids = list(text.encode("utf-8"))
        for merge_number in range(number_of_merges):
            pair_counts = self.get_pair_counts(token_ids)

            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            new_token_id = 256 + merge_number

            token_ids = self.merge_pair(token_ids, best_pair, new_token_id)
            self.vocab[new_token_id] = (
                self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            )
            self.merges[best_pair] = new_token_id      

    #converts text to bytes, counts pairs and chooses lowest ID (one learned earliest), merges chosen pair, repeating until no learned pairs remain. returns token IDs
    def encode(self,text):
        token_ids = list(text.encode("utf-8"))

        while len(token_ids) >= 2:
            pair_counts = self.get_pair_counts(token_ids)

            best_pair = min(
                pair_counts,
                key = lambda pair: self.merges.get(pair, float("inf"))
            )

            if best_pair not in self.merges:
                break

            token_ids = self.merge_pair(
                token_ids, best_pair, self.merges[best_pair]
            )
        return token_ids    

    #maps 256 byte values to single byte token, train adds merged tokens after 256
    def build_vocabulary(self):
        self.vocab = {
            byte_value: bytes([byte_value])
            for byte_value in range(256)
        }  

#Shakespeare from project gutenberg
url = "https://www.gutenberg.org/cache/epub/100/pg100.txt"
training_text = urllib.request.urlopen(url).read().decode("utf-8")[:200000]

tokenizer = BPETokenizer()
tokenizer.train(training_text, 300) 
print(len(tokenizer.merges))
print("Enter text to tokenize")
user_input = input("")
print(tokenizer.encode(user_input))
print([tokenizer.vocab[i] for i in tokenizer.encode(user_input)])