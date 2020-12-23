import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        print(f'CTCLabelConverter.__init__(character:{character})')
        # character (str): set of the possible characters.
        dict_character = list(character)    # 문자열을 문자리스트 로
        print(f'dict_character: {dict_character}')

        self.dict = {}
        # IndexToWord
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1
        print(f'self.dict: {self.dict}')
        self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)
        print(f'self.character: {self.character}')

    def encode(self, text, batch_max_length=25):
        # print(f'CTCLabelConverter.encode(text:{text}, batch_max_length:{batch_max_length})') # batch_max_length:25
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text: text index for CTCLoss. [batch_size, batch_max_length]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]

        # The index used for padding (=0) would not affect the CTC loss calculation.
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text = [self.dict[char] for char in text]
            batch_text[i][:len(text)] = torch.LongTensor(text)
        # print(f'batch_text.shape: {batch_text.shape}') # [64, 192]
        # print(f'length: {length}')  # [44, 44, 44, 44, 46, 47, 42, 36, 49, 39, 41, 40, 51, 37, 44, 44, 44, 44, 43, 34, 38, 50, 41, 40, 45, 42, 38, 43, 39, 45, 47, 42, 52, 45, 41, 44, 39, 46, 42, 48, 42, 46, 43, 50, 43, 38, 53, 43, 42, 43, 41, 43, 43, 45, 45, 41, 49, 45, 48, 46, 43, 44, 45, 49]
        # [1, 4, 3, 6, 3, 4, 3, 3, 4, 2, 4, 2, 4, 3, 1, 2, 3, 2, 4, 2, 3, 2, 3, 2, 4, 3, 6, 3, 3, 6, 3, 3, 3, 2, 2, 2, 3, 4, 2, 2, 4, 2, 3, 2, 3, 5, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 3, 4, 2, 2]
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        print(f'CTCLabelConverter.decode(text_index:{text_index}, length:{length})')
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            t = text_index[index, :]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
        print(f'texts: {texts}')
        return texts


class CTCLabelConverterForBaiduWarpctc(object):
    """ Convert between text-label and text-index for baidu warpctc """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        text = ''.join(text)
        text = [self.dict[char] for char in text]

        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        index = 0
        for l in length:
            t = text_index[index:index + l]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
            index += l
        return texts


class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        print(f'AttnLabelConverter().__init__()')
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        print(f'self.character:{self.character}')
        print(f'self.character.len:{len(self.character)}')

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i
        
        print(f'self.dict:{self.dict}')
        print(f'self.dict.len:{len(self.dict)}')

    def encode(self, text, batch_max_length=25):
        print(f'AttnLabelConverter().__init__()')

        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """

        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)

        print(f'length:{length}')
        print(f'batch_max_length:{batch_max_length}')
        print(f'batch_text:{batch_text}')

        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts


'''Loss 평균 계산 클래'''
class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        print(f'Averager.__init__()')
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res
