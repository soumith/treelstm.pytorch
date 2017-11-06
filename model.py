import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
import Constants
from torch.nn import Parameter

class Tree(object):
    def __init__(self, idx):
        self.children = []
        self.idx = idx

    def __repr__(self):
        if self.children:
            return '{0}: {1}'.format(self.idx, str(self.children))
        else:
            return str(self.idx)

tree = Tree(0)
tree.children.append(Tree(1))
tree.children.append(Tree(2))
tree.children.append(Tree(3))
tree.children[1].children.append(Tree(4))
print(tree)

class ChildSumLSTMCell(nn.Module):
    def __init__(self, hidden_size,
                 i2h_weight_initializer=None,
                 hs2h_weight_initializer=None,
                 hc2h_weight_initializer=None,
                 i2h_bias_initializer='zeros',
                 hs2h_bias_initializer='zeros',
                 hc2h_bias_initializer='zeros',
                 input_size=0):
        super(ChildSumLSTMCell, self).__init__()
        self._hidden_size = hidden_size
        self._input_size = input_size
        stdv = 1. / math.sqrt(input_size)
        self.i2h_weight = Parameter(torch.Tensor(4*hidden_size, input_size).uniform_(-stdv, stdv))
        self.i2h_bias = Parameter(torch.Tensor(4*hidden_size).uniform_(-stdv, stdv))
        stdv = 1. / math.sqrt(hidden_size)
        self.hs2h_weight = Parameter(torch.Tensor(3*hidden_size, hidden_size).uniform_(-stdv, stdv))
        self.hs2h_bias = Parameter(torch.Tensor(3*hidden_size).uniform_(-stdv, stdv))
        stdv = 1. / math.sqrt(hidden_size)
        self.hc2h_weight = Parameter(torch.randn(hidden_size, hidden_size).uniform_(-stdv, stdv))
        self.hc2h_bias = Parameter(torch.Tensor(hidden_size).uniform_(-stdv, stdv))

    def forward(self, inputs, tree):
        children_outputs = [self(inputs, child) for child in tree.children]
        if children_outputs:
            _, children_states = zip(*children_outputs) # unzip
        else:
            children_states = None

        return self.node_forward(inputs[tree.idx].unsqueeze(0),
                                 children_states,
                                 self.i2h_weight, self.hs2h_weight,
                                 self.hc2h_weight, self.i2h_bias,
                                 self.hs2h_bias, self.hc2h_bias)

    def node_forward(self, inputs, children_states,
                     i2h_weight, hs2h_weight, hc2h_weight,
                     i2h_bias, hs2h_bias, hc2h_bias):
        # comment notation:
        # N for batch size
        # C for hidden state dimensions
        # K for number of children.

        # FC for i, f, u, o gates (N, 4*C), from input to hidden
        i2h = F.linear(inputs, i2h_weight, i2h_bias)
        i2h_slices = torch.split(i2h, i2h.size(1) // 4, dim=1) # (N, C)*4
        i2h_iuo = torch.cat([i2h_slices[0], i2h_slices[2], i2h_slices[3]], dim=1) # (N, C*3)

        if children_states:
            # sum of children states, (N, C)
            hs = torch.sum(torch.cat([state[0].unsqueeze(0) for state in children_states]), dim=0)
            # concatenation of children hidden states, (N, K, C)
            hc = torch.cat([state[0].unsqueeze(1) for state in children_states], dim=1)
            # concatenation of children cell states, (N, K, C)
            cs = torch.cat([state[1].unsqueeze(1) for state in children_states], dim=1)
            # calculate activation for forget gate. addition in f_act is done with broadcast
            i2h_f_slice = i2h_slices[1]
            f_act = i2h_f_slice + hc2h_bias.unsqueeze(0).expand_as(i2h_f_slice) + torch.matmul(hc, hc2h_weight) # (N, K, C)
            forget_gates = F.sigmoid(f_act) # (N, K, C)
        else:
            # for leaf nodes, summation of children hidden states are zeros.
            # in > 0.2 you can use torch.zeros_like for this
            hs = Var(i2h_slices[0].data.new(*i2h_slices[0].size()).fill_(0))

        # FC for i, u, o gates, from summation of children states to hidden state
        hs2h_iuo = F.linear(hs, hs2h_weight, hs2h_bias)
        i2h_iuo = i2h_iuo + hs2h_iuo

        iuo_act_slices = torch.split(i2h_iuo, i2h_iuo.size(1) // 3, dim=1) # (N, C)*3
        i_act, u_act, o_act = iuo_act_slices[0], iuo_act_slices[1], iuo_act_slices[2] # (N, C) each

        # calculate gate outputs
        in_gate = F.sigmoid(i_act)
        in_transform = F.tanh(u_act)
        out_gate = F.sigmoid(o_act)

        # calculate cell state and hidden state
        next_c = in_gate * in_transform
        if children_states:
            next_c = torch.sum(forget_gates * cs, dim=1) + next_c
        next_h = out_gate * torch.tanh(next_c)

        return next_h, [next_h, next_c]


# module for distance-angle similarity
class Similarity(nn.Module):
    def __init__(self, sim_hidden_size, rnn_hidden_size, num_classes):
        super(Similarity, self).__init__()
        self.wh = nn.Linear(2*rnn_hidden_size, sim_hidden_size)
        self.wp = nn.Linear(sim_hidden_size, num_classes)

    def forward(self, F, lvec, rvec):
        # lvec and rvec will be tree_lstm cell states at roots
        mult_dist = lvec * rvec
        abs_dist = torch.abs(lvec - rvec)
        vec_dist = torch.cat([mult_dist, abs_dist], dim=1)
        out = F.log_softmax(self.wp(torch.sigmoid(self.wh(vec_dist))))
        return out


# putting the whole model together
class SimilarityTreeLSTM(nn.Module):
    def __init__(self, sim_hidden_size, rnn_hidden_size,
                 embed_in_size, embed_dim, num_classes):
        super(SimilarityTreeLSTM, self).__init__()
        self.embed = nn.Embedding(embed_in_size, embed_dim) 
        self.childsumtreelstm = ChildSumLSTMCell(rnn_hidden_size, input_size=embed_dim)
        self.similarity = Similarity(sim_hidden_size, rnn_hidden_size, num_classes)

    def forward(self, l_inputs, r_inputs, l_tree, r_tree):
        l_inputs = self.embed(l_inputs)
        r_inputs = self.embed(r_inputs)
        # get cell states at roots
        lstate = self.childsumtreelstm(l_inputs, l_tree)[1][1]
        rstate = self.childsumtreelstm(r_inputs, r_tree)[1][1]
        output = self.similarity(F, lstate, rstate)
        return output
