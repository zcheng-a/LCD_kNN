# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F




# Main implementation

class LearnCondDistn_kNN():

    def __init__(self, d_X , d_Y, data_tensor):
        self.d_X = d_X
        self.d_Y = d_Y
        self.data_tensor = data_tensor
        self.device = data_tensor.device
        self.indx_sorted_x = data_tensor.sort(0)[1]
        self.lb_x = torch.min(self.data_tensor[:,:d_X], 0)[0]
        self.ub_x = torch.max(self.data_tensor[:,:d_X], 0)[0]

    def init_net_std(self, n_atoms=8, n_layers=3, n_neurons=32,
                     input_actvn=nn.ReLU(), hidden_actvn=nn.ReLU()):
        ### initialize a neural network without Lipschitz constraint
        self.atomnet = AtomNetStd(
            self.d_X, self.d_Y, n_atoms=n_atoms,
            n_layers=n_layers, n_neurons=n_neurons,
            input_actvn=input_actvn, hidden_actvn=hidden_actvn
            )
        self.atomnet.to(self.device)
        self.lip_bool = False

    def init_net_lip(self, n_atoms=8, n_layers=3, n_neurons=32,
                     input_actvn=nn.ELU(), hidden_actvn=nn.ELU(),
                     L=1, L_requires_grad=False):
        ### initialize a neural network with adaptive Lipschitz continuity
        self.atomnet = AtomNetLip(
            self.d_X, self.d_Y,
            n_atoms=n_atoms, n_layers=n_layers, n_neurons=n_neurons,
            input_actvn=input_actvn, hidden_actvn=input_actvn,
            L=L, L_requires_grad=L_requires_grad
            )
        self.atomnet.to(self.device)
        self.lip_bool = True

    def set_compute_loss_param(self, k,
                               n_batch=256, n_bisect=5, n_part_batch=8,                                    ### for ANNS-RBSP, except that n_batch is for exact NNS
                               p_low=0.45, p_high=0.55, max_edge_ratio=3, ratio_skip=5,                    ### for ANNS-RBSP
                               n_iter_skh=1, one_over_eps=1, n_sparse=[1,4], gamma_sparse=0.9,             ### for Sinkhorn
                               nns_type='rbsp', bool_forloop_nns=False, bool_forloop_sparse=False):
        # set parameters for training
        self.k = k
        self.n_batch = n_batch
        self.n_bisect = n_bisect
        self.n_part_batch = n_part_batch
        self.p_low = p_low
        self.p_high = p_high
        self.max_edge_ratio = max_edge_ratio
        self.ratio_skip = ratio_skip
        self.n_iter_skh = n_iter_skh
        self.one_over_eps = one_over_eps
        self.n_sparse = n_sparse ### n_sparse should not have entries larger than k
        self.gamma_sparse = gamma_sparse
        self.nns_type = nns_type
        self.bool_forloop_nns = bool_forloop_nns
        self.bool_forloop_sparse = bool_forloop_sparse

    def compute_loss(self):
        if self.nns_type == 'rbsp':
            ret = compute_loss_sinkhorn_rbsp(
                self.atomnet, self.data_tensor, self.indx_sorted_x, self.lb_x, self.ub_x, self.k,
                n_bisect=self.n_bisect, n_part_batch=self.n_part_batch,
                p_low=self.p_low, p_high=self.p_high, max_edge_ratio=self.max_edge_ratio, ratio_skip=self.ratio_skip,
                n_iter_skh=self.n_iter_skh, one_over_eps=self.one_over_eps, n_sparse=self.n_sparse, gamma_sparse=self.gamma_sparse,
                device=self.device, bool_forloop_sparse=self.bool_forloop_sparse
                )
            return ret
        else:
            x_batch = self.lb_x + (self.ub_x - self.lb_x) * torch.rand(size=(self.n_batch, self.d_X), device=self.device)
            ret = compute_loss_sinkhorn_batch(
                self.atomnet, self.data_tensor, x_batch, self.k,
                n_iter_skh=self.n_iter_skh, one_over_eps=self.one_over_eps, n_sparse=self.n_sparse, gamma_sparse=self.gamma_sparse,
                device=self.device, bool_forloop_nns=self.bool_forloop_nns, bool_forloop_sparse =self.bool_forloop_sparse
                )
            return ret







# Neural nets

## Standard residual network with layerwise residual connection

class RLayer(nn.Module):
    ### R for Residual
    def __init__(self, n_in, n_out, actvn=nn.ReLU()):
        super(RLayer, self).__init__()
        self.bn = nn.BatchNorm1d(n_out)
        self.fc = nn.Linear(n_in, n_out)
        self.actvn = actvn

    def forward(self, x):
        out = self.actvn(self.bn(self.fc(x)))
        out = out + x
        return out

class AtomNetStd(nn.Module):
    ### return positions of atoms, represent atoms with uniform weights
    ### do not impose Lipschitz continuity
    ### input of forward should be a 2d matrix with size = (num of x's) * d_X
    ### forward returns 3d matrix with size = (num of x's) * n_atoms * d_Y
    def __init__(self, d_X=3, d_Y=2, n_atoms=8, n_layers=3, n_neurons=32, input_actvn=nn.ReLU(), hidden_actvn=nn.ReLU()):
        super(AtomNetStd, self).__init__()
        self.d_X = d_X
        self.d_Y = d_Y
        self.n_atoms = n_atoms
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.n_outlayer = d_Y * n_atoms
        self.input_fc = nn.Linear(d_X, n_neurons)
        self.input_bn = nn.BatchNorm1d(n_neurons)
        self.input_actvn = input_actvn
        self.hidden_layers = nn.ModuleList([RLayer(n_neurons, n_neurons, actvn=hidden_actvn) for _ in range(n_layers)])
        self.output_fc = nn.Linear(n_neurons, self.n_outlayer)

    def forward(self, x):
        x = self.input_actvn(self.input_fc(x)) ### bn cause nan at seed(0), also seems to downgrade performance
        # x = self.input_actvn(self.input_bn(self.input_fc(x)))
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_fc(x)
        x = x.reshape(-1, self.n_atoms, self.d_Y)
        return x




## Network for adaptive continuity

class CPLayer(nn.Module):
    ### CP for Convex Potential
    def __init__(self, n_in, n_out, actvn=nn.ELU()):
        super(CPLayer, self).__init__()
        self.u_nor = nn.Parameter(torch.ones(n_in).float(), requires_grad=False)
        self.h_nor = nn.Parameter(torch.tensor(1/n_in/n_out).float(), requires_grad=False)
        self.fc = nn.Linear(n_in, n_out)
        self.actvn = actvn

    def forward(self, x):
        out = F.linear(self.actvn(self.fc(x)), torch.transpose(self.fc.weight,0,1)) * self.h_nor
        out = x - out
        return out

    def update_uh_nor(self, mmt_uh=0.1):
        v = torch.matmul(self.fc.weight.data, self.u_nor.data)
        v = v / torch.sqrt(torch.sum(torch.square(v)))
        running_u_nor = torch.matmul(torch.transpose(self.fc.weight.data,-1,-2), v)
        running_u_nor = running_u_nor / torch.sqrt(torch.sum(torch.square(running_u_nor)))
        running_h_nor = 2 / ( torch.square(torch.dot(torch.matmul(self.fc.weight.data, running_u_nor), v)) + 1e-3)
        self.u_nor.data = (1-mmt_uh) * self.u_nor.data + mmt_uh * running_u_nor
        self.h_nor.data = (1-mmt_uh) * self.h_nor.data + mmt_uh * running_h_nor


class AtomNetLip(nn.Module):
    ### return positions of atoms, represent atoms with uniform weights
    ### impose adaptive Lipschitz continuity with convex potential layers and suitable normalization
    ### input of forward should be a 2d tensor with size = (num of x's) * d_X
    ### forward returns 3d tensor of size = (num of x's) * n_atoms * d_Y
    def __init__(self, d_X=3, d_Y=2, n_atoms=8, n_layers=3, n_neurons=32, L=1, input_actvn=nn.ELU(), hidden_actvn=nn.ELU(), L_requires_grad=False):
        super(AtomNetLip, self).__init__()
        self.d_X = d_X
        self.d_Y = d_Y
        self.n_atoms = n_atoms
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.n_outlayer = d_Y * n_atoms
        self.input_fc = nn.Linear(d_X, n_neurons)
        self.input_actvn = input_actvn
        self.input_nor = nn.Parameter(torch.ones(n_neurons), requires_grad=False)
        self.hidden_layers = nn.ModuleList([CPLayer(n_neurons, n_neurons, actvn=hidden_actvn) for _ in range(n_layers)])
        self.output_fc = nn.Linear(n_neurons, self.n_outlayer)
        self.output_nor = nn.Parameter(torch.ones(self.n_outlayer), requires_grad=False)
        self.L = nn.Parameter(torch.tensor(L).float(), requires_grad=L_requires_grad)


    def forward(self, x):
        x = self.input_actvn(self.input_fc(x)) * self.input_nor.view(1,-1) ### removing actvn seems to slightly downgrade performance
        # x = self.input_fc(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.L * self.output_fc(x) * self.output_nor.view(1,-1)
        x = x.reshape(-1, self.n_atoms, self.d_Y)
        return x


    def update_nor(self, mmt_inoutweight=0.1, mmt_mean=0.1, mmt_uh=0.1, p_update=0.1):
        if torch.rand(1) < p_update:
            # update input layer
            in_weight_row_l1 = self.input_fc.weight.data.abs().sum(-1)
            indx_in = in_weight_row_l1 > 1 / self.n_neurons
            if torch.sum(indx_in) > 0:
                self.input_nor[indx_in] = (1-mmt_inoutweight) + mmt_inoutweight / self.n_neurons / in_weight_row_l1[indx_in]
            # update hidden layer
            for layer in self.hidden_layers:
                layer.update_uh_nor(mmt_uh=mmt_uh)
            # update output layer
            out_weight_row_l1 = self.output_fc.weight.data.abs().sum(-1)
            indx_out = out_weight_row_l1 > 1 / self.d_Y
            if torch.sum(indx_out) > 0:
                self.output_nor[indx_out] = (1-mmt_inoutweight) + mmt_inoutweight / self.d_Y / out_weight_row_l1[indx_out]











# Sinkhorn algorithm

def my_Sinkhorn(cost_mat_batch, n_iter_skh=1, one_over_eps=1, n_sparse=[1,1], gamma_sparse=0.9, bool_forloop_sparse=False):
    ### cost_mat_batch should be of size n_batch * k * n_atoms
    ### one_over_eps = 1 / epsilon, note we also normalize the exponential indeces of K_batch
    ### the first entry of n_sparse imposes sparsity on transporting from empirical measure (obtained from data)
    ### the second entry of n_sparse imposes sparsity on transporting from approximating measure (obtained from net)
    ### sparsities imposed above are applied separately, resulting in two transport plans, which will be averaged latter using gamma_sparse to form a single plan
    ### gamma_sparse being 1 means all weight is allocated to the first plan
    ### compute W1 by multiplying plan and cost, note the transport plan is detached, thus the gradient graph only goes through cost
    ### return the average of W1's computed from different batches

    n_batch, k, n_atoms = cost_mat_batch.shape

    # prepare for iteration: K
    temp_nor_const = torch.amin( torch.amax(cost_mat_batch, -1), -1).view(n_batch, 1, 1)
    K_batch = torch.exp(- one_over_eps * cost_mat_batch / temp_nor_const).detach() # may apply skew on the exponential indexes, probably not useful

    # prepare for iteration: marginals
    marg_emp_batch, marg_approx_batch = torch.ones(n_batch, k, 1).to(device) / k, torch.ones(n_batch, n_atoms, 1).to(device) / n_atoms
    u_batch, v_batch = torch.ones(n_batch, k, 1).to(device), torch.ones(n_batch, n_atoms, 1).to(device)

    # excecute iteration
    for i in range(n_iter_skh):
        u_batch = marg_emp_batch / torch.matmul(K_batch, v_batch)
        v_batch = marg_approx_batch / torch.matmul(torch.transpose(K_batch, 1, 2), u_batch)
    tr_plan = (u_batch * K_batch * torch.transpose(v_batch,1,2))
    # loss_eval = torch.sum(tr_plan * cost_mat_batch).detach() / n_batch

    # impose sparsity on transport plan then compute loss
    if bool_forloop_sparse: ### in case of out-of-memory
        for _ in range(tr_plan.shape[0]):
            kth_from_data = -torch.kthvalue(-tr_plan[_], n_sparse[0], dim=-2, keepdim=True)[0]
            kth_from_net = -torch.kthvalue(-tr_plan[_], n_sparse[0], dim=-1, keepdim=True)[0]
            tr_plan[_] = gamma_sparse * tr_plan[_] * (tr_plan[_] >= kth_from_data) + (1-gamma_sparse) * tr_plan[_] * (tr_plan[_] >= kth_from_net)
    else:
        # topk_from_data = torch.topk(tr_plan, n_sparse[0], dim=-2)[0][:, (n_sparse[0]-1):n_sparse[0], :]
        # topk_from_net = torch.topk(tr_plan, n_sparse[1], dim=-1)[0][:, :, (n_sparse[1]-1):n_sparse[1]]
        kth_from_data = -torch.kthvalue(-tr_plan, n_sparse[0], dim=-2, keepdim=True)[0]
        kth_from_net = -torch.kthvalue(-tr_plan, n_sparse[0], dim=-1, keepdim=True)[0]
        tr_plan = gamma_sparse * tr_plan * (tr_plan >= kth_from_data) + (1-gamma_sparse) * tr_plan * (tr_plan >= kth_from_net)
    loss = torch.sum( tr_plan * cost_mat_batch ) / n_batch

    return loss








# Functions for loss computation with exact nearest neighbor search

def prepare_atoms_batch(atomnet, data_tensor, x_batch, k, weight=1, device=torch.device('cpu'), bool_forloop_nns=False):
    ### auxillary function for comput_loss_CDF below, use nearest neighborhood method
    ### atomnet should be an instance of AtomNet
    ### this function will group samples in a neighborhood of x, the neighborhood is specified as the set consisting of the closest k-th points
    ### then output the empirical measure formed by the corresponding y's from the samples, along with the approximating measure by evaluating atomnet at x
    ### output shape of mu_hat: (num of x) * k * d_Y, output shape of mu_approx  (num of x) * atomnet.n_atoms * d_Y

    if bool_forloop_nns:
        n_batch = x_batch.shape[0]
        mu_hat = torch.zeros(n_batch, k, atomnet.d_Y).to(device)
        temp = torch.zeros(data_tensor.shape[0], atomnet.d_X).to(device) # most memory-demanding part
        for i in range(n_batch):
            temp = torch.abs(data_tensor[:,:atomnet.d_X] - x_batch[i])
            sorted_indices = torch.sum(temp, -1).sort()[1]
            mu_hat[i] = data_tensor[sorted_indices[:k], atomnet.d_X:]
    else:
        distance_batch = torch.sum( torch.abs( torch.unsqueeze(data_tensor[:,:atomnet.d_X], 0) - torch.unsqueeze(x_batch, 1) ) , 2 )
        sorted_indices = distance_batch.sort()[1]
        mu_hat = data_tensor[sorted_indices[:,:k], atomnet.d_X:]

    mu_approx = atomnet(x_batch)

    return mu_hat, mu_approx



def compute_loss_sinkhorn_batch(atomnet, data_tensor, x_batch, k,                                ### basics
                                n_iter_skh=1, one_over_eps=1, n_sparse=[1,1], gamma_sparse=0.9,  ### for sinkhorn
                                device=torch.device('cpu'), bool_forloop_nns=False, bool_forloop_sparse=False):
    ### compute W1 loss between the empirical measure and the approximating measure from atomnet, using Sinkhorn iteration
    ### atomnet should be an instance of AtomNet
    ### one_over_eps = 1 / epsilon
    ### bool_forloop is for prepare_atoms_batch

    # prepare empirical measure and approximating measure
    mu_hat, mu_approx = prepare_atoms_batch(atomnet, data_tensor, x_batch, k=k, device=device, bool_forloop_nns=bool_forloop_nns)

    # Sinkhorn iteration
    n_batch = x_batch.shape[0]

    ## compute cost matrix
    mu_hat = torch.unsqueeze(mu_hat, -2).repeat(1, 1, atomnet.n_atoms, 1)
    mu_approx = torch.unsqueeze( mu_approx, -2).repeat(1, 1, k, 1)
    cost_mat_batch = torch.sum(torch.abs( (mu_hat - torch.transpose(mu_approx,1,2)) ), -1)  # shape = n_batch * k * n_atoms

    ## compute average of W1
    loss = my_Sinkhorn(cost_mat_batch, n_iter_skh, one_over_eps, n_sparse, gamma_sparse, bool_forloop_sparse=bool_forloop_sparse)

    return loss











# Functions for loss computation with approximate nearest neighbor search with random binary space partitioning (rbsp)

def bisect_index(n_bisect, x_data, indx_sorted_x, lb_x, ub_x, p_low=0.45, p_high=0.55, max_edge_ratio=5, device=torch.device('cpu')):
    ### n_bisect stand for the number of iteration, will produce 2**n_bisect partitions based on x_data
    ### x_data should be a tensor of size n_samples * d_X, where d_X is the dimension of space X
    ### indx_sorted_x is an torch integer tensor of size n_samples * d indicating the indeces for ascending order, it should resemble x_data.sort(0)[1]
    ### lb, ub should be torch tensors of length d_X
    ### return a triplet representing the partition
    ### first entry is boolean tensor of size (2**n_bisect) * n_samples, each row represents the x's that falls into that part
    ### second and third entries are tensor of size (2**n_bisect) * d_X, respetively indicating the lower and upper bounds of the covering rectangle

    # in case n_bisect too large
    while 2**n_bisect > x_data.shape[0]:
        n_bisect = int(n_bisect - 1)

    # initialize
    partition_bool = torch.zeros((2**n_bisect, x_data.shape[0]), dtype=torch.bool, device=device)
    partition_lb = torch.ones((2**n_bisect, x_data.shape[1]), device=device) * lb_x
    partition_ub = torch.ones((2**n_bisect, x_data.shape[1]), device=device) * ub_x
    partition_bool[0] = True
    dim_rand = torch.randint(0, x_data.shape[1], size=(2**n_bisect,))
    p_rand = p_low + (p_high-p_low) * torch.rand(size=(2**n_bisect,))

    # tiling
    for n_ in range(n_bisect):
        i_mid = int(2 ** n_)
        for i in range(i_mid):
            split_to =  int(i_mid + i) ### split to row (i_mid+i)
            p_, dim_ = p_rand[split_to], int(dim_rand[split_to])
            edge_len = partition_ub[i]-partition_lb[i]
            if torch.max(edge_len) >= max_edge_ratio * torch.min(edge_len):
                dim_ = torch.argmax(edge_len)
            j_mid = int(torch.sum(partition_bool[i]) * p_)
            temp = partition_bool[i][indx_sorted_x[:, dim_]]
            temp = torch.masked_select( indx_sorted_x[:, dim_], temp)
            ## split one rectangle into two for (partition_lb, partition_ub)
            partition_lb[split_to], partition_ub[split_to] = partition_lb[i], partition_ub[i]
            partition_ub[i, dim_], partition_lb[split_to, dim_] = x_data[temp[j_mid] , dim_], x_data[temp[j_mid-1] , dim_]
            ## split one rectangle into two for (partition_bool)
            partition_bool[split_to][temp[j_mid:]] = True
            partition_bool[i][temp[j_mid:]] = False

    return partition_bool, partition_lb, partition_ub


def prepare_atoms_batch_rbsp(atomnet, data_tensor, k, partition_bool, partition_lb, partition_ub, n_part_batch = 1, ratio_skip=3, device=torch.device('cpu')):
    ### auxillary function that prepares data for comput_loss_CDF below, use ANNS-RBSP
    ### atomnet should be an instance of AtomNet, e.g., P_hat or C_hat
    ### this function this function will first generate uniformly n_block_batch many x within each partition characterized by partition_lb and partition_ub
    ### then then group x_data in the neighborhood of x within the same partition characterized by partition_bool, the neighborhood is specified as the k-nearest neighbor
    ### finally output the empirical measure formed by the corresponding y's from the samples, along with the approximating measure by evaluating atomnet at x
    ### return x_batch (tensor of size n_x * d_X, where num of n_x = partition_bool.shape[0] * n_part_batch), mu_hat (tensor of size (num of x) * k * d_Y), and mu_net (tensor of size (num of x) * atomnet.n_atoms * d_Y)

    n_part, n_x = partition_bool.shape[0], partition_bool.shape[0] * n_part_batch
    partition_lb_interleave, partition_ub_interleave = torch.repeat_interleave(partition_lb, n_part_batch, dim=0), torch.repeat_interleave(partition_ub, n_part_batch, dim=0)
    x_batch_ = partition_lb_interleave + (partition_ub_interleave - partition_lb_interleave) * torch.rand(size=(n_x, atomnet.d_X), device=device)
    x_batch = torch.zeros_like(x_batch_)
    i_ = 0
    mu_hat = torch.zeros(n_x, k, atomnet.d_Y, device=device)

    for i in range(n_part):
        bool_array = partition_bool[i]
        if torch.sum(bool_array) < ratio_skip * k:
            # print(torch.sum(bool_array))
            continue
        data_part = data_tensor[bool_array]
        x_query_part = x_batch_[i*n_part_batch:(i+1)*n_part_batch] # query points
        distances_part = torch.sum(torch.abs( data_part[:, :atomnet.d_X].unsqueeze(0) - x_query_part.unsqueeze(1) ), -1)
        sorted_indx_part = distances_part.sort(-1)[1]
        mu_hat[i_*n_part_batch: (i_+1)*n_part_batch ] = data_part[sorted_indx_part[:,:k], atomnet.d_X:]
        x_batch[i_*n_part_batch: (i_+1)*n_part_batch ] = x_query_part
        i_ = i_ + 1

    x_batch, mu_hat = x_batch[:i_*n_part_batch], mu_hat[:i_*n_part_batch]
    mu_net = atomnet(x_batch)

    return x_batch, mu_hat, mu_net


def compute_loss_sinkhorn_rbsp(
        atomnet, data_tensor, indx_sorted_x, lb_x, ub_x, k,              ### basics
        n_bisect, n_part_batch=1,                                        ### for bisect_index
        p_low=0.45, p_high=0.55, max_edge_ratio=5, ratio_skip=10,        ### for bisect_index
        n_iter_skh=1, one_over_eps=1, n_sparse=[1,1], gamma_sparse=0.9,  ### for sinkhorn
        device=torch.device('cpu'), bool_forloop_sparse=False
        ):
    ### compute W1 distance between the empirical measure and the approximating measure from atomnet, using Sinkhorn iteration
    ### data preparation with rbsp, see function defined above
    ### atomnet should be an instance of AtomNet
    ### one_over_eps = 1 / epsilon

    # prepare empirical measure and approximating measure
    partition_bool, partition_lb, partition_ub = bisect_index(n_bisect, data_tensor[:, :atomnet.d_X], indx_sorted_x, lb_x, ub_x, p_low=p_low, p_high=p_high, max_edge_ratio=max_edge_ratio, device=device)
    x_batch, mu_hat, mu_net = prepare_atoms_batch_rbsp(atomnet, data_tensor, k, partition_bool, partition_lb, partition_ub, n_part_batch=n_part_batch, ratio_skip=ratio_skip, device=device)

    # Sinkhorn iteration
    n_batch = x_batch.shape[0]

    ## compute cost matrix
    mu_hat = torch.unsqueeze(mu_hat, -2).repeat(1, 1, atomnet.n_atoms, 1)
    mu_net = torch.unsqueeze( mu_net, -2).repeat(1, 1, k, 1)
    cost_mat_batch = torch.sum(torch.abs( (mu_hat - torch.transpose(mu_net,1,2)) ), -1)  # shape = n_batch * k * atomnet.n_atoms

    ## compute average of W1
    loss = my_Sinkhorn(cost_mat_batch, n_iter_skh, one_over_eps, n_sparse, gamma_sparse, bool_forloop_sparse=bool_forloop_sparse)

    return loss



