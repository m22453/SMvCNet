import copy
import torch
import random
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import torch.nn as nn

class RWDCL(torch.nn.Module):
    def __init__(self, layer_dims, temperature, n_classes, 
                 use_layer_norm,
                 mvrw_weights:list, 
                 n_heads=4,
                 n_neibs=5, 
                 n_rw:int=10, 
                 n_steps:int=5, 
                 drop_rate=0.5, 
                 knn4mvrw:bool=False,
                 ratio_nodes:float=1.0,
                 cosine=True):
        super(RWDCL, self).__init__()
        
        n_views = len(layer_dims)
        self.n_views = n_views
        self.n_classes = n_classes
        self.mvrw, self.weights = mvrw_weights
        self.n_neibs = n_neibs
        self.cosine = cosine
        self.n_rw = n_rw
        self.n_steps = n_steps
        self.knn4mvrw = knn4mvrw
        self.ratio_nodes = ratio_nodes

        self.online_encoder = nn.ModuleList([FCN(layer_dims[i], norm_layer=use_layer_norm, drop_out=drop_rate, norm_last_layer=True) for i in range(n_views)])
        self.target_encoder = copy.deepcopy(self.online_encoder)

        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data.copy_(param_o.data)  # initialize
            param_t.requires_grad = False  # not update by gradient

        
        # self.within_view_decoder = nn.ModuleList([FCN(layer_dims[i][::-1], drop_out=drop_rate, norm_last_layer=False) for i in range(n_views)])
        
        self.cross_view_decoder = nn.ModuleList([MLP(layer_dims[i][-1], layer_dims[i][-1]) for i in range(n_views)])


        self.feature_dim = [layer_dims[i][-1] for i in range(n_views)]

        assert len(set(self.feature_dim)) == 1

        self.multi_view_attention = MultiViewAttention(view_dim=self.feature_dim[0], hidden_dim=self.feature_dim[0]*4, num_heads=n_heads)


        # self.mse = torch.nn.MSELoss()
        self.cl = ContrastiveLoss(temperature)
        

        # self.cluster_head = ClusterHead(dim_in=self.feature_dim[0]*4, dim_out=n_classes)
        # self.ddcl = DDCLoss(n_classes)

    def forward(self, data, momentum, warm_up):

        gamma = 0

        self._update_target_branch(momentum)
        zs = [self.online_encoder[i](data[i]) for i in range(self.n_views)] # enc 
        # rs = [self.within_view_decoder[i](zs[i]) for i in range(self.n_views)] # rec
      
        ps = [self.cross_view_decoder[i](zs[i]) for i in range(self.n_views)] # trans

        zs_t = [self.target_encoder[i](data[i]) for i in range(self.n_views)]

        fvs, attention_weights = self.multi_view_attention(zs) 

        # pred = self.cluster_head(fvs)
                

        if warm_up:         
            mp = torch.eye(zs[0].shape[0]).cuda() 
            mp = [mp for _ in range(self.n_views)] 
            # mp = [self.knn_affinity(z) for z in zs]
        else:

            if self.mvrw:
                
                view_weights = attention_weights if self.weights else None
                mp = [self.multi_view_random_walk(zs_t, view_weights=view_weights) for i in range(self.n_views)]   
                gamma = 1 if self.weights else 0  # gamma learning

            else:
                mp = [self.kernel_affinity(zs_t[i]) for i in range(self.n_views)] # DIVIDE

                # mp = torch.eye(zs[0].shape[0]).cuda() # baseline
                # mp = [mp for _ in range(self.n_views)] 
                gamma = 0
            
        
        # l_rec = sum([self.mse(data[i], rs[i]) for i in range(self.n_views)])

        # intra view          

        l_intra = 0 
        for i in range(self.n_views):
            l_intra += self.cl(zs[i], zs_t[i], mp[i])


        # inter view 
        l_inter = 0
        for i in range(self.n_views):
            for j in range(self.n_views):
                if i == j:
                    continue
                l_inter += self.cl(ps[j], zs_t[i], mp[i])
            l_inter /= 2

        # fused view
        l_fused = 0
        for i in range(self.n_views):
            l_fused += self.cl(fvs, fvs, mp[i])
        

        # task loss
        # l_task = self.ddcl(pred, fvs) 

        # print(f'|l_rec = {l_rec}|')
        # print(f'|l_inter = {l_inter}|')
        # print(f'|l_intra = {l_intra}|')
        # print(f'|l_fused = {l_fused}|')
        # print(f'|l_task = {l_task}|')

        # loss = l_inter + l_intra if warm_up else l_inter + l_intra + l_fused * gamma
        loss = l_inter + l_intra + l_fused * gamma


        # loss = l_inter # cross-view
        # loss = l_intra # within-view 

        return loss
    
    
    def G_(self, z, temperature=0.1):
        z = F.normalize(z, p=2, dim=1)
        if self.cosine:
            G = 1 - torch.mm(z, z.t()).clamp(min=0.)
            G = torch.exp(-G / temperature)
        else:
            distance_matrix = torch.cdist(z, z, p=2)  
            G = torch.exp(-distance_matrix ** 2 / temperature)

        G = G / G.sum(dim=1, keepdim=True)
        return G

    # @torch.no_grad()
    # def knn_affinity(self, z,  k:int=5):
    #     G = self.G_(z)
    #     knn_G = G.clone()
    #     top_k, _ = torch.topk(knn_G, k, dim=1, largest=True, sorted=True)
    #     knn_mask = knn_G < top_k[:, -1].unsqueeze(1)
    #     knn_G.masked_fill_(knn_mask, 0)
    #     diag =  torch.diag(knn_G)
    #     knn_G = knn_G - torch.diag_embed(diag)  #  except self
    #     knn_G = knn_G / knn_G.sum(dim=1, keepdim=True)
    #     return knn_G

    @torch.no_grad()
    def knn_affinity(self, z):
        G = self.G_(z)
        num_nodes = G.size(0)

        k = min(self.n_neibs, num_nodes - 1)

        z_np = z.cpu().numpy()
    
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(z_np)
        _, indices = nbrs.kneighbors(z_np)
        
        knn_G = torch.zeros(num_nodes, num_nodes, device=z.device)
        
        for i in range(num_nodes):
            knn_G[i, indices[i, 1:]] = torch.tensor(G[i, indices[i, 1:]], device=z.device)
        
        knn_G = knn_G / knn_G.sum(dim=1, keepdim=True)
        return knn_G
    
    @torch.no_grad()
    def multi_view_knn(self, zs):
        knn_G = sum([self.knn_affinity(z) for z in zs])  # for view fusion
        knn_G = knn_G / knn_G.sum(dim=1, keepdim=True) 
        alpha = 0.5
        knn_G = torch.eye(knn_G.shape[0]).cuda() * alpha + knn_G * (1 - alpha)  # self-loop
        return knn_G

    
    # @torch.no_grad()
    # def multi_view_random_walk(self, zs, rw_nums:int=5, steps:int=10, use_knn:bool=True):
    #     def random_walk(Gs, start_node, steps):
    #         current_node = start_node
    #         path = [current_node]
    #         probabilities = [1.0]

    #         for s in range(steps):
    #             # Randomly select a graph for each step
    #             G = Gs[random.randint(0, len(Gs) - 1)]

    #             # Get the transition probabilities for the current node
    #             probabilities_at_node = G[current_node]

    #             # Sample the next node based on the probability distribution
    #             next_node = torch.multinomial(probabilities_at_node, 1).item()

    #             path.append(next_node)
    #             prob = probabilities_at_node[next_node].item()
    #             probabilities.append(prob)
    #             current_node = next_node

    #         return path, probabilities

    #     if use_knn:
    #         Gs = [self.knn_affinity(z) for z in zs]
    #     else:
    #         Gs = [self.G_(z) for z in zs]

    #     # Initialize an adjacency matrix with zeros
    #     rw_G = torch.zeros_like(Gs[0])

    #     for start_node in range(len(Gs[0])):
    #         for _ in range(rw_nums):
    #             path, prob = random_walk(Gs, start_node, steps)
    #             prob_ = [1.0]  # Start with probability 1 for the start node

    #             for i in range(1, len(path)):
    #                 prob_.append(prob_[i - 1] * prob[i])
    #                 rw_G[start_node][path[i]] += prob_[i]

    #     return rw_G

         
    @torch.no_grad()
    def  multi_view_random_walk(self, zs, view_weights=None):
        device_ = zs[0].device

        if self.knn4mvrw:
            Gs = [self.knn_affinity(z) for z in zs]
        else:
            Gs = [self.G_(z) for z in zs]

        # Initialize the multi-view knn fusion graph
        knn_G = self.multi_view_knn(zs)

        # Initialize an adjacency matrix with zeros
        rw_G = torch.zeros_like(Gs[0]).to(device_)
        num_all_nodes = len(Gs[0])

        for _ in range(self.n_rw):
            # Select random start nodes for all walks
            start_nodes = torch.arange(num_all_nodes).to(device_)

            if self.ratio_nodes < 1.0:
                start_nodes_indices = torch.randperm(num_all_nodes)[:round(num_all_nodes * self.ratio_nodes)]
                start_nodes = start_nodes[start_nodes_indices]

                ex_knn_G = torch.zeros_like(knn_G).to(device_)
                ex_knn_G[start_nodes_indices] = knn_G[start_nodes_indices]
                knn_G = ex_knn_G


                if view_weights is not None:
                    view_weights = view_weights[start_nodes_indices]

            
            num_nodes = len(start_nodes)
            # print(f'num_nodes = {num_nodes}') 

            # Initialize current nodes and probability matrix
            current_nodes = start_nodes.clone()
            probabilities = torch.ones(num_nodes, device=device_)

            for _ in range(self.n_steps):
                # Select a random graph for each step
                if view_weights is not None:
                    selected_view_indices = torch.multinomial(view_weights, 1).squeeze(-1)  # (n_samples,)
                    stacked_graphs = torch.stack([Gs[view_idx] for view_idx in selected_view_indices], dim=0)  # (n_samples, num_nodes, num_nodes)
                else:
                    # random
                    random_graphs = random.choices(Gs, k=num_nodes)
                    stacked_graphs = torch.stack(random_graphs).to(device_)

                transition_probs = stacked_graphs[torch.arange(num_nodes), current_nodes]
                
                next_nodes = torch.multinomial(transition_probs, 1).squeeze()

                # Update paths and probabilities
                probabilities *= transition_probs[torch.arange(num_nodes), next_nodes]
                rw_G[start_nodes, next_nodes] += probabilities

                current_nodes = next_nodes

        rw_G = rw_G / rw_G.sum(dim=1, keepdim=True)  # ensure probility
        alpha = 0.5
        rw_G =  knn_G* alpha + rw_G * (1 - alpha)
        return rw_G
        
    
    @torch.no_grad()
    def kernel_affinity(self, z, temperature=0.1, step: int = 5):
        z = L2norm(z)
        G = (2 - 2 * (z @ z.t())).clamp(min=0.)
        G = torch.exp(-G / temperature)
        G = G / G.sum(dim=1, keepdim=True)

        G = torch.matrix_power(G, step)
        alpha = 0.5
        G = torch.eye(G.shape[0]).cuda() * alpha + G * (1 - alpha)
        return G

    @torch.no_grad()
    def _update_target_branch(self, momentum):
        for i in range(self.n_views):
            for param_o, param_t in zip(self.online_encoder[i].parameters(), self.target_encoder[i].parameters()):
                param_t.data = param_t.data * momentum + param_o.data * (1 - momentum)

    @torch.no_grad()
    def extract_feature(self, data, mask):
        N = data[0].shape[0]
        z = [torch.zeros(N, self.feature_dim[i]).cuda() for i in range(self.n_views)]
        for i in range(self.n_views): 
            z[i][mask[:, i]] = self.target_encoder[i](data[i][mask[:, i]])

        for i in range(self.n_views):
            z[i][~mask[:, i]] = self.cross_view_decoder[1 - i](z[1 - i][~mask[:, i]])

        # z = [self.cross_view_decoder[i](z[i]) for i in range(self.n_views)]
        z = [L2norm(z[i]) for i in range(self.n_views)]

        return z



L2norm = nn.functional.normalize

class FCN(nn.Module):
    def __init__(self, dim_layer=None, norm_layer=True, act_layer=None, drop_out=0.0, norm_last_layer=False):
        super(FCN, self).__init__()
        act_layer = act_layer or nn.ReLU
        
        layers = []
        for i in range(1, len(dim_layer) - 1):
            layers.append(nn.Linear(dim_layer[i - 1], dim_layer[i]))
            if norm_layer:
                layers.append(nn.BatchNorm1d(dim_layer[i]))
            layers.append(act_layer())
            if drop_out != 0.0 and i != len(dim_layer) - 2:
                layers.append(nn.Dropout(drop_out))

        if norm_last_layer:
            layers.append(nn.Linear(dim_layer[-2], dim_layer[-1]))
            layers.append(nn.BatchNorm1d(dim_layer[-1], affine=False))
        else:
            layers.append(nn.Linear(dim_layer[-2], dim_layer[-1]))

        self.ffn = nn.Sequential(*layers)

    def forward(self, x):
        return self.ffn(x)
    

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out=None, hidden_ratio=4.0, act_layer=None):
        super(MLP, self).__init__()
        dim_out = dim_out or dim_in
        dim_hidden = int(dim_in * hidden_ratio)
        act_layer = act_layer or nn.ReLU
        self.mlp = nn.Sequential(nn.Linear(dim_in, dim_hidden),
                                 act_layer(),
                                 nn.Linear(dim_hidden, dim_out))

    def forward(self, x):
        x = self.mlp(x)
        return x
    
class ClusterHead(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ClusterHead, self).__init__()
        self.mlp = MLP(dim_in, dim_out)

    def forward(self, x):
        x = self.mlp(x)
        x = F.softmax(x, dim=-1)
        return x
    

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=1):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding dimension must be divisible by number of heads"
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value, mask=None):
        N = query.shape[0]
        Q = self.q_linear(query).view(N, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(N, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(N, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        energy = torch.matmul(Q, K.transpose(2, 3)) / self.head_dim**0.5
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))
        attention = F.softmax(energy, dim=3)
        x = torch.matmul(attention, V)
        
        x = x.transpose(1, 2).contiguous().view(N, -1, self.embed_dim)
        out = self.out(x)
        return out

class MultiViewAttention(nn.Module):
    def __init__(self, view_dim, hidden_dim, num_heads:int=0):
        super(MultiViewAttention, self).__init__()
        self.num_heads = num_heads
        self.fc1 = nn.Linear(view_dim, hidden_dim)

        if self.num_heads > 0:
            self.multihead_attn = MultiHeadAttention(hidden_dim, num_heads)

        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, zs):
        view_embeddings = torch.stack([self.fc1(z) for z in zs], dim=1)  # (n_samples, n_views, hidden_dim)
        
        if self.num_heads > 0:
            attn_output = self.multihead_attn(view_embeddings, view_embeddings, view_embeddings)  # (n_samples, n_views, hidden_dim)
        else:
            attn_output = view_embeddings

        attention_scores = self.fc2(torch.tanh(attn_output))  # (n_samples, n_views, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # (n_samples, n_views, 1)
        
        fused_view_embeddings = torch.sum(attention_weights * view_embeddings, dim=1)  # (n_samples, hidden_dim)

        return fused_view_embeddings, attention_weights.squeeze(-1)


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, x_q, x_k, mask_pos=None):
        x_q = L2norm(x_q)
        x_k = L2norm(x_k)
        N = x_q.shape[0]
        if mask_pos is None:
            mask_pos = torch.eye(N).cuda()
        similarity = torch.div(torch.matmul(x_q, x_k.T), self.temperature)
        similarity = -torch.log(torch.softmax(similarity, dim=1))
        nll_loss = similarity * mask_pos / mask_pos.sum(dim=1, keepdim=True)
        loss = nll_loss.mean()
        return loss


class DDCLoss(torch.nn.Module):

    def __init__(self, num_cluster, 
                 epsilon=1e-9, 
                 rel_sigma=0.15,
                 use_l2_flipped=False):
        """
        :param epsilon:
        :param rel_sigma: Gaussian kernel bandwidth
        """
        super(DDCLoss, self).__init__()
        self.epsilon = epsilon
        self.rel_sigma = rel_sigma
        # EAMC use l2_flipped
        self.use_l2_flipped = use_l2_flipped 
        self.num_cluster = num_cluster
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    def forward(self, logist, hidden):
        hidden_kernel = self._calc_hidden_kernel(hidden)
        l1_loss = self._l1_loss(logist, hidden_kernel, self.num_cluster)
        l2_loss = self._l2_flipped(logist) if self.use_l2_flipped else self._l2_loss(logist)
        l3_loss = self._l3_loss(logist, hidden_kernel, self.num_cluster)
        return l1_loss + l2_loss + l3_loss
    
    def _l1_loss(self, logist, hidden_kernel, num_cluster):
        return self._d_cs(logist, hidden_kernel, num_cluster)

    def _l2_loss(self, logist):
        n = logist.size(0)
        return 2 / (n * (n - 1)) * self._triu(logist @ torch.t(logist))
    
    def _l2_flipped(self, logist):
        return 2 / (self.num_cluster * (self.num_cluster - 1)) * self._triu(torch.t(logist) @ logist)

    def _l3_loss(self, logist, hidden_kernel, num_cluster):
        if not hasattr(self, 'eye'):
            self.eye = torch.eye(num_cluster, device=self.device)
        m = torch.exp(-self._cdist(logist, self.eye))
        return self._d_cs(m, hidden_kernel, num_cluster)

    def _triu(self, X):
        # Sum of strictly upper triangular part
        return torch.sum(torch.triu(X, diagonal=1))

    def _calc_hidden_kernel(self, x):
        return self._kernel_from_distance_matrix(self._cdist(x, x), self.epsilon)

    def _d_cs(self, A, K, n_clusters):
        """
        Cauchy-Schwarz divergence.
        :param A: Cluster assignment matrix
        :type A:  torch.Tensor
        :param K: Kernel matrix
        :type K: torch.Tensor
        :param n_clusters: Number of clusters
        :type n_clusters: int
        :return: CS-divergence
        :rtype: torch.Tensor
        """
        nom = torch.t(A) @ K @ A
        dnom_squared = torch.unsqueeze(torch.diagonal(nom), -1) @ torch.unsqueeze(torch.diagonal(nom), 0)

        nom = self._atleast_epsilon(nom, eps=self.epsilon)
        dnom_squared = self._atleast_epsilon(dnom_squared, eps=self.epsilon ** 2)

        d = 2 / (n_clusters * (n_clusters - 1)) * self._triu(nom / torch.sqrt(dnom_squared))
        return d

    def _atleast_epsilon(self, X, eps):
        """
        Ensure that all elements are >= `eps`.
        :param X: Input elements
        :type X: torch.Tensor
        :param eps: epsilon
        :type eps: float
        :return: New version of X where elements smaller than `eps` have been replaced with `eps`.
        :rtype: torch.Tensor
        """
        return torch.where(X < eps, X.new_tensor(eps), X)

    def _cdist(self, X, Y):
        """
        Pairwise distance between rows of X and rows of Y.
        :param X: First input matrix
        :type X: torch.Tensor
        :param Y: Second input matrix
        :type Y: torch.Tensor
        :return: Matrix containing pairwise distances between rows of X and rows of Y
        :rtype: torch.Tensor
        """
        xyT = X @ torch.t(Y)
        x2 = torch.sum(X ** 2, dim=1, keepdim=True)
        y2 = torch.sum(Y ** 2, dim=1, keepdim=True)
        d = x2 - 2 * xyT + torch.t(y2)
        return d

    def _kernel_from_distance_matrix(self, dist, min_sigma):
        """
        Compute a Gaussian kernel matrix from a distance matrix.
        :param dist: Disatance matrix
        :type dist: torch.Tensor
        :param min_sigma: Minimum value for sigma. For numerical stability.
        :type min_sigma: float
        :return: Kernel matrix
        :rtype: torch.Tensor
        """
        # `dist` can sometimes contain negative values due to floating point errors, so just set these to zero.
        dist = F.relu(dist)
        sigma2 = self.rel_sigma * torch.median(dist)
        # Disable gradient for sigma
        sigma2 = sigma2.detach()
        sigma2 = torch.where(sigma2 < min_sigma, sigma2.new_tensor(min_sigma), sigma2)
        k = torch.exp(- dist / (2 * sigma2))
        return k