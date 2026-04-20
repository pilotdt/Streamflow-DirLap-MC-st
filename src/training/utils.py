import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StandardScaler:
    def __init__(self):
        self.mean = None  
        self.std = None   

    def fit(self, X):
        if X.ndim == 2:
            X = X[..., None]  

        assert X.ndim == 3, f"Expected (T, N, F), got {X.shape}"

        self.mean = X.mean(axis=0)  
        self.std  = X.std(axis=0)   
        self.std[self.std == 0] = 1e-6

    def transform(self, X):
        if X.ndim == 2:
            X = X[..., None]

        return (X - self.mean[None, :, :]) / self.std[None, :, :]

    def inverse_transform(self, X):
        if X.ndim == 3:
            X = X[..., None]

        return X * self.std[None, :, :] + self.mean[None, :, :]


class MinMaxScaler:
    def __init__(self, eps=1e-6):
        self.min = None
        self.max = None
        self.eps = eps

    def fit(self, X):
        if X.ndim == 2:
            X = X[..., None] 
        assert X.ndim == 3, f"Expected (T, N, F), got {X.shape}"

        self.min = X.min(axis=0) 
        self.max = X.max(axis=0) 
        self.range = self.max - self.min
        self.range[self.range == 0] = self.eps

    def transform(self, X):
        if X.ndim == 2:
            X = X[..., None]  
        return (X - self.min[None, :, :]) / self.range[None, :, :]

    def inverse_transform(self, X):
        if X.ndim == 3:
            X = X[..., None]
        return X * self.range[None, :, :] + self.min[None, :, :]


def build_advection_operator(A, add_storage=False, learn_stor=None):
    if A.is_sparse:
        A_dense = A.to_dense()
    else:
        A_dense = A
    A_dense = A_dense.to(device)

    # Compute D_out (in case of directed A its D)
    d_out = torch.sum(A_dense, dim=1)
    
    # Apply station-specific learn_stor to the diagonal
    if add_storage and learn_stor is not None:
        storage_phys = learn_stor
        d_total = d_out + storage_phys
    else:
        d_total = d_out

    # L = D_total - A
    L_dir = torch.diag(d_total) - A_dense

    return L_dir.to_sparse().coalesce()


def dir_laplacian_regularizer(preds, L_sparse):

    """
    preds: (B, H, N)
    L_sparse: (N, N) sparse Laplacian
    """
    preds = preds.squeeze(-1)
    B, H, N = preds.shape

    y_flat = preds.reshape(B * H, N)       
    y = y_flat
    yT = y_flat.transpose(0, 1)            

    Ly = torch.sparse.mm(L_sparse, yT)

    Ly = Ly.t().reshape(B, H, N)
    reg = Ly
    reg = torch.pow(Ly, 2).mean()
    return reg
