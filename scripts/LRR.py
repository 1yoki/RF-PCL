import torch
import torch.nn as nn
from torch.autograd.functional import jacobian
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from match_cost import match_cost
from torch.utils.tensorboard import SummaryWriter
import os
from scipy.spatial import cKDTree
import sympy as sp
from sklearn.manifold import LocallyLinearEmbedding,Isomap
import pandas as pd 


def estimate_normals_curvature(points, k=20):
    """
    points: (N, 3)
    return:
        normals: (N, 3)
        curvature: (N, 1)   # PCA surface variation
    """
    tree = cKDTree(points)
    normals = np.zeros_like(points, dtype=np.float32)
    curvature = np.zeros((points.shape[0], 1), dtype=np.float32)

    center = points.mean(axis=0, keepdims=True)

    for i, p in enumerate(points):
        _, idx = tree.query(p, k=k)
        neigh = points[idx]                         # (k, 3)
        neigh_centered = neigh - neigh.mean(axis=0, keepdims=True)
        cov = neigh_centered.T @ neigh_centered / max(k - 1, 1)

        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        n = eigvecs[:, 0] 
        n = n / (np.linalg.norm(n) + 1e-8)

        if np.dot(n, p - center[0]) < 0:
            n = -n

        normals[i] = n
        curvature[i, 0] = eigvals[0] / (eigvals.sum() + 1e-8)

    return normals.astype(np.float32), curvature.astype(np.float32)

def chamfer_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    compute the symmetric Chamfer Distance between two torch.Tensor point clouds.

    parameters:
        x: shape = (N, 3)，torch.Tensor
        y: shape = (M, 3)，torch.Tensor

    return:
        float: Chamfer Distance
    """
    assert x.ndim == 2 and y.ndim == 2, "input tensors must be 2D (N, 3) and (M, 3)"
    assert x.shape[1] == y.shape[1], "point clouds must have the same number of dimensions (3)"

    dist = torch.cdist(x, y, p=2)

    cd_forward = torch.mean(torch.min(dist, dim=1).values)  # x → y
    cd_backward = torch.mean(torch.min(dist, dim=0).values) # y → x

    return (cd_forward + cd_backward).item()

def resample_point_cloud(pc: torch.Tensor, target_N: int) -> torch.Tensor:
    N, D = pc.shape
    if N == target_N:
        return pc
    elif N > target_N:
        idx = torch.randperm(N)[:target_N]
    else:
        # upsample by random repetition
        idx = torch.randint(0, N, (target_N,))
    return pc[idx]

def compute_emd_single_pair(x_tensor: torch.Tensor, recon_tensor: torch.Tensor) -> float:
    """
    compute approximate EMD loss（based on Sinkhorn）。

    input:
        x_tensor: torch.Tensor, shape = (N, 3)
        recon_tensor: torch.Tensor, shape = (N, 3)

    return:
        EMD distance（equal toevaluation_metrics.py）
    """
    x_sampled = resample_point_cloud(x_tensor, 4096)
    recon_sampled = resample_point_cloud(recon_tensor, 4096)

    x_sampled = x_sampled.unsqueeze(0)
    recon_sampled = recon_sampled.unsqueeze(0)

    emd = match_cost(x_sampled, recon_sampled)  # (1,)
    return (emd / 4096).item()

class DualDecoderAutoencoder(nn.Module):
    def __init__(self,N,dim):
        super(DualDecoderAutoencoder, self).__init__()
        self.n = N
        # encoder(f_map): from 3D input to 2D latent code
        self.encoder = nn.Sequential(
            nn.Linear(dim, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, N)
        )

        # decoderA(f_imm): from 2D latent code to 4D output (reshape to 2x2 metric)
        self.decoder_A = nn.Sequential(
            nn.Linear(N, 100),
            nn.SiLU(),
            nn.Linear(100, 100),
            nn.SiLU(),
            nn.Linear(100, dim)
        )

        # decoderB(f_iso): from 2D latent code to 3D output (reconstruct input)
        self.decoder_B = nn.Sequential(
            nn.Linear(N, 1000),
            nn.SiLU(),
            nn.Linear(1000, 1000),
            nn.SiLU(),
            nn.Linear(1000, dim)
        )

    def forward(self, x):
        # encode to latent code
        latent_code = self.encoder(x)  # two-dimensional latent code (N,)

        metric_output = self.decoder_A(latent_code)
        metric_2x2 = metric_output.view(-1)

        reconstructed = self.decoder_B(latent_code)

        return latent_code, metric_2x2, reconstructed


def rieman_metric(tensor_tuple):


    metric = torch.matmul(tensor_tuple, tensor_tuple.T)

    return metric.view(-1)

def normalize_point_cloud(point_cloud):
    mean = np.mean(point_cloud, axis=0)
    std = np.std(point_cloud, axis=0)
    point_cloud = point_cloud - np.expand_dims(np.mean(point_cloud, axis=0), 0)
    dist = np.max(np.sqrt(np.sum(point_cloud ** 2, axis=1)), 0)
    point_cloud = point_cloud / dist
    return point_cloud, mean, std

def normalize(matrix):
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    standardized_matrix = (matrix - mean) / std
    return standardized_matrix, mean, std
def compute_Rieman_matric_baojiao(Func,inputs):
    #print("inputs",inputs)
    #input()
    J_mat = jacobian(Func,inputs[0], create_graph=True).t()

    R_metric = rieman_metric(J_mat)
    R_metric_list = torch.unsqueeze(R_metric,dim=0)

    #ideal_dis = [[],[],[],[],[],[],[],[]]
    for tensor in inputs[1:]:
        J_mat = jacobian(Func,tensor, create_graph=True).t()
        R_metric = rieman_metric(J_mat)
        R_metric_list = torch.cat((R_metric_list, torch.unsqueeze(R_metric,dim=0)), dim=0)

    return R_metric_list

def compute_Rieman_matric(Func,inputs):
    #print("inputs",inputs)
    #input()
    J_mat = jacobian(Func,inputs[0], create_graph=True).t()
    R_metric = rieman_metric(J_mat)
    R_metric_list = torch.unsqueeze(R_metric,dim=0)
    #ideal_dis = [[],[],[],[],[],[],[],[]]
    for tensor in inputs[1:]:
        J_mat = jacobian(Func,tensor, create_graph=True).t()
        R_metric = rieman_metric(J_mat)
        R_metric_list = torch.cat((R_metric_list, torch.unsqueeze(R_metric,dim=0)), dim=0)
    return R_metric_list

class Autoencoder_1(nn.Module):
        def __init__(self,input):
            super(Autoencoder_1, self).__init__()
            # encoder
            self.encoder = nn.Sequential(
                nn.Linear(input, 16),
                nn.ReLU(), 
                nn.Linear(16, 5)
            )
            # decoder
            self.decoder = nn.Sequential(
                nn.Linear(5, 16),
                nn.ReLU(),
                nn.Linear(16, 3)
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded



class Autoencoder_2(nn.Module):
        def __init__(self,input):
            super(Autoencoder_2, self).__init__()
            # encoder
            self.encoder = nn.Sequential(
                nn.Linear(input, 1048),
                nn.ReLU(),
                nn.Linear(1048, 100)
            )
            # decoder
            self.decoder = nn.Sequential(
                nn.Linear(100, 1048),
                nn.ReLU(),
                nn.Linear(1048, 3)
            )

        def forward(self, x):
            encoded = self.encoder(x)  # encode
            decoded = self.decoder(encoded)  # decode
            return decoded


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch.optim as optim

def batch_log_spd(g_batch, eps=1e-12):
    """
    g_batch: (N, 2, 2)
    返回 log_g_batch: (N, 2, 2)
    """
    N = g_batch.shape[0]
    log_g = np.zeros_like(g_batch)

    for i in range(N):
        G = g_batch[i]

        # 2x2 SPD
        w, V = np.linalg.eigh(G)   # SPD -> eigh

        # avoid log(0/negative)
        w_clipped = np.clip(w, eps, None)
        log_w = np.log(w_clipped)

        # V @ diag(log_w) @ V^T
        log_G = V @ np.diag(log_w) @ V.T
        log_g[i] = log_G

    return log_g

def build_input_features(data, typ, alpha_rie=1.0):
    """
    [xyz(3), rie(4), normal(3), curv(1)]
    """
    coord = data[:, :3]
    rie = data[:, 3:7]
    normal = data[:, 7:10]
    curv = data[:, 10:11]

    if typ == "coord":
        X = coord
    elif typ == "rie":
        X = np.concatenate((coord, alpha_rie * rie), axis=1)
    elif typ == "coord_normal":
        X = np.concatenate((coord, normal), axis=1)
    elif typ == "coord_curv":
        X = np.concatenate((coord, curv), axis=1)
    elif typ == "coord_normal_curv":
        X = np.concatenate((coord, normal, curv), axis=1)
    else:
        raise ValueError(f"Unknown typ: {typ}")

    y = coord
    return X, y

def autoencoder_recover_forDraw(alpha,vali_train_ratio,typ,dataset,object,min_loss,autoencoder_typ=2,split_n = 10):
    split_dim_id = 1
    vali_train_ratio = 7

    loaded_data = np.loadtxt('./data_2048/eye_04_dataset.txt')
    data = torch.tensor(loaded_data,dtype=torch.float32)
    print("data",data.shape)

    min_value = data[:, split_dim_id].min().item()  # minvalue of the first split dimension
    max_value = data[:, split_dim_id].max().item()  # maxvalue of the first split dimension

    # compute the span of the split dimension
    span = max_value - min_value
    step_size = span / split_n

    # create bin indices for the split_dim_id dimension
    bin_indices = ((data[:, split_dim_id] - min_value) // step_size).long()

    bin_indices = torch.clamp(bin_indices, 0, split_n-1)

    # Dataset Partitioning
    if object == "head":
        train_mask = bin_indices < vali_train_ratio
        test_mask = bin_indices >= vali_train_ratio 
    elif object == "tail":
        train_mask = bin_indices >= 10-vali_train_ratio 
        test_mask = bin_indices < 10-vali_train_ratio 
    elif object == "mid":
        train_mask = (bin_indices < 3) | (bin_indices >= vali_train_ratio) 
        test_mask = (bin_indices >= 3) & (bin_indices < vali_train_ratio) 

    train_data = data[train_mask]
    test_data = data[test_mask]

    train_data = train_data.cpu().numpy() if isinstance(train_data, torch.Tensor) else train_data
    test_data = test_data.cpu().numpy() if isinstance(test_data, torch.Tensor) else test_data

    X_train, y_train = build_input_features(train_data, typ, alpha_rie=alpha)
    X_test, y_test = build_input_features(test_data, typ, alpha_rie=alpha)

    # init model
    input_dim = X_train.shape[1]

    if autoencoder_typ == 1:
        model_new = Autoencoder_1(input_dim)
    elif autoencoder_typ == 2:
        model_new = Autoencoder_2(input_dim)
    else:
        raise ValueError("Unsupported autoencoder_typ")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_new.parameters(), lr=0.001)

    # translate to torch tensors
    X_train_tensor = torch.Tensor(X_train)
    y_train_tensor = torch.Tensor(y_train)
    X_test_tensor = torch.Tensor(X_test)
    y_test_tensor = torch.Tensor(y_test)

    # get chamfer distance function
    cd_loss = chamfer_distance

    # train the autoencoder
    num_epochs = 450
    train_losses = []
    val_losses = []
    min_test_loss = 1
    min_test_output = None
    for epoch in range(num_epochs):
        model_new.train()

        # forward pass
        outputs_n = model_new(X_train_tensor)
        loss = criterion(outputs_n, y_train_tensor)

        # optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            outputs_n = model_new(X_train_tensor)
            #train_loss = mean_squared_error(y_train_tensor.numpy(), outputs_n.numpy())
            train_loss = cd_loss(y_train_tensor, outputs_n)
            train_losses.append(train_loss)
            test_outputs = model_new(X_test_tensor)
            test_loss = cd_loss(y_test_tensor, test_outputs)
            val_losses.append(test_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
            with torch.no_grad():
                test_outputs = model_new(X_test_tensor)
                test_loss = cd_loss(y_test_tensor, test_outputs)
                #test_loss = criterion(y_test_tensor, test_outputs)
                if min_test_loss>test_loss:
                    min_test_loss = test_loss
                    min_test_output = test_outputs
                print(f"Test MSE: {test_loss:.4f}")
    model_new.eval()
    # get train mse loss
    TRAIN_MSE_LOSS_ = None
    with torch.no_grad():
        train_outputs = model_new(X_train_tensor)

        train_loss = cd_loss(y_train_tensor, train_outputs)
        print(f"Train MSE: {train_loss:.4f}")
        TRAIN_MSE_LOSS_ = np.sqrt((y_train_tensor.numpy()-train_outputs.numpy())**2)
    # inference in validation set
    TEST_MSE_LOSS_ = None
    with torch.no_grad():
        test_outputs = model_new(X_test_tensor)
        test_loss = cd_loss(y_test_tensor, test_outputs)
        print(f"Test MSE: {test_loss:.4f}")
        TEST_MSE_LOSS_ = np.sqrt((y_test_tensor.numpy()-test_outputs.numpy())**2)
    reconstruct_eyes = np.concatenate((train_outputs.numpy(),test_outputs.numpy()))
    test_reconstruct_eyes = test_outputs.numpy()
    return np.mean(TRAIN_MSE_LOSS_,axis=1),np.mean(TEST_MSE_LOSS_,axis=1),train_mask,test_mask,min_test_output

x_train_raw = np.genfromtxt("./data_2048/eye_04.txt", dtype=np.float32)

# load the original point cloud data (you can replace this with your actual data loading code)
root = "./model_record/eye_04/"
device = torch.device("cuda")

x_train_raw, print_mean, print_std = normalize(x_train_raw)

model_n = DualDecoderAutoencoder(2, dim=3).to(device)
model_n.load_state_dict(torch.load(root + 'global_best_model.pth', weights_only=True))

latent_code, _, outputs = model_n(torch.tensor(x_train_raw, dtype=torch.float32).cuda())
rg = compute_Rieman_matric_baojiao(model_n.decoder_A, latent_code.detach())   # (N, 4)

# ===== normal / curvature =====
normals_np, curv_np = estimate_normals_curvature(x_train_raw, k=20)

normals = torch.tensor(normals_np, dtype=torch.float32, device=device)
curv = torch.tensor(curv_np, dtype=torch.float32, device=device)

# curvature standardization
curv = (curv - curv.mean(dim=0, keepdim=True)) / (curv.std(dim=0, keepdim=True) + 1e-8)

coords = torch.tensor(x_train_raw, dtype=torch.float32, device=device)

# save all features to txt for later use (e.g., LRR)
result = torch.cat((coords, rg, normals, curv), dim=1)   # [N, 11]
np.savetxt('./data_2048/eye_04_dataset.txt',
           result.cpu().detach().numpy(),
           fmt='%.6f')

print("------已保存all-feature数据集------")


rie_test1 = []
coor_test1 = []
rie_test2 = []
coor_test2 = []
min_rie_tloss = 1
min_coor_tloss = 1
ar = []


# ================= 1. random seed setup  =================
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


# ================= 2. path and setup =================
save_base_dir = "./result/eye_04"
save_pc_dir = os.path.join(save_base_dir, "avg_pointclouds")
log_dir = os.path.join(save_base_dir, "logs")  # TensorBoard 日志目录
os.makedirs(save_pc_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# init TensorBoard writer
writer = SummaryWriter(log_dir=log_dir)

num_experiments = 20
feature_modes = ["coord", "coord_normal", "coord_curv", "coord_normal_curv", "rie"]
error_records = []
pc_accumulators = {}
pc_counts = {}

# ================= 3. experiment loop =================
base_seed = 2025  # set a base seed for reproducibility; each experiment will use base_seed + run_idx as its seed

print(f"Start running {num_experiments} experiments...")

for run_idx in range(num_experiments):
    # every experiment uses a different seed, but the same seed will yield the same results if you rerun this script
    # ensuring both variability and reproducibility.
    current_seed = base_seed + run_idx
    setup_seed(current_seed)

    print(f"\n--- Experiment {run_idx + 1}/{num_experiments} (Seed: {current_seed}) ---")

    mean = torch.tensor(print_mean, device=device)
    std = torch.tensor(print_std, device=device)

    for mode in feature_modes:
        if mode == "rie":
            current_alpha_list = [0.1, 1, 2, 10, 0]
        else:
            current_alpha_list = [1.0]  # use alpha=1 for non-rie features

        for alpha in current_alpha_list:
            recon_out = torch.empty(0, device=device)

            for objects in ["head", "mid", "tail"]:
                _, _, _, _, output = autoencoder_recover_forDraw(
                    alpha=alpha,
                    vali_train_ratio=7,
                    typ=mode,
                    dataset="eye",
                    object=objects,
                    min_loss=1,
                    split_n=10
                )
                output = output.to(device)
                recon_out = torch.cat([recon_out, output], dim=0)

            x_train_tensor = torch.from_numpy(x_train_raw).to(device).type_as(recon_out)
            x_train_recovered = x_train_tensor * std + mean
            recon_recovered = recon_out * std + mean

            cd_val = chamfer_distance(x_train_recovered, recon_recovered)
            emd_val = compute_emd_single_pair(x_train_recovered, recon_recovered)

            error_records.append({
                "feature_mode": mode,
                "alpha": alpha,
                "chamfer_dist": cd_val,
                "emd": emd_val
            })

            # ================= TensorBoard visualization =================
            # 1. record scalars (Scalars)
            # you will see curves of each Alpha's error across experiments (run_idx)
            writer.add_scalar(f'Chamfer_Distance/{mode}_{alpha}', cd_val, global_step=run_idx)
            writer.add_scalar(f'EMD_Distance/{mode}_{alpha}', emd_val, global_step=run_idx)

            # 2. record point clouds (Point Clouds)
            # 3D point cloud visualization is basic but can give you a quick visual check of the reconstruction quality.
            pc_to_vis = recon_recovered.unsqueeze(0)  # 增加 batch 维度: [N, 3] -> [1, N, 3]

            writer.add_mesh(
                f'Reconstructed_PC/{mode}_{alpha}',
                vertices=pc_to_vis,
                global_step=run_idx,
                config_dict={"material": {"cls": "PointsMaterial", "size": 2}}
            )
            # ============================================================

            # sum up for average point cloud
            if alpha not in pc_accumulators:
                pc_accumulators[alpha] = torch.zeros_like(recon_recovered)
                pc_counts[alpha] = 0
            pc_accumulators[alpha] += recon_recovered
            pc_counts[alpha] += 1

# ================= 4. save results =================
writer.close()  # shutdown TensorBoard writer

# save average point clouds for each alpha
for alpha, total_tensor in pc_accumulators.items():
    if pc_counts[alpha] > 0:
        avg_tensor = total_tensor / pc_counts[alpha]
        file_path = os.path.join(save_pc_dir, f"alpha_{alpha}_avg_recon.txt")
        np.savetxt(file_path, avg_tensor.cpu().numpy(), fmt='%.6f')

# save error records to CSV
df = pd.DataFrame(error_records)
df.to_csv(os.path.join(save_base_dir, "experiment_errors.csv"), index=False)
print("All Done.")