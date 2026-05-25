import torch
import torch.nn as nn
from torch.autograd.functional import jacobian
import math
import copy
import heapq
import os
device = torch.device("cuda")
import numpy as np
from scipy.sparse import coo_matrix
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.func import jacrev, vmap

def calculate_zuhe(i,ty="map"):
    if ty=="map":
        count = 0
        for m in range(i):
            for n in range(m + 1, i):
                count+=1
        return count
    else:
        return 1

def normalize(matrix):
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    standardized_matrix = (matrix - mean) / std
    return standardized_matrix, mean, std

def NearestPoint(coords,k):
    f_n = coords.shape[1]
    dists = torch.cdist(coords, coords)

    # Set diagonal entries, i.e. self-distances, to infinity to exclude each point itself
    dists.fill_diagonal_(float('inf'))

    # Find the k nearest indices in each distance row
    _, indices = torch.topk(dists, k, largest=False)

    # Extract the corresponding nearest-neighbor coordinates
    nearest_neighbors = coords[indices]

    # Reshape nearest-neighbor coordinates to N*k by feature_dim
    nearest_neighbors = nearest_neighbors.view(-1, f_n)
    return nearest_neighbors,indices


def nearPointEucliAngle(A, k):
    A = torch.tensor(A, dtype=torch.float32)
    EculiAngle = []
    EcuDistance = []
    nearest_neighbors, indices = NearestPoint(A, k)  # knn
    for i in range(A.shape[0]):
        point = A[i]  # Current point
        neighbors = nearest_neighbors[i * k:(i + 1) * k]  # Select neighboring points
        vectors_euclidean = neighbors - point  # Neighborhood vectors

        for j in range(vectors_euclidean.shape[0]):
            # Compute the inner product of the same vector in the original space
            inner_product_euclidean = torch.dot(vectors_euclidean[j], vectors_euclidean[j])
            cos_theta_euclidean = inner_product_euclidean  # / (norm_m * norm_n)
            EcuDistance.append(cos_theta_euclidean)

        for m in range(vectors_euclidean.shape[0]):
            for n in range(m + 1, vectors_euclidean.shape[0]):
                # Compute the inner product between different vectors in the original space
                inner_product_euclidean = torch.dot(vectors_euclidean[m], vectors_euclidean[n])
                cos_theta_euclidean = inner_product_euclidean  # / (norm_m * norm_n)
                EculiAngle.append(cos_theta_euclidean)
    return torch.stack(EculiAngle).cuda(), torch.stack(EcuDistance).cuda(), indices


class DualDecoderAutoencoder(nn.Module):
    def __init__(self, N, dim):
        super(DualDecoderAutoencoder, self).__init__()
        self.n = N
        # Encoder: encode input features into a latent code
        self.encoder = nn.Sequential(
            nn.Linear(dim, 1000),  # Input features
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, N)  # Output latent code
        )

        # Decoder A: decode the latent code into metric-related features for a 2x2 metric matrix
        self.decoder_A = nn.Sequential(
            nn.Linear(N, 100),
            nn.SiLU(),
            nn.Linear(100, 100),
            nn.SiLU(),
            nn.Linear(100, dim)  # Output metric-related features
        )

        # Decoder B: decode the latent code into reconstructed input features
        self.decoder_B = nn.Sequential(
            nn.Linear(N, 1000),
            nn.SiLU(),
            nn.Linear(1000, 1000),
            nn.SiLU(),
            nn.Linear(1000, dim)  # Output reconstructed features
        )

    def forward(self, x):
        # Encode
        latent_code = self.encoder(x)  # Latent code

        # Decoder A: decode metric-related features and flatten them
        metric_output = self.decoder_A(latent_code)
        metric_2x2 = metric_output.view(-1)  # Flatten the metric output

        # Decoder B: decode reconstructed input features
        reconstructed = self.decoder_B(latent_code)

        return latent_code, metric_2x2, reconstructed


def rieman_metric(tensor_tuple):
    """a =torch.dot(tensor_tuple[0], tensor_tuple[0])
    b =torch.dot(tensor_tuple[0], tensor_tuple[1])
    c =torch.dot(tensor_tuple[1], tensor_tuple[0])
    d =torch.dot(tensor_tuple[1], tensor_tuple[1])


    result = torch.cat((torch.unsqueeze(a,dim=0),torch.unsqueeze(b,dim=0),torch.unsqueeze(c,dim=0),torch.unsqueeze(d,dim=0)))"""

    metric = torch.matmul(tensor_tuple, tensor_tuple.T)

    return metric

    result = torch.zeros(4, requires_grad=True)
    count = 0

    for column1 in zip(*tensor_tuple):
        for column2 in zip(*tensor_tuple):
            column_result = torch.dot(torch.tensor(column1), torch.tensor(column2))
            result = result.clone()  # Create a new tensor to avoid in-place operations
            result[count] = column_result
            count += 1
    # Return a 2x2 matrix
    return result.view(2, 2)

def compute_Rieman_matric_baojiao(Func,inputs,gap):
    # Compute the Riemannian metric
    J_mat = jacobian(Func,inputs[0], create_graph=True).t()  # Jacobian matrix

    R_metric = rieman_metric(J_mat)
    R_metric_list = torch.unsqueeze(R_metric,dim=0)

    for tensor in inputs[1:]:
        J_mat = jacobian(Func,tensor, create_graph=True).t()  # Jacobian matrix
        R_metric = rieman_metric(J_mat)
        R_metric_list = torch.cat((R_metric_list, torch.unsqueeze(R_metric,dim=0)), dim=0)

    return R_metric_list

def compute_Rieman_metric_vmap(decoder_A, Z, chunk=256):
    """
    decoder_A: nn.Module, z:[d]->x:[D]
    Z: [B, d]
    return: [B, d, d]
    """
    Z = Z.detach().requires_grad_(True)

    def f(z):
        return decoder_A(z)  # [D]

    J_single = jacrev(f)     # Single-point Jacobian: [D, d]

    outs = []
    for i in range(0, Z.shape[0], chunk):
        zc = Z[i:i+chunk]
        J = vmap(J_single)(zc)                 # [b, D, d]
        G = J.transpose(-1, -2) @ J            # [b, d, d]
        outs.append(G)
    return torch.cat(outs, dim=0)

def conformalMap_new(x_train, A, Rg,
                 nj_mean, nj_std, ds_mean, ds_std,
                 gap, nearest_neighbors,
                 E_nearest_neighbors, D_nearest_neighbors,
                 k, single_count, ty="map"):
    """
    A:                 [B, 2]  (center-point latents in the batch)
    Rg:                [B, 2, 2] (metric matrix for each point)
    nearest_neighbors: [B*k, 2]  (k neighbor latents for each center point, flattened by batch)
    E_nearest_neighbors: [B*single_count] (flattened m<n pair inner-product table for each point)
    D_nearest_neighbors: [B*k] (flattened j=j self inner-product/distance table for each point)
    """

    device = A.device
    dtype = A.dtype
    B = A.shape[0]

    # ---- 1) Reshape neighbor latents: [B, k, 2] ----
    neighbors = nearest_neighbors.view(B, k, -1)  # [B,k,2]
    vectors = neighbors - A[:, None, :]           # [B,k,2]

    # ---- 2) Compute Gram: Gamma = V G V^T -> [B,k,k] ----
    # V: [B,k,2], G: [B,2,2]
    VG = torch.matmul(vectors, Rg)                        # [B,k,2]
    gram = torch.matmul(VG, vectors.transpose(-1, -2))    # [B,k,k]

    # ---- 3) Take upper-triangular values (m<n) and align them with E_nearest_neighbors ----
    # In theory, single_count should equal k*(k-1)//2
    iu, iv = torch.triu_indices(k, k, offset=1, device=device)
    gram_pairs = gram[:, iu, iv]                          # [B, single_count]

    E_pairs = E_nearest_neighbors.view(B, single_count)   # [B, single_count]

    # run_norm(theta, mean, std)  -> (x-mean)/std  :contentReference[oaicite:2]{index=2}
    nj_mean_t = torch.as_tensor(nj_mean, device=device, dtype=dtype)
    nj_std_t  = torch.as_tensor(nj_std,  device=device, dtype=dtype)
    theta_pairs = (gram_pairs - nj_mean_t) / nj_std_t

    # The original version applies MSELoss(scalar, scalar) to each pair and then averages
    # Equivalent to mean((a-b)^2)
    loss_map = (theta_pairs - E_pairs).pow(2).mean()

    # ---- 4) dist branch: take the diagonal and align it with D_nearest_neighbors ----
    if ty == "map":
        loss_ds = torch.zeros((), device=device, dtype=dtype)
        return loss_map, loss_ds
    else:
        diag = gram.diagonal(dim1=-2, dim2=-1)            # [B,k]
        D_list = D_nearest_neighbors.view(B, k)           # [B,k]

        ds_mean_t = torch.as_tensor(ds_mean, device=device, dtype=dtype)
        ds_std_t  = torch.as_tensor(ds_std,  device=device, dtype=dtype)
        theta_diag = (diag - ds_mean_t) / ds_std_t

        loss_ds = (theta_diag - D_list).pow(2).mean()

        # Preserve the original return order for ty=="dist": return mean(loss_ds), mean(loss)
        # :contentReference[oaicite:3]{index=3}
        return loss_ds, loss_map


def riemannian_inner_product(u, v, g):
    return torch.matmul(u, torch.matmul(g, v))

# Compute the angle between two vectors under the Riemannian metric
def angle_between_vectors(u, v, g):
    inner_product = riemannian_inner_product(u, v, g)
    norm_u = torch.sqrt(riemannian_inner_product(u, u, g))
    norm_v = torch.sqrt(riemannian_inner_product(v, v, g))
    cos_theta = inner_product # inner_product / (norm_u * norm_v)
    """if torch.isnan(cos_theta).any():
        cos_theta = inner_product / 1e-8
    # Prevent numerical errors by clamping cos_theta to [-1, 1]
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)"""
    #theta = torch.acos(cos_theta)
    return cos_theta

def run_norm(matrix,mean,std):
    standardized_matrix = (matrix - mean) / std
    return standardized_matrix

def conformalMap(x_train,A,Rg,nj_mean,nj_std,ds_mean,ds_std,gap,nearest_neighbors,E_nearest_neighbors,D_nearest_neighbors,k,single_count,ty="map"):
    offsets = torch.tensor([[0, gap], [0, -gap], [gap, 0], [-gap, 0], [gap, gap], [gap, -gap], [-gap, gap], [-gap, -gap]], dtype=torch.float32).cuda()
    # Compute the loss function
    loss = torch.tensor(0.0, requires_grad=True)
    loss_ds = torch.tensor(0.0, requires_grad=True)
    l1_loss = nn.MSELoss()
    # Iterate over each point in tensor A
    for i in range(A.shape[0]):
        point = A[i]  # Current point
        g = Rg[i]
        neighbors = nearest_neighbors[i*k : (i+1)*k]
        vectors_manifold = neighbors - point  # kx2
        single_neiji_list = E_nearest_neighbors[i*single_count:(i+1)*single_count]
        single_dist_list = D_nearest_neighbors[i*k : (i+1)*k]
        count = 0
        for m in range(vectors_manifold.shape[0]):
            for n in range(m + 1, vectors_manifold.shape[0]):
                # Compute the inner product on the Riemannian manifold
                theta_manifold = angle_between_vectors(vectors_manifold[m], vectors_manifold[n], g)
                inner_product_euclidean = single_neiji_list[count]
                theta_manifold = run_norm(theta_manifold,nj_mean,nj_std)
                cos_theta_euclidean = inner_product_euclidean #inner_product_euclidean / (norm_m * norm_n)
                count += 1
                if i == 0 and m ==0 and n == m + 1:
                    loss = torch.unsqueeze(l1_loss(theta_manifold , cos_theta_euclidean),dim=0)
                else:
                    loss = torch.cat((loss, torch.unsqueeze(l1_loss(theta_manifold , cos_theta_euclidean), dim=0)),0)
        for j in range(vectors_manifold.shape[0]):
            if ty == 'map':
                break
            # Compute the inner product on the Riemannian manifold
            theta_manifold = angle_between_vectors(vectors_manifold[j], vectors_manifold[j], g)
            inner_product_euclidean = single_dist_list[j]
            theta_manifold = run_norm(theta_manifold,ds_mean,ds_std)
            cos_theta_euclidean = inner_product_euclidean #inner_product_euclidean / (norm_m * norm_n)
            if i == 0 and j ==0:
                loss_ds = torch.unsqueeze(l1_loss(theta_manifold , cos_theta_euclidean),dim=0)
            else:
                loss_ds = torch.cat((loss_ds, torch.unsqueeze(l1_loss(theta_manifold , cos_theta_euclidean), dim=0)),0)
    if ty =="map":
        return torch.mean(loss),torch.mean(loss_ds)
    elif ty =="dist":
        return torch.mean(loss_ds),torch.mean(loss)


def PunishLocalUnflat_baojiao(x_train, mini_x, gap, Decoder_A, nj_mean, nj_std, ds_mean, ds_std, nearest_neighbors,
                              E_nearest_neighbors, D_nearest_neighbors, k, single_count, ty="map", FLAG=False, root="",
                              mean=[], t=0):

    riemanG = compute_Rieman_metric_vmap(Decoder_A, mini_x.detach())

    bj_Loss, se_loss = conformalMap_new(x_train, mini_x, riemanG, nj_mean, nj_std, ds_mean, ds_std, gap, nearest_neighbors,
                                    E_nearest_neighbors, D_nearest_neighbors, k, single_count, ty=ty)

    return bj_Loss, se_loss

c_list = []
def run_main(file_id, trytime, loss_type, num_train=2000, c_list=c_list, train_epo=100,
             beta=1.0, c=10, theta=0.1, ideal_gap=0.1, noise_ratio=100,
             alpha=2, converge_fasle=0, model_dir="none", d=3,
             train_ty="map", k=5, data_type="high"):

    # Use global variables to track the best model across all run_main calls
    global global_best_loss, global_best_info

    batch_size = 50
    ty = train_ty

    # Compute C(k, 2), the number of pairwise combinations among each point's k neighbors
    # Used later as the label length when slicing pairwise neighborhood-vector inner products
    single_count = calculate_zuhe(k)

    # Read the original point-cloud/sample data
    x_train_raw = np.genfromtxt("./data_2048/eye_04.txt", dtype=np.float32)

    # Standardize the original input for more stable training
    x_train_raw, print_mean, print_std = normalize(x_train_raw)

    # Input feature dimension, e.g. 3 for a 3D point cloud
    f_n = x_train_raw.shape[1]

    torch.set_printoptions(threshold=100000)

    # Precompute:
    # 1) The kNN neighborhood for each point
    # 2) Pairwise neighborhood-vector inner products as the angular-constraint supervision signal
    # 3) Self inner products of individual neighborhood vectors as the distance-constraint supervision signal
    cos_theta_euclidean, distance_euclidean, nearest_neighbors_indices = nearPointEucliAngle(x_train_raw, k)

    # Standardize inner products between different neighborhood vectors as map-type constraint labels
    cos_theta_euclidean, nj_mean, nj_std = normalize(cos_theta_euclidean.cpu().detach().numpy())
    cos_theta_euclidean = torch.tensor(cos_theta_euclidean, dtype=torch.float32).cuda()

    # Standardize self inner products of individual neighborhood vectors as dist-type constraint labels
    distance_euclidean, ds_mean, ds_std = normalize(distance_euclidean.cpu().detach().numpy())
    distance_euclidean = torch.tensor(distance_euclidean, dtype=torch.float32).cuda()

    # Create the model checkpoint directory
    root = "./model_record/eye_04"
    if not os.path.exists(root):
        os.makedirs(root)

    # Create a dedicated subdirectory for this experiment configuration to separate hyperparameters/repeats
    root = root + '/' + str(file_id) + '/_' + str(train_epo) + '_' + str(alpha) + '_' + \
           str(theta) + '_' + str(ideal_gap) + '_' + str(c) + '_' + str(trytime) + '/'

    if not os.path.exists(root):
        os.makedirs(root)

    # Set the latent-space dimension, i.e. the manifold dimension
    latent_n = d

    # Build the dual-decoder autoencoder:
    # encoder: input -> latent variable
    # decoder_A: latent variable -> metric-related output
    # decoder_B: latent variable -> reconstructed original input
    model_n = DualDecoderAutoencoder(latent_n, dim=3)
    model_n = model_n.to(device)

    # Optimizer A: train encoder + decoder_A for geometric/isometry constraints
    optimizer_A = torch.optim.Adam(
        list(model_n.encoder.parameters()) + list(model_n.decoder_A.parameters()),
        lr=0.0001
    )

    # Optimizer B: train encoder + decoder_B for reconstruction
    optimizer_B = torch.optim.Adam(
        list(model_n.encoder.parameters()) + list(model_n.decoder_B.parameters()),
        lr=0.0001
    )

    # Both A and B branches use MSE loss
    criterion_A = nn.MSELoss()
    criterion_B = nn.MSELoss()

    # Build the DataLoader to read training data by batch
    dataset_loader = DataLoader(x_train_raw, batch_size=batch_size, num_workers=0, shuffle=False)

    # If an existing model directory is specified, run only one epoch, usually for testing/loading
    if model_dir != "none":
        num_epochs = 1
    else:
        num_epochs = train_epo

    # Track losses during training
    final_loss = []
    DJ_LOSS_LIST = []
    se_loss_list = []
    rec_loss_list = []
    FINAL_LOSS1 = []
    bj_list = []
    se_list = []
    rec_list = []
    avg_spearman_loss = []

    # Best validation loss within the current run
    best_loss = float('inf')

    # Use the precomputed kNN indices to fetch each sample's nearest-neighbor coordinates
    # Reshape to (N*k, f_n) for later batch slicing
    P_nearest_neighbors = x_train_raw[nearest_neighbors_indices]
    P_nearest_neighbors = torch.tensor(P_nearest_neighbors).cuda().view(-1, f_n)

    # ===== Main training loop =====
    for epoch in range(num_epochs):

        # Intended to clear the lists, but the original code misses parentheses, so nothing is cleared
        final_loss.clear
        FINAL_LOSS1.clear
        avg_spearman_loss.clear

        # Iterate over data by batch
        for j, x_train in enumerate(dataset_loader):

            # Slice the following according to the current batch position:
            # 1) nearest-neighbor points
            # 2) pairwise neighborhood-vector inner product labels
            # 3) single-vector neighborhood distance labels
            if batch_size * k * (j + 1) > P_nearest_neighbors.shape[0]:
                nearest_neighbors = P_nearest_neighbors[batch_size * k * j:]
                neiji_list = cos_theta_euclidean[batch_size * single_count * j:]
                dist_list = distance_euclidean[batch_size * k * j:]
            else:
                nearest_neighbors = P_nearest_neighbors[batch_size * k * j: batch_size * k * (j + 1)]
                neiji_list = cos_theta_euclidean[batch_size * single_count * j: batch_size * single_count * (j + 1)]
                dist_list = distance_euclidean[batch_size * k * j: batch_size * k * (j + 1)]

            # Move the current batch input to GPU
            x_train = x_train.cuda()

            # ==================================================
            # Step 1: train the reconstruction branch (decoder_B)
            # Goal: make the latent variables preserve as much input information as possible
            # ==================================================
            optimizer_B.zero_grad()                 # Clear reconstruction-branch gradients
            _, _, reconstructed = model_n(x_train) # Forward pass and take the reconstruction
            loss_B = criterion_B(reconstructed, x_train)  # Compute reconstruction error
            loss_rec = alpha * loss_B              # Use alpha to weight the reconstruction loss
            loss_rec.backward()                    # Backpropagate the reconstruction loss
            optimizer_B.step()                     # Update encoder + decoder_B parameters

            # ==================================================
            # Step 2: train the geometric-constraint branch (decoder_A)
            # Goal: make local geometric relations in latent space satisfy isometry/conformality constraints
            # ==================================================
            optimizer_A.zero_grad()

            # Encode the current batch samples into latent space
            latent_code, _, _ = model_n(x_train)   # shape: batch_size x latent_dim

            # Encode the current batch's nearest-neighbor points into latent space as well
            L_nearest_neighbors, _, _ = model_n(nearest_neighbors)  # shape: batch_size*k x latent_dim

            for i in range(1):
                # Compute local geometric-constraint losses:
                # bjLoss: main constraint loss, depending on ty as map or dist
                # seloss: auxiliary geometric error for monitoring
                bjLoss, seloss = PunishLocalUnflat_baojiao(
                    x_train, latent_code, ideal_gap, model_n.decoder_A,
                    nj_mean, nj_std, ds_mean, ds_std,
                    L_nearest_neighbors, neiji_list, dist_list,
                    k, single_count, ty=ty, root=root
                )

                # Use theta to control the geometric-constraint strength
                loss_iso = theta * bjLoss

                # Print the average loss over the latest segment every 10 batches
                if (j + 1) % 10 == 0:
                    bj_mean = float(np.array(bj_list).mean()) if len(bj_list) > 0 else 0.0
                    rec_mean = float(np.array(rec_list).mean()) if len(rec_list) > 0 else 0.0
                    se_mean = float(np.array(se_list).mean()) if len(se_list) > 0 else 0.0

                    print(f"Epoch [{epoch + 1}/{num_epochs}]")
                    print("concentrate bjLoss mean:", bj_mean)

                    # When ty=map, seloss is the remaining distance-constraint error
                    # When ty=dist, seloss is the remaining angular-constraint error
                    if ty == "map":
                        print("remain dist mean:", se_mean)
                    elif ty == "dist":
                        print("remain map mean:", se_mean)

                    print("concentrate recon mean:", rec_mean)

                    # Record staged average losses for later saving and plotting
                    DJ_LOSS_LIST.append(bj_mean)
                    se_loss_list.append(se_mean)
                    rec_loss_list.append(rec_mean)

                    # Use geometric loss + reconstruction loss as the validation metric for the current run
                    val_loss = bj_mean + rec_mean

                    # Save best_model.pth if the current run obtains a better result
                    if val_loss < best_loss:
                        best_loss = val_loss
                        torch.save(model_n.state_dict(), root + 'best_model.pth')
                        print("new best in THIS run!  (bj+rec = {:.6f})".format(val_loss))

                    # Update the global best model if this beats all previous run_main results
                    if val_loss < global_best_loss:
                        global_best_loss = val_loss

                        # Record the key hyperparameters that produced the global best model
                        global_best_info = {
                            "alpha": alpha,
                            "theta": theta,
                            "k": k,
                        }

                        # Save as the unified global-best model file
                        torch.save(model_n.state_dict(), "./model_record/eye_04/global_best_model.pth")
                        print(">>> NEW GLOBAL BEST! bj+rec = {:.6f}".format(val_loss))
                        print("    config:", global_best_info)

                    # Clear statistics from the latest 10 batches and start the next accumulation window
                    bj_list.clear()
                    rec_list.clear()
                    se_list.clear()

                else:
                    # For non-10th batches, collect the current batch losses first
                    bj_list.append(bjLoss.item())
                    rec_list.append(loss_B.item())
                    se_list.append(seloss.item())

                # Backpropagate the geometric-constraint loss and update parameters
                optimizer_A.zero_grad()
                loss_iso.backward()
                optimizer_A.step()

            # End of one batch; proceed to the next batch
            continue

    # ===== Save loss-curve data after training =====
    np.save(root + 'DJ_LOSS_LIST.npy', DJ_LOSS_LIST)

    if ty == "map":
        np.save(root + 'dist_LOSS.npy', se_loss_list)
    elif ty == "dist":
        np.save(root + 'map_LOSS.npy', se_loss_list)

    np.save(root + 'rec_loss_list.npy', rec_loss_list)

    return True

# Global record of the best result across all run_main calls
global_best_loss = float('inf')   # Minimum value of bjMean + recMean
global_best_info = None           # Optional: record which hyperparameter set produced it

if __name__ == '__main__':
    data_type = "low"  # Whether the dataset is high- or low-dimensional; 3D datasets are low, above 3D are high
    train_ty = "map"  # dist means sample-distance inner products; map means inner products between different vectors
    false_time_record = 0  # Unused
    for beta in [0.02]:  # Unused
        for t_e in [500]:  # Number of model-training epochs
            for c in [10]:  # Unused
                for alhpa in [1,5,10]:  # Reconstruction-accuracy strength
                    for theta in [10]:  # Isometry-accuracy strength
                        for ideal_gap in [0.1]:  # Unused
                            for dim in [2]:  # Controls the manifold dimension; for 3D datasets, this is always set to 2
                                for k in [10,7]:  # Controls kNN
                                    t = 0
                                    while t <2:  # Number of repeated experiments for a single model setting
                                        ret = run_main("eye04"+"_"+str(dim)+"_"+str(k),t,2,d=dim,c_list =  c_list,train_epo=t_e,beta =beta,c=c,alpha=alhpa,theta = theta,ideal_gap=ideal_gap,noise_ratio=1000,converge_fasle=false_time_record,train_ty=train_ty,k = k,data_type=data_type)
                                        if not ret:
                                            false_time_record+=1
                                        if ret:
                                            t+=1
