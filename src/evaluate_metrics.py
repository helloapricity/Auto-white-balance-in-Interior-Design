import torch
import numpy as np
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
import cv2
from skimage import color
import json
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# """FID SCORE"""
# fid = FrechetInceptionDistance(feature=2048)

# def load_image_to_tensor(image_path):
#     img = Image.open(image_path)
#     img_tensor = torch.tensor(np.array(img), dtype=torch.uint8).permute(2, 0, 1).unsqueeze(0)
#     return img_tensor

# version = "AWB_style_loss_author_dataset"
# sample_epoch_name = "05"
# # Đường dẫn chứa các thư mục con
# base_path = f"datahub/results/{version}/sample-epoch={sample_epoch_name}"

# # Lấy danh sách các chỉ số từ tên thư mục
# indices = [int(folder) for folder in os.listdir(base_path) if folder.isdigit()]

# # Sắp xếp lại danh sách chỉ số
# indices.sort()

# # Tạo danh sách đường dẫn cho ảnh gốc và ảnh đã chỉnh sửa
# real_images_paths = [
#     f"{base_path}/{i}/{i}_G.jpg" for i in indices
# ]

# fake_images_paths = [
#     f"{base_path}/{i}/{i}_output.png" for i in indices
# ]

# # print("Calculating FID Score...")
# # for real_img_path in tqdm(real_images_paths, desc="Real Images"):
# #     img_tensor = load_image_to_tensor(real_img_path)
# #     fid.update(img_tensor, real=True)

# # for fake_img_path in tqdm(fake_images_paths, desc="Fake Images"):
# #     img_tensor = load_image_to_tensor(fake_img_path)
# #     fid.update(img_tensor, real=False)

# # fid_value = fid.compute()
# # print(f"1. FID score: {fid_value.item():.2f}")

# # """MS SSIM"""
# # # Initialize the MS-SSIM metric
# # ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
# # """LPIPS"""
# # lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

# # # List to store SSIM results
# # ssim_scores = []
# # lpips_scores = []

# # Function to load and normalize image
# def load_image_to_tensor_norm_0_1(image_path):
#     img = Image.open(image_path)
#     img_tensor = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
#     img_tensor /= 255.0  # Normalize to [0, 1]
#     return img_tensor

# # print("Calculating MS-SSIM and LPIPS Scores...")
# # for real_path, fake_path in tqdm(zip(real_images_paths, fake_images_paths), desc="Image Pairs", total=len(real_images_paths)):
# #     real_img = load_image_to_tensor_norm_0_1(real_path)
# #     fake_img = load_image_to_tensor_norm_0_1(fake_path)

# #     # Calculate SSIM
# #     ssim_score = ms_ssim(real_img, fake_img).item()
# #     ssim_scores.append(ssim_score)

# #     # Calculate LPIPS
# #     lpips_score = lpips(real_img, fake_img).item()
# #     lpips_scores.append(lpips_score)

# # Save the SSIM scores
# # with open("./1_evaluate_score/SSIM_Score.json", "w") as f:
# #     json.dump(ssim_scores, f)
    
# # with open("./1_evaluate_score/LPIPS_Score.json", "w") as f:
# #     json.dump(lpips_scores, f)

# """DELTA E2000"""
# def calc_deltaE2000(source, target):
#     source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
#     target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
#     source = color.rgb2lab(source)
#     target = color.rgb2lab(target)
#     source = np.reshape(source, [-1, 3]).astype(np.float32)
#     target = np.reshape(target, [-1, 3]).astype(np.float32)
#     deltaE00 = deltaE2000(source, target)
#     return sum(deltaE00) / (np.shape(deltaE00)[0])

 
def deltaE2000(Labstd, Labsample):
    kl = 1
    kc = 1
    kh = 1
    
    Lstd = np.transpose(Labstd[:, 0])
    astd = np.transpose(Labstd[:, 1])
    bstd = np.transpose(Labstd[:, 2])
    
    Cabstd = np.sqrt(np.power(astd, 2) + np.power(bstd, 2))
    
    Lsample = np.transpose(Labsample[:, 0])
    asample = np.transpose(Labsample[:, 1])
    bsample = np.transpose(Labsample[:, 2])
    
    Cabsample = np.sqrt(np.power(asample, 2) + np.power(bsample, 2))
    
    Cabarithmean = (Cabstd + Cabsample) / 2
    
    G = 0.5 * (1 - np.sqrt((np.power(Cabarithmean, 7)) / (np.power(Cabarithmean, 7) + np.power(25, 7))))
    
    apstd = (1 + G) * astd
    apsample = (1 + G) * asample
    
    Cpsample = np.sqrt(np.power(apsample, 2) + np.power(bsample, 2))
    Cpstd = np.sqrt(np.power(apstd, 2) + np.power(bstd, 2))
    
    Cpprod = (Cpsample * Cpstd)
    
    zcidx = np.argwhere(Cpprod == 0)
    hpstd = np.arctan2(bstd, apstd)
    
    hpstd[np.argwhere((np.abs(apstd) + np.abs(bstd)) == 0)] = 0
    hpsample = np.arctan2(bsample, apsample)
    hpsample = hpsample + 2 * np.pi * (hpsample < 0)
    hpsample[np.argwhere((np.abs(apsample) + np.abs(bsample)) == 0)] = 0
    dL = (Lsample - Lstd)
    dC = (Cpsample - Cpstd)
    dhp = (hpsample - hpstd)
    dhp = dhp - 2 * np.pi * (dhp > np.pi)
    dhp = dhp + 2 * np.pi * (dhp < (-np.pi))
    dhp[zcidx] = 0
    dH = 2 * np.sqrt(Cpprod) * np.sin(dhp / 2)
    Lp = (Lsample + Lstd) / 2
    Cp = (Cpstd + Cpsample) / 2
    hp = (hpstd + hpsample) / 2
    hp = hp - (np.abs(hpstd - hpsample) > np.pi) * np.pi
    hp = hp + (hp < 0) * 2 * np.pi
    hp[zcidx] = hpsample[zcidx] + hpstd[zcidx]
    Lpm502 = np.power((Lp - 50), 2)
    Sl = 1 + 0.015 * Lpm502 / np.sqrt(20 + Lpm502)
    Sc = 1 + 0.045 * Cp
    T = 1 - 0.17 * np.cos(hp - np.pi / 6) + 0.24 * np.cos(2 * hp) + \
        0.32 * np.cos(3 * hp + np.pi / 30) \
        - 0.20 * np.cos(4 * hp - 63 * np.pi / 180)
    Sh = 1 + 0.015 * Cp * T
    delthetarad = (30 * np.pi / 180) * np.exp(
    - np.power((180 / np.pi * hp - 275) / 25, 2))
    Rc = 2 * np.sqrt((np.power(Cp, 7)) / (np.power(Cp, 7) + np.power(25, 7)))
    RT = - np.sin(2 * delthetarad) * Rc
    klSl = kl * Sl
    kcSc = kc * Sc
    khSh = kh * Sh
    de00 = np.sqrt(np.power((dL / klSl), 2) + np.power((dC / kcSc), 2) +
                    np.power((dH / khSh), 2) + RT * (dC / kcSc) * (dH / khSh))
    return de00

"""MSE SCORE"""
# def calc_mse(source, target):
#     source = np.reshape(source, [-1, 1]).astype(np.float64)
#     target = np.reshape(target, [-1, 1]).astype(np.float64)
#     mse = sum(np.power((source - target), 2))
#     return mse / (np.shape(source)[0])

def calc_mse(source, target):
    source = np.reshape(source, [-1, 3]).astype(np.float64)  # Đảm bảo mỗi điểm ảnh có 3 kênh (RGB)
    target = np.reshape(target, [-1, 3]).astype(np.float64)
    mse = np.mean(np.square(source - target))  # Tính MSE trên toàn bộ pixel và tất cả kênh
    return mse

"""MAE SCORE"""
def calc_mae(source, target):
    source = np.reshape(source, [-1, 3]).astype(np.float32)
    target = np.reshape(target, [-1, 3]).astype(np.float32)
    source_norm = np.sqrt(np.sum(np.power(source, 2), 1))
    target_norm = np.sqrt(np.sum(np.power(target, 2), 1))
    norm = source_norm * target_norm
    L = np.shape(norm)[0]
    inds = norm != 0
    angles = np.sum(source[inds, :] * target[inds, :], 1) / norm[inds]
    angles[angles > 1] = 1
    f = np.arccos(angles)
    f[np.isnan(f)] = 0
    f = f * 180 / np.pi
    return sum(f) / (L)


"""DELTA E"""
def calc_deltaE(source, target):
    source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    source = color.rgb2lab(source)
    target = color.rgb2lab(target)
    source = np.reshape(source, [-1, 3]).astype(np.float32)
    target = np.reshape(target, [-1, 3]).astype(np.float32)
    delta_e = np.sqrt(np.sum(np.power(source - target, 2), 1))
    return sum(delta_e) / (np.shape(delta_e)[0])

# def evaluate_cc(corrected, gt, opt=4):
#     if opt == 1:
#         return  calc_deltaE2000(corrected, gt)
#     elif opt == 2:
#         return  calc_deltaE2000(corrected, gt), \
#                 calc_mse(corrected, gt)
#     elif opt == 3:
#         return  calc_deltaE2000(corrected, gt), \
#                 calc_mse(corrected, gt), \
#                 calc_mae(corrected, gt)
#     elif opt == 4:
#         return  calc_deltaE2000(corrected, gt), \
#                 calc_mse(corrected, gt), \
#                 calc_mae(corrected, gt), \
#                 calc_deltaE(corrected, gt)
#     else:
#         raise Exception('Error in evaluate_cc function')

# # Tạo các danh sách để lưu trữ kết quả
# delta_e2000_scores = []
# mse_scores = []
# mae_scores = []
# delta_e_scores = []

# print("Calculating Delta E2000, MSE, MAE, and Delta E Scores...")
# for real_path, fake_path in tqdm(zip(real_images_paths, fake_images_paths), desc="Evaluating Color Metrics", total=len(real_images_paths)):
#     source_image = cv2.imread(real_path)
#     target_image = cv2.imread(fake_path)
#     results = evaluate_cc(source_image, target_image, opt=4)

#     # Lưu kết quả
#     delta_e2000_scores.append(results[0])
#     mse_scores.append(results[1][0])
#     mae_scores.append(results[2])
#     delta_e_scores.append(results[3])

# # Lưu các kết quả ra file JSON
# with open("./1_evaluate_score/DeltaE2000.json", "w") as f:
#     json.dump(delta_e2000_scores, f)

# with open("./1_evaluate_score/MSE_Score.json", "w") as f:
#     json.dump(mse_scores, f)

# with open("./1_evaluate_score/MAE_Score.json", "w") as f:
#     json.dump(mae_scores, f)

# with open("./1_evaluate_score/DeltaE.json", "w") as f:
#     json.dump(delta_e_scores, f)

# # print("Scores saved to JSON files.")

# def calculate_statistics(data):
#     mean = np.mean(data)
#     q1 = np.percentile(data, 25)
#     q2 = np.median(data)
#     q3 = np.percentile(data, 75)
#     return mean, q1, q2, q3

# # mean, q1, q2, q3 = calculate_statistics(ssim_scores)
# # print(f"1. MS-SSIM - Mean: {mean:.4f}, Q1: {q1:.4f}, Q2 (Median): {q2:.4f}, Q3: {q3:.4f}")

# # mean, q1, q2, q3 = calculate_statistics(lpips_scores)
# # print(f"2. LPIPS - Mean: {mean:.4f}, Q1: {q1:.4f}, Q2 (Median): {q2:.4f}, Q3: {q3:.4f}")

# mean, q1, q2, q3 = calculate_statistics(delta_e2000_scores)
# print(f"3. Delta E2000 - Mean: {mean:.2f}, Q1: {q1:.2f}, Q2 (Median): {q2:.2f}, Q3: {q3:.2f}")

# mean, q1, q2, q3 = calculate_statistics(mse_scores)
# print(f"4. MSE - Mean: {mean:.2f}, Q1: {q1:.2f}, Q2 (Median): {q2:.2f}, Q3: {q3:.2f}")

# mean, q1, q2, q3 = calculate_statistics(mae_scores)
# print(f"5. MAE - Mean: {mean:.2f}, Q1: {q1:.2f}, Q2 (Median): {q2:.2f}, Q3: {q3:.2f}")

# mean, q1, q2, q3 = calculate_statistics(delta_e_scores)
# print(f"6. Delta E - Mean: {mean:.2f}, Q1: {q1:.2f}, Q2 (Median): {q2:.2f}, Q3: {q3:.2f}")




NUM_PROCESSES = min(12, cpu_count())

"""Function Definitions"""
def load_image_to_tensor_norm_0_1(image_path):
    img = Image.open(image_path)
    img_tensor = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    img_tensor /= 255.0  # Normalize to [0, 1]
    return img_tensor

def calc_deltaE2000(source, target):
    source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    source = color.rgb2lab(source)
    target = color.rgb2lab(target)
    source = np.reshape(source, [-1, 3]).astype(np.float32)
    target = np.reshape(target, [-1, 3]).astype(np.float32)
    deltaE00 = deltaE2000(source, target)
    return sum(deltaE00) / (np.shape(deltaE00)[0])

def evaluate_cc(corrected, gt, opt=4):
    if opt == 1:
        return calc_deltaE2000(corrected, gt)
    elif opt == 2:
        return calc_deltaE2000(corrected, gt), calc_mse(corrected, gt)
    elif opt == 3:
        return calc_deltaE2000(corrected, gt), calc_mse(corrected, gt), calc_mae(corrected, gt)
    elif opt == 4:
        return calc_deltaE2000(corrected, gt), calc_mse(corrected, gt), calc_mae(corrected, gt), calc_deltaE(corrected, gt)
    else:
        raise Exception('Error in evaluate_cc function')

def process_image_pair(image_pair):
    real_path, fake_path = image_pair
    source_image = cv2.imread(real_path)
    target_image = cv2.imread(fake_path)
    results = evaluate_cc(source_image, target_image, opt=4)
    return results

def calculate_statistics(data):
    mean = np.mean(data)
    q1 = np.percentile(data, 25)
    q2 = np.median(data)
    q3 = np.percentile(data, 75)
    return mean, q1, q2, q3

"""Main Execution"""
if __name__ == "__main__":
    version = "AWB_style_loss_author_dataset"
    sample_epoch_name = "26"
    base_path = f"datahub/results/{version}/sample-epoch={sample_epoch_name}"

    indices = [int(folder) for folder in os.listdir(base_path) if folder.isdigit()]
    indices.sort()

    real_images_paths = [f"{base_path}/{i}/{i}_G.jpg" for i in indices]
    fake_images_paths = [f"{base_path}/{i}/{i}_output.png" for i in indices]
    image_pairs = list(zip(real_images_paths, fake_images_paths))

    print("Calculating Delta E2000, MSE, MAE, and Delta E Scores using multiprocessing...")

    with Pool(processes=NUM_PROCESSES) as pool:
        results = list(tqdm(pool.imap(process_image_pair, image_pairs), total=len(image_pairs)))

    # Tách các kết quả ra từng danh sách
    delta_e2000_scores = [res[0] for res in results]
    mse_scores = [res[1][0] for res in results]
    mae_scores = [res[2] for res in results]
    delta_e_scores = [res[3] for res in results]

    # Lưu các kết quả ra file JSON
    with open("./1_evaluate_score/DeltaE2000.json", "w") as f:
        json.dump(delta_e2000_scores, f)

    with open("./1_evaluate_score/MSE_Score.json", "w") as f:
        json.dump(mse_scores, f)

    with open("./1_evaluate_score/MAE_Score.json", "w") as f:
        json.dump(mae_scores, f)

    with open("./1_evaluate_score/DeltaE.json", "w") as f:
        json.dump(delta_e_scores, f)

    # Tính toán thống kê
    mean, q1, q2, q3 = calculate_statistics(delta_e2000_scores)
    print(f"1. Delta E2000 - Mean: {mean:.2f}, Q1: {q1:.2f}, Q2 (Median): {q2:.2f}, Q3: {q3:.2f}")

    mean, q1, q2, q3 = calculate_statistics(mse_scores)
    print(f"2. MSE - Mean: {mean:.2f}, Q1: {q1:.2f}, Q2 (Median): {q2:.2f}, Q3: {q3:.2f}")

    mean, q1, q2, q3 = calculate_statistics(mae_scores)
    print(f"3. MAE - Mean: {mean:.2f}, Q1: {q1:.2f}, Q2 (Median): {q2:.2f}, Q3: {q3:.2f}")

    mean, q1, q2, q3 = calculate_statistics(delta_e_scores)
    print(f"4. Delta E - Mean: {mean:.2f}, Q1: {q1:.2f}, Q2 (Median): {q2:.2f}, Q3: {q3:.2f}")

    print("Statistics calculated and displayed.")