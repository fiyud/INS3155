import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

def load_stereo_images(left_path, right_path):
    img_left = cv2.imread(left_path)
    img_right = cv2.imread(right_path)
    if img_left is None or img_right is None:
        raise ValueError(f"Error loading images from {left_path} and {right_path}")
    if img_left.shape != img_right.shape:
        raise ValueError("Left and right images must have the same dimensions")
    return img_left, img_right

def compute_disparity_block_matching(img_left, img_right, block_size=15, max_disp=64):
    if len(img_left.shape) == 3:
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    else:
        gray_left = img_left
        gray_right = img_right
    stereo = cv2.StereoBM_create(numDisparities=max_disp, blockSize=block_size)
    disparity = stereo.compute(gray_left, gray_right)
    disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255,
                                        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return disparity, disparity_normalized

def compute_disparity_sgbm(img_left, img_right, min_disp=0, max_disp=160, block_size=5):
    max_disp = (max_disp // 16) * 16
    window_size = block_size
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=max_disp,
        blockSize=block_size,
        P1=8 * 3 * window_size**2,
        P2=32 * 3 * window_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    disparity = left_matcher.compute(img_left, img_right).astype(np.float32) / 16.0
    disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255,
                                        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return disparity, disparity_normalized

def reconstruct_3d_point_cloud(disparity, img_left, Q=None, mask_threshold=0):
    h, w = disparity.shape
    if Q is None:
        f = 0.8 * w
        cx, cy = w / 2, h / 2
        baseline = 0.1
        Q = np.array([
            [1, 0, 0, -cx],
            [0, 1, 0, -cy],
            [0, 0, 0, f],
            [0, 0, -1/baseline, 0]
        ])
    mask = disparity > mask_threshold
    points_3d = cv2.reprojectImageTo3D(disparity, Q)
    if len(img_left.shape) == 3:
        colors = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
    else:
        colors = np.stack([img_left, img_left, img_left], axis=2)
    points = points_3d[mask]
    colors = colors[mask]
    return points, colors

def visualize_point_cloud(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    o3d.io.write_point_cloud("point_cloud.ply", pcd)
    o3d.visualization.draw_geometries([pcd])
    return pcd

def visualize_point_cloud_matplotlib(points, colors, step=100):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    points_downsampled = points[::step]
    colors_downsampled = colors[::step]
    colors_normalized = colors_downsampled / 255.0
    ax.scatter(
        points_downsampled[:, 0],
        points_downsampled[:, 1],
        points_downsampled[:, 2],
        c=colors_normalized,
        s=1
    )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=-70, azim=-90)
    plt.savefig('point_cloud_matplotlib.png', dpi=300, bbox_inches='tight')
    plt.show()

def find_keypoints_and_matches(img_left, img_right):
    if len(img_left.shape) == 3:
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    else:
        gray_left = img_left
        gray_right = img_right
    sift = cv2.SIFT_create()
    kp_left, des_left = sift.detectAndCompute(gray_left, None)
    kp_right, des_right = sift.detectAndCompute(gray_right, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_left, des_right, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    return kp_left, kp_right, good_matches

def estimate_fundamental_matrix(kp_left, kp_right, matches):
    if len(matches) < 8:
        print("Not enough matches to compute fundamental matrix (need at least 8)")
        F = np.eye(3)
        return F, np.array([]), np.array([])
    pts_left = np.float32([kp_left[m.queryIdx].pt for m in matches])
    pts_right = np.float32([kp_right[m.trainIdx].pt for m in matches])
    try:
        F, mask = cv2.findFundamentalMat(pts_left, pts_right, cv2.FM_RANSAC,
                                      ransacReprojThreshold=3.0, confidence=0.99)
        if F is None or mask is None:
            print("Could not compute a valid fundamental matrix")
            F = np.eye(3)
            return F, np.array([]), np.array([])
        inlier_mask = mask.ravel() == 1
        pts_left_inliers = pts_left[inlier_mask]
        pts_right_inliers = pts_right[inlier_mask]
        if len(pts_left_inliers) < 8:
            print(f"Too few inliers ({len(pts_left_inliers)}) to compute a reliable fundamental matrix")
    except Exception as e:
        print(f"Error computing fundamental matrix: {e}")
        F = np.eye(3)
        return F, np.array([]), np.array([])
    return F, pts_left_inliers, pts_right_inliers

def draw_epipolar_lines(img_left, img_right, pts_left, pts_right, F, num_lines=10):
    h, w = img_left.shape[:2]
    img_left_lines = img_left.copy()
    img_right_lines = img_right.copy()
    if len(pts_left) == 0 or len(pts_right) == 0:
        print("No inliers to draw epipolar lines. Showing original images instead.")
        text = "No valid inliers found for epipolar lines"
        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(text, font, 1, 2)[0]
        textX = (img_left.shape[1] - textsize[0]) // 2
        textY = (img_left.shape[0] + textsize[1]) // 2
        cv2.putText(img_left_lines, text, (textX, textY), font, 1, (0, 0, 255), 2)
        cv2.putText(img_right_lines, text, (textX, textY), font, 1, (0, 0, 255), 2)
        img_epipolar = np.hstack((img_left_lines, img_right_lines))
        cv2.imwrite('epipolar_lines.png', img_epipolar)
        plt.figure(figsize=(15, 7))
        plt.imshow(cv2.cvtColor(img_epipolar, cv2.COLOR_BGR2RGB))
        plt.title('Original Images (No Epipolar Lines)')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('epipolar_lines_plt.png', dpi=300, bbox_inches='tight')
        plt.show()
        return img_epipolar
    if len(pts_left) > num_lines:
        indices = np.linspace(0, len(pts_left) - 1, num_lines).astype(int)
        pts_left_subset = pts_left[indices]
        pts_right_subset = pts_right[indices]
    else:
        pts_left_subset = pts_left
        pts_right_subset = pts_right
    for pt in pts_left_subset:
        cv2.circle(img_left_lines, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
    for pt in pts_right_subset:
        cv2.circle(img_right_lines, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
    try:
        lines_right = cv2.computeCorrespondEpilines(pts_left_subset.reshape(-1, 1, 2), 1, F)
        lines_right = lines_right.reshape(-1, 3)
        lines_left = cv2.computeCorrespondEpilines(pts_right_subset.reshape(-1, 1, 2), 2, F)
        lines_left = lines_left.reshape(-1, 3)
        for i, (pt_left, pt_right) in enumerate(zip(pts_left_subset, pts_right_subset)):
            try:
                color = np.random.randint(0, 255, 3).tolist()
                right_line = lines_right[i]
                if abs(right_line[1]) > 1e-5:
                    x0, y0 = 0, int(-right_line[2] / right_line[1])
                    x1, y1 = w, int(-(right_line[2] + right_line[0] * w) / right_line[1])
                    if 0 <= y0 <= h and 0 <= y1 <= h:
                        img_right_lines = cv2.line(img_right_lines, (x0, y0), (x1, y1), color, 1)
                left_line = lines_left[i]
                if abs(left_line[1]) > 1e-5:
                    x0, y0 = 0, int(-left_line[2] / left_line[1])
                    x1, y1 = w, int(-(left_line[2] + left_line[0] * w) / left_line[1])
                    if 0 <= y0 <= h and 0 <= y1 <= h:
                        img_left_lines = cv2.line(img_left_lines, (x0, y0), (x1, y1), color, 1)
            except Exception as e:
                print(f"Error drawing epipolar line {i}: {e}")
                continue
    except Exception as e:
        print(f"Error computing epipolar lines: {e}")
    img_epipolar = np.hstack((img_left_lines, img_right_lines))
    cv2.imwrite('epipolar_lines.png', img_epipolar)
    plt.figure(figsize=(15, 7))
    plt.imshow(cv2.cvtColor(img_epipolar, cv2.COLOR_BGR2RGB))
    plt.title('Epipolar Lines')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('epipolar_lines_plt.png', dpi=300, bbox_inches='tight')
    plt.show()
    return img_epipolar

def visualize_matches(img_left, img_right, kp_left, kp_right, matches, mask=None):
    img_matches = cv2.drawMatches(img_left, kp_left, img_right, kp_right, matches, None,
                                 matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
                                 matchesMask=mask, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite('feature_matches.png', img_matches)
    plt.figure(figsize=(15, 7))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title('Feature Matches')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('feature_matches_plt.png', dpi=300, bbox_inches='tight')
    plt.show()
    return img_matches

def rectify_stereo_images(img_left, img_right, F):
    h, w = img_left.shape[:2]
    _, H1, H2 = cv2.stereoRectifyUncalibrated(
        np.float32(np.column_stack(np.where(np.ones((h, w)) > 0)[::-1])),
        np.float32(np.column_stack(np.where(np.ones((h, w)) > 0)[::-1])),
        F, (w, h)
    )
    img_left_rectified = cv2.warpPerspective(img_left, H1, (w, h))
    img_right_rectified = cv2.warpPerspective(img_right, H2, (w, h))
    img_rectified = np.hstack((img_left_rectified, img_right_rectified))
    cv2.imwrite('rectified_stereo.png', img_rectified)
    plt.figure(figsize=(15, 7))
    plt.imshow(cv2.cvtColor(img_rectified, cv2.COLOR_BGR2RGB))
    plt.title('Rectified Stereo Images')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('rectified_stereo_plt.png', dpi=300, bbox_inches='tight')
    plt.show()
    return img_left_rectified, img_right_rectified

def evaluate_disparity_map(disparity, ground_truth=None):
    if ground_truth is None:
        print("No ground truth disparity map provided for evaluation.")
        return None
    if ground_truth.max() > 255:
        ground_truth_norm = cv2.normalize(ground_truth, None, alpha=0, beta=255,
                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    else:
        ground_truth_norm = ground_truth
    if disparity.shape != ground_truth_norm.shape:
        print(f"Error: Disparity ({disparity.shape}) and ground truth ({ground_truth_norm.shape}) shapes don't match.")
        return None
    mae = np.mean(np.abs(disparity - ground_truth_norm))
    rmse = np.sqrt(np.mean((disparity - ground_truth_norm) ** 2))
    threshold = 5
    bpr = np.sum(np.abs(disparity - ground_truth_norm) > threshold) / disparity.size
    results = {
        'MAE': mae,
        'RMSE': rmse,
        'BPR': bpr
    }
    print("Disparity Map Evaluation Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Square Error (RMSE): {rmse:.2f}")
    print(f"Bad Pixel Ratio (BPR) at threshold {threshold}: {bpr:.4f}")
    return results

def compare_disparity_methods(img_left, img_right):
    disparity_bm, disparity_bm_norm = compute_disparity_block_matching(
        img_left, img_right, block_size=15, max_disp=64
    )
    disparity_sgbm, disparity_sgbm_norm = compute_disparity_sgbm(
        img_left, img_right, min_disp=0, max_disp=160, block_size=5
    )
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(disparity_bm_norm, cmap='plasma')
    plt.title('Block Matching Disparity')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(disparity_sgbm_norm, cmap='plasma')
    plt.title('SGBM Disparity')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('disparity_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    return (disparity_bm, disparity_bm_norm), (disparity_sgbm, disparity_sgbm_norm)

def create_synthetic_stereo_pair(width=600, height=400):
    gradient = np.linspace(150, 220, width).astype(np.uint8)
    img_left = np.zeros((height, width, 3), dtype=np.uint8)
    img_right = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        img_left[i, :, :] = gradient.reshape(1, -1, 1)
        img_right[i, :, :] = gradient.reshape(1, -1, 1)
    grid_step = 50
    grid_color = (130, 130, 130)
    grid_thickness = 1
    for x in range(0, width, grid_step):
        cv2.line(img_left, (x, 0), (x, height), grid_color, grid_thickness)
        cv2.line(img_right, (x, 0), (x, height), grid_color, grid_thickness)
    for y in range(0, height, grid_step):
        cv2.line(img_left, (0, y), (width, y), grid_color, grid_thickness)
        cv2.line(img_right, (0, y), (width, y), grid_color, grid_thickness)
    np.random.seed(42)
    shapes = [
        {'type': 'rectangle', 'center': (150, 150), 'size': (80, 80), 'color': (255, 0, 0), 'disparity': 25},
        {'type': 'circle', 'center': (400, 200), 'radius': 60, 'color': (0, 0, 255), 'disparity': 15},
        {'type': 'triangle', 'vertices': np.array([[300, 100], [350, 200], [250, 200]]), 'color': (0, 255, 0), 'disparity': 35},
        {'type': 'rectangle', 'center': (450, 300), 'size': (60, 60), 'color': (255, 255, 0), 'disparity': 20},
        {'type': 'circle', 'center': (200, 300), 'radius': 40, 'color': (255, 0, 255), 'disparity': 10}
    ]
    for shape in shapes:
        if shape['type'] == 'rectangle':
            x, y = shape['center']
            w, h = shape['size']
            disp = shape['disparity']
            pt1_left = (x - w//2, y - h//2)
            pt2_left = (x + w//2, y + h//2)
            cv2.rectangle(img_left, pt1_left, pt2_left, shape['color'], -1)
            pt1_right = (x - w//2 - disp, y - h//2)
            pt2_right = (x + w//2 - disp, y + h//2)
            cv2.rectangle(img_right, pt1_right, pt2_right, shape['color'], -1)
        elif shape['type'] == 'circle':
            x, y = shape['center']
            r = shape['radius']
            disp = shape['disparity']
            cv2.circle(img_left, (x, y), r, shape['color'], -1)
            cv2.circle(img_right, (x - disp, y), r, shape['color'], -1)
        elif shape['type'] == 'triangle':
            vertices = shape['vertices']
            disp = shape['disparity']
            cv2.fillPoly(img_left, [vertices], shape['color'])
            vertices_right = vertices.copy()
            vertices_right[:, 0] -= disp
            cv2.fillPoly(img_right, [vertices_right], shape['color'])
    for _ in range(30):
        x = np.random.randint(50, width-50)
        y = np.random.randint(50, height-50)
        r = np.random.randint(3, 8)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        disp = np.random.randint(5, 30)
        cv2.circle(img_left, (x, y), r, color, -1)
        cv2.circle(img_right, (x - disp, y), r, color, -1)
    noise_left = np.random.randint(0, 15, (height, width, 3), dtype=np.int16)
    noise_right = np.random.randint(0, 15, (height, width, 3), dtype=np.int16)
    img_left = np.clip(img_left.astype(np.int16) + noise_left, 0, 255).astype(np.uint8)
    img_right = np.clip(img_right.astype(np.int16) + noise_right, 0, 255).astype(np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_left, 'Left Image', (20, 30), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img_right, 'Right Image', (20, 30), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite('synthetic_left.png', img_left)
    cv2.imwrite('synthetic_right.png', img_right)
    return img_left, img_right

def main():
    try_filenames = [
        ('left.jpg', 'right.jpg'),
        ('im0.png', 'im1.png'),
        ('left.png', 'right.png'),
        ('stereo_left.jpg', 'stereo_right.jpg')
    ]
    img_left = None
    img_right = None
    for left_name, right_name in try_filenames:
        try:
            img_left, img_right = load_stereo_images(left_name, right_name)
            print(f"Loaded images from '{left_name}' and '{right_name}'")
            break
        except Exception:
            continue
    if img_left is None or img_right is None:
        print("No stereo images found. Creating synthetic stereo pair...")
        img_left, img_right = create_synthetic_stereo_pair(600, 400)
    print("Finding keypoints and matches...")
    kp_left, kp_right, matches = find_keypoints_and_matches(img_left, img_right)
    print(f"Found {len(kp_left)} keypoints in left image, {len(kp_right)} in right image")
    print(f"Found {len(matches)} good matches")
    visualize_matches(img_left, img_right, kp_left, kp_right, matches[:100])
    print("Estimating fundamental matrix...")
    F, pts_left_inliers, pts_right_inliers = estimate_fundamental_matrix(kp_left, kp_right, matches)
    print(f"Fundamental matrix:\n{F}")
    print(f"Found {len(pts_left_inliers)} inlier matches")
    print("Drawing epipolar lines...")
    draw_epipolar_lines(img_left, img_right, pts_left_inliers, pts_right_inliers, F, num_lines=15)
    print("Computing disparity maps...")
    (disparity_bm, disparity_bm_norm), (disparity_sgbm, disparity_sgbm_norm) = compare_disparity_methods(img_left, img_right)
    print("Reconstructing 3D point cloud...")
    points, colors = reconstruct_3d_point_cloud(disparity_sgbm, img_left, mask_threshold=5)
    try:
        print("Visualizing point cloud with Open3D...")
        visualize_point_cloud(points, colors)
    except Exception as e:
        print(f"Error using Open3D: {e}")
        print("Falling back to Matplotlib for point cloud visualization...")
        visualize_point_cloud_matplotlib(points, colors)
    print("Rectifying stereo images...")
    img_left_rectified, img_right_rectified = rectify_stereo_images(img_left, img_right, F)
    print("Computing disparity maps on rectified images...")
    compare_disparity_methods(img_left_rectified, img_right_rectified)
    print("Stereo reconstruction pipeline completed!")

if __name__ == "__main__":
    main()