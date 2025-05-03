import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def load_images(folder_path):
    for ext in ["jpg", "png", "jpeg", "JPG", "PNG"]:
        image_paths = sorted(glob.glob(os.path.join(folder_path, f"*.{ext}")))
        if image_paths:
            break
    if not image_paths:
        print(f"No images found in {folder_path}")
        return []
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
            print(f"Loaded image: {path}, shape: {img.shape}")
    return images

def create_test_images():
    base = np.zeros((400, 1200, 3), dtype=np.uint8)
    for i in range(1200):
        value = int(180 + 75 * np.sin(i * np.pi / 600))
        base[:, i] = [value, value, value]
    cv2.rectangle(base, (500, 150), (700, 250), (0, 0, 255), -1)
    cv2.circle(base, (300, 200), 70, (255, 0, 0), -1)
    cv2.circle(base, (900, 200), 70, (0, 255, 0), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(base, "PANORAMA", (550, 300), font, 1, (255, 255, 255), 2)
    cv2.putText(base, "LEFT", (250, 300), font, 1, (255, 255, 255), 2)
    cv2.putText(base, "RIGHT", (850, 300), font, 1, (255, 255, 255), 2)
    images = []
    width = 400
    for i in range(4):
        start_x = i * 200
        img = base[:, start_x:start_x+width].copy()
        cv2.putText(img, f"Image {i+1}", (10, 30), font, 1, (255, 255, 255), 2)
        images.append(img)
        cv2.imwrite(f"test_image_{i+1}.jpg", img)
    cv2.imwrite("full_panorama.jpg", base)
    return images

def detect_and_match_features(img1, img2, method="sift"):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    if method.lower() == "sift":
        detector = cv2.SIFT_create()
    elif method.lower() == "orb":
        detector = cv2.ORB_create(nfeatures=2000)
    else:
        print(f"Unknown method: {method}. Using SIFT.")
        detector = cv2.SIFT_create()
    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)
    print(f"Detected {len(kp1)} keypoints in image 1 and {len(kp2)} in image 2")
    if method.lower() == "orb":
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        matcher = cv2.BFMatcher()
    raw_matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in raw_matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    print(f"Found {len(good_matches)} good matches out of {len(raw_matches)} total matches")
    return kp1, kp2, good_matches

def find_homography(kp1, kp2, matches):
    if len(matches) < 4:
        print(f"Not enough matches to estimate homography: {len(matches)} < 4")
        return None, []
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
    if mask is None:
        print("No valid homography found")
        return None, []
    matches_mask = mask.ravel().tolist()
    inliers = [matches[i] for i, mask_bit in enumerate(matches_mask) if mask_bit]
    print(f"Found {sum(matches_mask)} inliers out of {len(matches)} matches")
    return H, inliers

def visualize_matches(img1, kp1, img2, kp2, matches, title="Matches"):
    if len(matches) > 100:
        matches = matches[:100]
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                              matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.jpg", dpi=300)
    plt.show()
    return match_img

def stitch_images(images, method="sift"):
    if len(images) < 2:
        print("Need at least 2 images to stitch")
        return None
    result = images[0]
    for i in range(1, len(images)):
        print(f"\nStitching image {i+1}/{len(images)}")
        img = images[i]
        kp1, kp2, matches = detect_and_match_features(result, img, method)
        visualize_matches(result, kp1, img, kp2, matches, f"Matches {i}")
        H, inliers = find_homography(kp1, kp2, matches)
        if H is None:
            print(f"Failed to find homography between result and image {i+1}")
            continue
        h1, w1 = result.shape[:2]
        h2, w2 = img.shape[:2]
        pts = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, H)
        dst_corners = np.concatenate((dst, np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)))
        [xmin, ymin] = np.int32(dst_corners.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(dst_corners.max(axis=0).ravel() + 0.5)
        t = [-xmin, -ymin]
        Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
        result_warped = cv2.warpPerspective(result, Ht.dot(H), (xmax-xmin, ymax-ymin))
        img_offset = np.zeros_like(result_warped)
        img_offset[t[1]:h2+t[1], t[0]:w2+t[0]] = img
        mask = np.zeros_like(result_warped)
        mask[t[1]:h2+t[1], t[0]:w2+t[0]] = 255
        result = np.where(mask == 0, result_warped, img_offset)
        cv2.imwrite(f"panorama_step_{i}.jpg", result)
    return result

def compare_methods(images):
    print("\n===== Stitching with SIFT =====")
    panorama_sift = stitch_images(images, "sift")
    if panorama_sift is not None:
        cv2.imwrite("panorama_sift.jpg", panorama_sift)
        plt.figure(figsize=(12, 6))
        plt.imshow(cv2.cvtColor(panorama_sift, cv2.COLOR_BGR2RGB))
        plt.title("Panorama with SIFT")
        plt.savefig("panorama_sift_plt.jpg", dpi=300)
        plt.show()
    print("\n===== Stitching with ORB =====")
    panorama_orb = stitch_images(images, "orb")
    if panorama_orb is not None:
        cv2.imwrite("panorama_orb.jpg", panorama_orb)
        plt.figure(figsize=(12, 6))
        plt.imshow(cv2.cvtColor(panorama_orb, cv2.COLOR_BGR2RGB))
        plt.title("Panorama with ORB")
        plt.savefig("panorama_orb_plt.jpg", dpi=300)
        plt.show()
    if panorama_sift is not None and panorama_orb is not None:
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 1, 1)
        plt.imshow(cv2.cvtColor(panorama_sift, cv2.COLOR_BGR2RGB))
        plt.title("Panorama with SIFT")
        plt.axis('off')
        plt.subplot(2, 1, 2)
        plt.imshow(cv2.cvtColor(panorama_orb, cv2.COLOR_BGR2RGB))
        plt.title("Panorama with ORB")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("panorama_comparison.jpg", dpi=300)
        plt.show()

def main():
    images = []
    folders = ["stit", "images", "."]
    for folder in folders:
        if os.path.exists(folder):
            images = load_images(folder)
            if len(images) >= 2:
                print(f"Found {len(images)} images in folder '{folder}'")
                break
    if len(images) < 2:
        print("No suitable images found. Creating test images...")
        images = create_test_images()
    compare_methods(images)
    print("\nImage stitching pipeline completed!")

if __name__ == "__main__":
    main()